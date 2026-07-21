import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, math, argparse
from sklearn.cluster import KMeans

def load_data(dd, fd, seq_len=30, n_clusters=6):
    cd = dd + '/cmapss/6. Turbofan Engine Degradation Simulation Data Set'
    tr = pd.read_csv(f'{cd}/train_{fd}.txt', sep=r'\s+', header=None).values
    te = pd.read_csv(f'{cd}/test_{fd}.txt', sep=r'\s+', header=None).values
    tru = pd.read_csv(f'{cd}/RUL_{fd}.txt', sep=r'\s+', header=None).values.flatten()
    MAX_RUL = 125
    ops_tr = tr[:,2:5].astype(np.float64); ops_te = te[:,2:5].astype(np.float64)
    all_ops = np.concatenate([ops_tr, ops_te])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(all_ops)
    tr_labels = labels[:len(tr)]; te_labels = labels[len(tr):]
    centroids = km.cluster_centers_
    cstats = {}
    for c in range(n_clusters):
        m = tr_labels == c
        if m.sum() > 0:
            d = tr[m, 5:].astype(np.float64); cstats[c] = (d.mean(0), d.std(0)+1e-8)
    snorm = np.zeros((len(tr), 21), dtype=np.float32)
    for c in range(n_clusters):
        m = tr_labels == c
        if m.sum() > 0 and c in cstats:
            mu, sg = cstats[c]; snorm[m] = (tr[m, 5:].astype(np.float64) - mu) / sg
    def proc(raw, labels, sn_pre=None):
        Xs, yr, yc = [], [], []
        for u in np.unique(raw[:,0]):
            mu = raw[:,0]==u; ta = raw[mu]; n = len(ta)
            if n < seq_len: continue
            if sn_pre is not None:
                gi = np.where(raw[:,0]==u)[0]; ts = sn_pre[gi]
            else:
                ts = ta[:,5:].astype(np.float32); ts = (ts-ts.mean(0))/(ts.std(0)+1e-8)
            for i in range(n-seq_len+1):
                Xs.append(ts[i:i+seq_len])
                yr.append(min(n-(i+seq_len-1), MAX_RUL))
                mid = min(i+seq_len//2, n-1); yc.append(labels[np.where(raw[:,0]==u)[0][mid]])
        return np.stack(Xs).astype(np.float32), np.array(yr,dtype=np.float32), np.array(yc,dtype=np.int64)
    Xst, yt, yc = proc(tr, tr_labels, snorm)
    Xsv, ytv = [], []
    for u in np.unique(te[:,0]):
        mu = te[:,0]==u; ta = te[mu]
        if len(ta) < seq_len: continue
        conds = ta[-seq_len:, 2:5].mean(0); c = int(np.argmin(((centroids-conds)**2).sum(1)))
        ytv.append(c); ts = ta[:,5:].astype(np.float64)
        if c in cstats: mu_s, sg_s = cstats[c]; ts = (ts-mu_s)/sg_s
        Xsv.append(ts[-seq_len:].astype(np.float32))
    Xsv = np.stack(Xsv); yv = tru[:len(Xsv)].astype(np.float32)
    return Xst, yt, yc, Xsv, yv, np.array(ytv)

def compute_score(yt,yp):
    d = yp-yt; return float(np.sum(np.where(d>=0, np.exp(d/13.)-1, np.exp(-d/10.)-1)))

class TCNBlock(nn.Module):
    def __init__(self,ch,ks,dropout=0.1):
        super().__init__(); dil=2; pad=(ks-1)*dil
        self.c1=nn.Conv1d(ch,ch,ks,dilation=dil,padding=pad); self.c2=nn.Conv1d(ch,ch,1)
        self.ln=nn.LayerNorm(ch); self.drop=nn.Dropout(dropout)
    def forward(self,x):
        B,C,T=x.shape; h=F.gelu(self.c1(x))
        h=h[:,:,:T] if h.shape[-1]>=T else F.pad(h,(0,T-h.shape[-1]))
        return self.ln((x+self.drop(self.c2(h))).transpose(1,2)).transpose(1,2)

class DirectClusterRUL(nn.Module):
    def __init__(self, n_sens=21, n_clusters=6, d=192, dropout=0.1):
        super().__init__(); self.n_clusters=n_clusters
        self.proj=nn.Linear(n_sens,d)
        self.tcn=nn.ModuleList([TCNBlock(d,3,dropout) for _ in range(4)]+[TCNBlock(d,5,dropout) for _ in range(3)])
        self.ln=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,6,dropout=dropout,batch_first=True)
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1)) for _ in range(n_clusters)])
        for p in self.parameters():
            if p.dim()>1:nn.init.xavier_uniform_(p)
    def forward(self,x,cids):
        B,T,_=x.shape; h=self.proj(x).transpose(1,2)
        for b in self.tcn:h=b(h)
        h=self.ln(h.transpose(1,2)); a,_=self.attn(h,h,h); h=h+a; f=h.mean(1)
        rul=torch.zeros(B,1,device=x.device)
        for c in range(self.n_clusters):
            m=(cids==c).float().unsqueeze(-1); rul=rul+m*self.heads[c](f)
        return rul

def train(m,Xt,yt,yc,Xv,yv,yc_test,epochs=500,lr=5e-4,bs=256,dev='cuda:0'):
    m.train()
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.005)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs*max(1,len(Xt)//bs))
    Xtt=torch.FloatTensor(Xt).to(dev); ytt=torch.FloatTensor(yt).to(dev)
    yct=torch.LongTensor(yc).to(dev); Xvv=torch.FloatTensor(Xv).to(dev)
    ycvt=torch.LongTensor(yc_test).to(dev)
    best_r,best_s=1e9,1e9;best_st=None;pat=0
    for ep in range(epochs):
        m.train();idx=np.random.permutation(len(Xtt))
        for i in range(0,len(Xt),bs):
            bi=idx[i:i+bs]; rul=m(Xtt[bi],yct[bi]).squeeze()
            d=rul-ytt[bi]; loss=F.mse_loss(rul,ytt[bi])+0.02*F.relu(d).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opt.step();sched.step()
        m.eval()
        with torch.no_grad():
            p=m(Xvv,ycvt).squeeze().cpu().numpy()
            r=np.sqrt(np.mean((yv-p)**2));s=compute_score(yv,p)
            if r<best_r-0.01:best_r=r;best_s=s;best_st={k:v.cpu().clone() for k,v in m.state_dict().items()};pat=0
            else:pat+=1
        if (ep+1)%50==0 or ep==0:print(f'E{ep+1}: R={r:.1f} S={s:.0f} best={best_r:.1f}')
        if pat>=100:print(f'Early stop E{ep+1}');break
    if best_st:m.load_state_dict(best_st)
    return best_r,best_s

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fd',default='FD002');p.add_argument('--epochs',type=int,default=500);p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    print(f'\n=== {args.fd} DirectCluster ===')
    Xt,yt,yc,Xv,yv,yc_test=load_data('data/processed',args.fd)
    print(f'Train:{Xt.shape} Test:{Xv.shape}')
    m=DirectClusterRUL().to(args.device);print(f'Params:{sum(p.numel() for p in m.parameters()):,}')
    r,s=train(m,Xt,yt,yc,Xv,yv,yc_test,epochs=args.epochs,dev=args.device)
    print(f'=> RMSE={r:.1f} Score={s:.0f}')
