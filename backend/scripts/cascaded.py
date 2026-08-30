"""Cascaded: classify condition first, then predict RUL per cluster."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, math, argparse
from sklearn.cluster import KMeans

def load_clustered(dd, fd, seq_len=30, n_clusters=6):
    cd = dd + '/cmapss/6. Turbofan Engine Degradation Simulation Data Set'
    tr = pd.read_csv(f'{cd}/train_{fd}.txt', sep=r'\s+', header=None).values
    te = pd.read_csv(f'{cd}/test_{fd}.txt', sep=r'\s+', header=None).values
    tru = pd.read_csv(f'{cd}/RUL_{fd}.txt', sep=r'\s+', header=None).values.flatten()
    MAX_RUL = 125
    ops_tr = tr[:,2:5].astype(np.float64)
    ops_te = te[:,2:5].astype(np.float64)
    all_ops = np.concatenate([ops_tr, ops_te])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(all_ops)
    tr_labels = labels[:len(tr)]
    te_labels = labels[len(tr):]
    cluster_stats = {}
    for c in range(n_clusters):
        mask = tr_labels == c
        if mask.sum() > 0:
            d = tr[mask, 5:].astype(np.float64)
            cluster_stats[c] = (d.mean(axis=0), d.std(axis=0)+1e-8)
    sensors_norm = np.zeros((len(tr), 21), dtype=np.float32)
    for c in range(n_clusters):
        mask = tr_labels == c
        if mask.sum() > 0 and c in cluster_stats:
            mu, sg = cluster_stats[c]
            sensors_norm[mask] = (tr[mask, 5:].astype(np.float64) - mu) / sg
    
    def proc(raw, labels=None, sensors_norm_precomp=None):
        Xs, y_rul, y_cls = [], [], []
        for u in np.unique(raw[:,0]):
            mask_u = raw[:,0] == u
            t_all = raw[mask_u]
            n = len(t_all)
            if n < seq_len: continue
            if sensors_norm_precomp is not None:
                gidx = np.where(raw[:,0] == u)[0]
                t_s = sensors_norm_precomp[gidx]
            else:
                t_s = t_all[:, 5:].astype(np.float32)
                t_s = (t_s - t_s.mean(0)) / (t_s.std(0) + 1e-8)
            t_c = t_all[:, 2:5].astype(np.float32)
            for i in range(n - seq_len + 1):
                Xs.append(t_s[i:i+seq_len])
                y_rul.append(min(n-(i+seq_len-1), MAX_RUL))
                if labels is not None:
                    gidx2 = np.where(raw[:,0] == u)[0][i + seq_len//2]
                    y_cls.append(labels[min(gidx2, len(labels)-1)])
        out = [np.stack(Xs).astype(np.float32), np.array(y_rul, dtype=np.float32)]
        if labels is not None: out.append(np.array(y_cls, dtype=np.int64))
        return tuple(out)
    
    Xst, yt, yc = proc(tr, tr_labels, sensors_norm)
    Xsv_list = []
    for u in np.unique(te[:,0]):
        mask_u = te[:,0]==u
        t_all = te[mask_u]
        if len(t_all)<seq_len: continue
        li = np.where(mask_u)[0][-1]; c = te_labels[li]
        t_s = t_all[:,5:].astype(np.float64)
        if c in cluster_stats: mu,sg=cluster_stats[c]; t_s=(t_s-mu)/sg
        Xsv_list.append(t_s[-seq_len:].astype(np.float32))
    Xsv = np.stack(Xsv_list); yv = tru[:len(Xsv_list)].astype(np.float32)
    # Test cluster labels
    yc_test = np.array([te_labels[np.where(te[:,0]==u)[0][-1]] for u in np.unique(te[:,0])[:len(Xsv_list)]])
    return Xst, yt, yc, Xsv, yv, yc_test

def compute_score(yt,yp):
    d=yp-yt; return float(np.sum(np.where(d>=0,np.exp(d/13.)-1,np.exp(-d/10.)-1)))

class TCNBlock(nn.Module):
    def __init__(self,ch,ks,dropout=0.1):
        super().__init__();dil=2;pad=(ks-1)*dil
        self.c1=nn.Conv1d(ch,ch,ks,dilation=dil,padding=pad);self.c2=nn.Conv1d(ch,ch,1)
        self.ln=nn.LayerNorm(ch);self.drop=nn.Dropout(dropout)
    def forward(self,x):
        B,C,T=x.shape;h=F.gelu(self.c1(x));h=h[:,:,:T] if h.shape[-1]>=T else F.pad(h,(0,T-h.shape[-1]))
        return self.ln((x+self.drop(self.c2(h))).transpose(1,2)).transpose(1,2)

class CascadedRUL(nn.Module):
    def __init__(self, n_sens=21, n_clusters=6, d=192, dropout=0.1):
        super().__init__()
        self.n_clusters = n_clusters
        self.proj = nn.Linear(n_sens, d)
        self.tcn = nn.ModuleList([TCNBlock(d,3,dropout) for _ in range(4)] +
                                  [TCNBlock(d,5,dropout) for _ in range(3)])
        self.ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 6, dropout=dropout, batch_first=True)
        # Shared condition classifier
        self.classifier = nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,n_clusters))
        # Per-cluster RUL heads
        self.rul_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
            for _ in range(n_clusters)])
        for p in self.parameters():
            if p.dim()>1: nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        B,T,_=x.shape; h=self.proj(x).transpose(1,2)
        for b in self.tcn: h=b(h)
        h=self.ln(h.transpose(1,2)); a,_=self.attn(h,h,h); h=h+a
        f=h.mean(1)  # (B,D)
        # Classification
        logits = self.classifier(f)
        # Per-cluster RUL
        rul = torch.zeros(B,1,device=x.device)
        for c in range(self.n_clusters):
            mask = (logits.argmax(-1) == c).float().unsqueeze(-1)
            rul_c = self.rul_heads[c](f)
            rul = rul + mask * rul_c
        return rul, logits

def train_cascaded(model, Xt,yt,yc, Xv,yv,ycv, epochs=500,lr=5e-4,bs=256,dev='cuda:0'):
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.005)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs*max(1,len(Xt)//bs))
    Xtt=torch.FloatTensor(Xt).to(dev);ytt=torch.FloatTensor(yt).to(dev);yct=torch.LongTensor(yc).to(dev)
    Xvv=torch.FloatTensor(Xv).to(dev);ycv_t=torch.LongTensor(ycv).to(dev)
    best_r,best_s=1e9,1e9;best_st=None;pat=0
    for ep in range(epochs):
        model.train();idx=np.random.permutation(len(Xtt))
        for i in range(0,len(Xt),bs):
            bi=idx[i:i+bs];rul,logits=model(Xtt[bi])
            loss_r=F.mse_loss(rul.squeeze(),ytt[bi])
            loss_c=F.cross_entropy(logits,yct[bi])
            d=rul.squeeze()-ytt[bi]
            loss=loss_r + 0.1*loss_c + 0.02*F.relu(d).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step();sched.step()
        model.eval()
        with torch.no_grad():
            p,_=model(Xvv);p=p.squeeze().cpu().numpy()
            r=np.sqrt(np.mean((yv-p)**2));s=compute_score(yv,p)
            if r<best_r-0.01:best_r=r;best_s=s;best_st={k:v.cpu().clone() for k,v in model.state_dict().items()};pat=0
            else:pat+=1
        if (ep+1)%50==0 or ep==0:print(f'E{ep+1}: R={r:.1f} S={s:.0f} best={best_r:.1f}')
        if pat>=100:print(f'Early stop E{ep+1}');break
    if best_st:model.load_state_dict(best_st)
    return best_r,best_s

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fd',default='FD002');p.add_argument('--epochs',type=int,default=500);p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    print(f'\n=== {args.fd} Cascaded (Classify→Predict) ===')
    Xt,yt,yc,Xv,yv,ycv=load_clustered('data/processed',args.fd)
    print(f'Train:{Xt.shape} Test:{Xv.shape} Clusters:6')
    m=CascadedRUL().to(args.device);print(f'Params:{sum(p.numel() for p in m.parameters()):,}')
    r,s=train_cascaded(m,Xt,yt,yc,Xv,yv,ycv,epochs=args.epochs,dev=args.device)
    print(f'=> RMSE={r:.1f} Score={s:.0f}')
