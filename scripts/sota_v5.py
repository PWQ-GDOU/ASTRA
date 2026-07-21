import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, math, argparse
from sklearn.cluster import KMeans

def load_percluster(dd, fd, seq_len=30, n_clusters=6):
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
    
    def proc(raw, labels, sensors_norm=None, is_train=True):
        Xs, Xc, yl = [], [], []
        for u in np.unique(raw[:,0]):
            mask_u = raw[:,0] == u
            t_all = raw[mask_u]
            n = len(t_all)
            if n < seq_len: continue
            if is_train and sensors_norm is not None:
                global_indices = np.where(raw[:,0] == u)[0]
                t_s = sensors_norm[global_indices]
            else:
                t_s = t_all[:, 5:].astype(np.float32)
                t_s = (t_s - t_s.mean(axis=0)) / (t_s.std(axis=0) + 1e-8)
            t_c = t_all[:, 2:5].astype(np.float32)
            for i in range(n - seq_len + 1):
                Xs.append(t_s[i:i+seq_len])
                Xc.append(t_c[i:i+seq_len])
                yl.append(min(n - (i+seq_len-1), MAX_RUL))
        return np.stack(Xs).astype(np.float32), np.stack(Xc).astype(np.float32), np.array(yl, dtype=np.float32)
    
    Xst, Xct, yt = proc(tr, tr_labels, sensors_norm, True)
    Xsv, Xcv = [], []
    for u in np.unique(te[:,0]):
        mask_u = te[:,0] == u
        t_all = te[mask_u]
        if len(t_all) < seq_len: continue
        last_idx = np.where(mask_u)[0][-1]
        c = te_labels[last_idx]
        t_s = t_all[:, 5:].astype(np.float64)
        if c in cluster_stats:
            mu, sg = cluster_stats[c]
            t_s = (t_s - mu) / sg
        t_s = t_s.astype(np.float32)
        t_c = t_all[:, 2:5].astype(np.float32)
        Xsv.append(t_s[-seq_len:]); Xcv.append(t_c[-seq_len:])
    Xsv = np.stack(Xsv).astype(np.float32); Xcv = np.stack(Xcv).astype(np.float32)
    yv = tru[:len(Xsv)].astype(np.float32)
    return Xst, Xct, yt, Xsv, Xcv, yv

def compute_score(yt,yp):
    d=yp-yt; return float(np.sum(np.where(d>=0,np.exp(d/13.)-1,np.exp(-d/10.)-1)))

class ECA(nn.Module):
    def __init__(self, ch, k=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = x.mean(-1, keepdim=True)
        y = y.transpose(-1,-2)
        y = self.conv(y)
        y = y.transpose(-1,-2)
        return x * self.sigmoid(y)

class TCNBlock(nn.Module):
    def __init__(self, ch, ks, dropout=0.1):
        super().__init__()
        dil=2; pad=(ks-1)*dil
        self.conv1 = nn.Conv1d(ch, ch, ks, dilation=dil, padding=pad)
        self.conv2 = nn.Conv1d(ch, ch, 1)
        self.eca = ECA(ch)
        self.ln = nn.LayerNorm(ch)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,C,T = x.shape
        h = F.gelu(self.conv1(x))
        h = h[:,:,:T] if h.shape[-1]>=T else F.pad(h,(0,T-h.shape[-1]))
        h = x + self.dropout(self.conv2(h))
        h = self.eca(h)
        return self.ln(h.transpose(1,2)).transpose(1,2)

class DualECARUL(nn.Module):
    def __init__(self, n_sens=21, n_cond=3, d=256, dropout=0.1):
        super().__init__()
        self.s_proj = nn.Linear(n_sens, d)
        self.s_tcn = nn.ModuleList([TCNBlock(d,3,dropout) for _ in range(4)] +
                                    [TCNBlock(d,5,dropout) for _ in range(4)])
        self.s_ln = nn.LayerNorm(d)
        self.c_proj = nn.Linear(n_cond, d)
        self.c_mlp = nn.Sequential(nn.Linear(d,d*2), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(d*2,d), nn.LayerNorm(d))
        self.cross_attn = nn.MultiheadAttention(d, 8, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, 8, dropout=dropout, batch_first=True)
        self.self_ln = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1))
        for p in self.parameters():
            if p.dim()>1: nn.init.xavier_uniform_(p)
    def forward(self, x_sens, x_cond):
        B,T,_ = x_sens.shape
        hs = self.s_proj(x_sens).transpose(1,2)
        for tcn in self.s_tcn: hs = tcn(hs)
        hs = self.s_ln(hs.transpose(1,2))
        hc = self.c_proj(x_cond)
        hc = self.c_mlp(hc)
        ca, _ = self.cross_attn(hs, hc, hc)
        h = self.attn_ln(hs + ca)
        sa, _ = self.self_attn(h, h, h)
        h = self.self_ln(h + sa)
        return self.head(h.mean(1))

def train_model(model, Xst,Xct,yt,Xsv,Xcv,yv, epochs=500, lr=5e-4, bs=256, dev='cuda:0'):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*max(1,len(Xst)//bs))
    Xstt=torch.FloatTensor(Xst).to(dev); Xctt=torch.FloatTensor(Xct).to(dev); ytt=torch.FloatTensor(yt).to(dev)
    Xsvv=torch.FloatTensor(Xsv).to(dev); Xcvv=torch.FloatTensor(Xcv).to(dev)
    best_r,best_s=1e9,1e9;best_st=None;pat=0
    for ep in range(epochs):
        model.train(); idx=np.random.permutation(len(Xstt))
        for i in range(0,len(Xst),bs):
            bi=idx[i:i+bs]; p=model(Xstt[bi],Xctt[bi]).squeeze()
            d=p-ytt[bi]; loss=F.mse_loss(p,ytt[bi])+0.02*F.relu(d).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step();sched.step()
        model.eval()
        with torch.no_grad():
            p=model(Xsvv,Xcvv).squeeze().cpu().numpy()
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
    for fd in ([args.fd] if args.fd!='all' else ['FD002','FD004']):
        print(f'\n=== {fd} DualECA (v5) {args.epochs}epoch ===')
        Xst,Xct,yt,Xsv,Xcv,yv=load_percluster('data/processed',fd)
        print(f'Train:{Xst.shape} Test:{Xsv.shape}')
        m=DualECARUL().to(args.device);print(f'Params:{sum(p.numel() for p in m.parameters()):,}')
        r,s=train_model(m,Xst,Xct,yt,Xsv,Xcv,yv,epochs=args.epochs,dev=args.device)
        print(f'=> RMSE={r:.1f} Score={s:.0f}')
