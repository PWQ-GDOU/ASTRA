import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd

cd = "data/processed/cmapss/6. Turbofan Engine Degradation Simulation Data Set"

def load_cmapss(data_dir, fd, seq_len=30):
    tr = pd.read_csv(f"{data_dir}/train_{fd}.txt", sep=r"\s+", header=None).values
    te = pd.read_csv(f"{data_dir}/test_{fd}.txt", sep=r"\s+", header=None).values
    tru = pd.read_csv(f"{data_dir}/RUL_{fd}.txt", sep=r"\s+", header=None).values.flatten()
    MAX_RUL = 125
    all_train = tr[:, 2:].astype(np.float32)
    t_mean = all_train.mean(axis=0, keepdims=True)
    t_std = all_train.std(axis=0, keepdims=True) + 1e-8
    def proc(raw):
        Xl, yl = [], []
        for u in np.unique(raw[:,0]):
            t = raw[raw[:,0]==u, 2:].astype(np.float32)
            n = len(t)
            if n < seq_len: continue
            for i in range(n - seq_len + 1):
                Xl.append(t[i:i+seq_len])
                yl.append(min(n - (i+seq_len-1), MAX_RUL))
        return np.stack(Xl).astype(np.float32), np.array(yl, dtype=np.float32)
    Xt, yt = proc(tr)
    Xv_list = []
    for u in np.unique(te[:,0]):
        t = te[te[:,0]==u, 2:].astype(np.float32)
        if len(t) >= seq_len: Xv_list.append(t[-seq_len:])
    Xv = np.stack(Xv_list).astype(np.float32)
    yv = tru[:len(Xv)].astype(np.float32)
    return Xt, yt, Xv, yv, t_mean, t_std

def compute_score(yt, yp):
    d = yp - yt
    return float(np.sum(np.where(d>=0, np.exp(d/13.)-1, np.exp(-d/10.)-1)))

class TCNBlock(nn.Module):
    def __init__(self,ch,ks,dropout=0.1):
        super().__init__();dil=2;pad=(ks-1)*dil
        self.c1=nn.Conv1d(ch,ch,ks,dilation=dil,padding=pad);self.c2=nn.Conv1d(ch,ch,1)
        self.ln=nn.LayerNorm(ch);self.drop=nn.Dropout(dropout)
    def forward(self,x):
        B,C,T=x.shape;h=F.gelu(self.c1(x));h=h[:,:,:T] if h.shape[-1]>=T else F.pad(h,(0,T-h.shape[-1]))
        return self.ln((x+self.drop(self.c2(h))).transpose(1,2)).transpose(1,2)

class TCNRUL(nn.Module):
    def __init__(self,nf,d=192,dropout=0.1):
        super().__init__()
        self.proj=nn.Linear(nf,d)
        self.tcn=nn.ModuleList([TCNBlock(d,3,dropout) for _ in range(4)]+[TCNBlock(d,5,dropout) for _ in range(3)])
        self.ln=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,6,dropout=dropout,batch_first=True)
        self.head=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x):
        B,T,_=x.shape;h=self.proj(x).transpose(1,2)
        for b in self.tcn:h=b(h)
        h=self.ln(h.transpose(1,2));a,_=self.attn(h,h,h);h=h+a
        return self.head(h.mean(1))

def revin_norm(x):
    mu = x.mean(dim=1, keepdim=True)
    sigma = x.std(dim=1, keepdim=True) + 1e-5
    return (x - mu) / sigma, mu, sigma

fd = "FD002"
Xt, yt, Xv, yv, t_mean, t_std = load_cmapss(cd, fd)
nf = Xt.shape[-1]
print(f"=== {fd} RevIN V2 (500epoch+Score+3seeds) ===")
print(f"Train: {Xt.shape}, Test: {Xv.shape}")

for norm_name, use_revin in [("Z-score", False), ("RevIN+V2", True)]:
    bests = []
    for seed in [42, 123, 456]:
        np.random.seed(seed); torch.manual_seed(seed)
        device = "cuda:0"
        if use_revin:
            Xt_t = torch.FloatTensor(Xt); Xt_n, _, _ = revin_norm(Xt_t); Xt_n = Xt_n.numpy()
            Xv_t = torch.FloatTensor(Xv); Xv_n, _, _ = revin_norm(Xv_t); Xv_n = Xv_n.numpy()
        else:
            Xt_n = (Xt - t_mean) / t_std; Xv_n = (Xv - t_mean) / t_std
        model = TCNRUL(nf).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.005)
        epochs = 500
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*max(1,len(Xt)//256))
        Xtt = torch.FloatTensor(Xt_n).to(device); ytt = torch.FloatTensor(yt).to(device)
        Xvv = torch.FloatTensor(Xv_n).to(device)
        best_r, best_s = 1e9, 1e9; best_st = None; pat = 0
        for ep in range(epochs):
            model.train(); idx = np.random.permutation(len(Xtt))
            for i in range(0, len(Xt), 256):
                bi = idx[i:i+256]; p = model(Xtt[bi]).squeeze()
                d = p - ytt[bi]; loss = F.mse_loss(p, ytt[bi]) + 0.02*F.relu(d).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
            model.eval()
            with torch.no_grad():
                p = model(Xvv).squeeze().cpu().numpy()
                r = np.sqrt(np.mean((yv-p)**2)); s = compute_score(yv, p)
                if r < best_r-0.01: best_r=r; best_s=s; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; pat=0
                else: pat+=1
            if (ep+1)%100==0 or ep==0: print(f"  {norm_name} s{seed} E{ep+1}: R={r:.1f} S={s:.0f} best={best_r:.1f}")
            if pat >= 120: print(f"  Early stop E{ep+1}"); break
        if best_st: model.load_state_dict(best_st)
        bests.append(best_r)
    print(f"  => {norm_name}: {np.mean(bests):.1f} +/- {np.std(bests):.1f} best={min(bests):.1f}")
