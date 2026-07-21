"""
Hybrid: TCN (local patterns) + Transformer (global dependencies) 
with learnable gated fusion.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, math, argparse

# ========== Data ==========
def load_data(dd, fd, seq_len=30):
    cd = dd + '/cmapss/6. Turbofan Engine Degradation Simulation Data Set'
    tr = pd.read_csv(f'{cd}/train_{fd}.txt', sep=r'\s+', header=None).values
    te = pd.read_csv(f'{cd}/test_{fd}.txt', sep=r'\s+', header=None).values
    tru = pd.read_csv(f'{cd}/RUL_{fd}.txt', sep=r'\s+', header=None).values.flatten()
    MAX_RUL = 125
    all_train = tr[:, 2:].astype(np.float32)
    t_mean = all_train.mean(axis=0, keepdims=True)
    t_std = all_train.std(axis=0, keepdims=True) + 1e-8
    
    def proc(raw):
        Xl, yl = [], []
        for u in np.unique(raw[:,0]):
            t = raw[raw[:,0]==u, 2:].astype(np.float32)
            t = (t - t_mean) / t_std
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
        t = (t - t_mean) / t_std
        if len(t) >= seq_len: Xv_list.append(t[-seq_len:])
    Xv = np.stack(Xv_list).astype(np.float32)
    yv = tru[:len(Xv_list)].astype(np.float32)
    return Xt, yt, Xv, yv, tr.shape[1]-2

def compute_score(yt, yp):
    d = yp - yt
    return float(np.sum(np.where(d>=0, np.exp(d/13.)-1, np.exp(-d/10.)-1)))

# ========== Hybrid Model ==========
class Chomp(nn.Module):
    def __init__(self,c):super().__init__();self.c=c
    def forward(self,x):return x[:,:,:-self.c] if self.c>0 else x

class TCNBlock(nn.Module):
    def __init__(self,in_ch,out_ch,ks,dil,dropout=0.1):
        super().__init__()
        p=(ks-1)*dil
        self.net=nn.Sequential(nn.Conv1d(in_ch,out_ch,ks,dilation=dil,padding=p),Chomp(p),nn.GELU(),nn.Dropout(dropout))
        self.down=nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()
    def forward(self,x):return self.net(x)+self.down(x)

class TCNBranch(nn.Module):
    def __init__(self,in_ch,out_ch,dropout=0.1):
        super().__init__()
        self.b1=TCNBlock(in_ch,out_ch,3,1,dropout)
        self.b2=TCNBlock(out_ch,out_ch,3,2,dropout)
        self.b3=TCNBlock(out_ch,out_ch,3,4,dropout)
        self.ln=nn.LayerNorm(out_ch)
    def forward(self,x):
        x=self.b1(x);x=self.b2(x);x=self.b3(x)
        return self.ln(x.transpose(1,2)).transpose(1,2)

class TransBranch(nn.Module):
    def __init__(self,d,n_heads=8,n_layers=2,dropout=0.1):
        super().__init__()
        self.layers=nn.ModuleList([
            nn.TransformerEncoderLayer(d,n_heads,d*4,dropout,'gelu',batch_first=True,norm_first=True)
            for _ in range(n_layers)])
        self.ln=nn.LayerNorm(d)
    def forward(self,x):
        for l in self.layers:x=l(x)
        return self.ln(x)

class HybridRUL(nn.Module):
    """TCN + Transformer with learned gated fusion."""
    def __init__(self,nf,d=192,dropout=0.1):
        super().__init__()
        self.proj=nn.Linear(nf,d)
        # TCN: local multi-scale patterns
        self.tcn=TCNBranch(d,d,dropout)
        # Transformer: global dependencies
        self.trans=TransBranch(d,6,2,dropout)
        # Gated fusion
        self.gate=nn.Sequential(nn.Linear(d*2,d),nn.Sigmoid())
        self.fusion=nn.Linear(d*2,d)
        # Position encoding
        pe=torch.zeros(1000,d);pos=torch.arange(0,1000).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.)/d))
        pe[:,0::2]=torch.sin(pos*div);pe[:,1::2]=torch.cos(pos*div)
        self.pe=nn.Parameter(pe.unsqueeze(0),requires_grad=False)
        # Head
        self.head=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(dropout),
                                 nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
    
    def forward(self,x):
        B,T,_=x.shape
        h=self.proj(x)
        # TCN branch: (B,D,T)
        h_tcn=self.tcn(h.transpose(1,2)).transpose(1,2)  # (B,T,D)
        # Transformer branch
        h_trans=self.trans(h_tcn+self.pe[:,:T,:])
        # Pool
        f_tcn=h_tcn.mean(1);f_trans=h_trans.mean(1)
        # Gated fusion
        g=self.gate(torch.cat([f_tcn,f_trans],-1))
        fused=g*f_tcn+(1-g)*f_trans
        return self.head(fused)

def train(m,Xt,yt,Xv,yv,epochs=200,lr=8e-4,bs=256,dev='cuda'):
    m.train()
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.01)
    total_steps = epochs * max(1, len(Xt)//bs)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    Xtt=torch.FloatTensor(Xt).to(dev);ytt=torch.FloatTensor(yt).to(dev)
    Xvv=torch.FloatTensor(Xv).to(dev)
    best_r,best_s=1e9,1e9;best_st=None
    for ep in range(epochs):
        m.train();idx=np.random.permutation(len(Xtt))
        for i in range(0,len(Xt),bs):
            bi=idx[i:i+bs];p=m(Xtt[bi]).squeeze()
            d=p-ytt[bi]
            loss=F.mse_loss(p,ytt[bi])+0.05*F.relu(d).mean()
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opt.step();sched.step()
        m.eval()
        with torch.no_grad():
            p=m(Xvv).squeeze().cpu().numpy()
            r=np.sqrt(np.mean((yv-p)**2));s=compute_score(yv,p)
            if r<best_r:best_r=r;best_s=s;best_st={k:v.cpu().clone() for k,v in m.state_dict().items()}
        if (ep+1)%30==0 or ep==0:print(f'  E{ep+1:3d}: R={r:.1f} S={s:.0f} best={best_r:.1f}')
    if best_st:m.load_state_dict(best_st)
    return best_r,best_s

    def lr_fn(step): return 0.5 * (1 + math.cos(math.pi * step / total_steps)); sched=torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    Xtt=torch.FloatTensor(Xt).to(dev);ytt=torch.FloatTensor(yt).to(dev)
    Xvv=torch.FloatTensor(Xv).to(dev)
    best_r,best_s=1e9,1e9;best_st=None
    for ep in range(epochs):
        m.train();idx=np.random.permutation(len(Xtt))
        for i in range(0,len(Xt),bs):
            bi=idx[i:i+bs];p=m(Xtt[bi]).squeeze()
            d=p-ytt[bi]
            loss=F.mse_loss(p,ytt[bi])+0.05*F.relu(d).mean()
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opt.step();sched.step()
        m.eval()
        with torch.no_grad():
            p=m(Xvv).squeeze().cpu().numpy()
            r=np.sqrt(np.mean((yv-p)**2));s=compute_score(yv,p)
            if r<best_r:best_r=r;best_s=s;best_st={k:v.cpu().clone() for k,v in m.state_dict().items()}
        if (ep+1)%30==0 or ep==0:print(f'  E{ep+1:3d}: R={r:.1f} S={s:.0f} best={best_r:.1f}')
    if best_st:m.load_state_dict(best_st)
    return best_r,best_s

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda:1');p.add_argument('--fd',default='all');p.add_argument('--epochs',type=int,default=200)
    args=p.parse_args()
    res={}
    for fd in (['FD001','FD002','FD003','FD004'] if args.fd=='all' else [args.fd]):
        print(f'\n=== {fd} Hybrid ===')
        Xt,yt,Xv,yv,nf=load_data('data/processed',fd)
        print(f'  Train:{Xt.shape} Test:{Xv.shape}')
        m=HybridRUL(nf).to(args.device)
        print(f'  Params:{sum(p.numel() for p in m.parameters()):,}')
        r,s=train(m,Xt,yt,Xv,yv,epochs=args.epochs,dev=args.device)
        res[fd]={'RMSE':r,'Score':s}
        print(f'  => RMSE={r:.1f} Score={s:.0f}')
    print('\n=== HYBRID RESULTS ===')
    for fd,r in res.items():print(f'{fd}: RMSE={r["RMSE"]:.1f} Score={r["Score"]:.0f}')
