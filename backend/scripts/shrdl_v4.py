"""
SHRDL-Pro v4: Multi-scale TCN + Attention + Ensemble-ready
Key: Parallel TCN branches with different receptive fields
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, math, argparse
from shrdl_pro_v2 import combined_loss, compute_score, load_cmapss_std

class Chomp1d(nn.Module):
    def __init__(self, cs):super().__init__();self.cs=cs
    def forward(self,x):return x[:,:,:-self.cs].contiguous() if self.cs>0 else x

class TCNBranch(nn.Module):
    """Single TCN branch with specific kernel size."""
    def __init__(self, in_ch, out_ch, kernel_size, n_blocks=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            dil = 2**i
            pad = (kernel_size-1)*dil
            self.blocks.append(nn.Sequential(
                nn.Conv1d(in_ch if i==0 else out_ch, out_ch, kernel_size, dilation=dil, padding=pad),
                Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dil, padding=pad),
                Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            ))
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else None
        self.norm = nn.LayerNorm(out_ch)
    
    def forward(self, x):
        # x: (B, D, T)
        res = x if self.down is None else self.down(x)
        for block in self.blocks:
            x = block(x)
        x = x + res
        x = self.norm(x.transpose(1,2)).transpose(1,2)
        return F.relu(x)

class MultiScaleTCN(nn.Module):
    """Multi-scale TCN: 3 parallel branches with different kernel sizes."""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        # Small kernel: captures fast/local changes
        self.small = TCNBranch(in_ch, out_ch//3, kernel_size=3, dropout=dropout)
        # Medium kernel: mid-range patterns
        self.medium = TCNBranch(in_ch, out_ch//3, kernel_size=5, dropout=dropout)
        # Large kernel: slow/global trends
        self.large = TCNBranch(in_ch, out_ch//3, kernel_size=7, dropout=dropout)
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1),
            nn.ReLU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(out_ch)
    
    def forward(self, x):
        s = self.small(x)
        m = self.medium(x)
        l = self.large(x)
        fused = torch.cat([s,m,l], dim=1)  # (B, out_ch, T)
        out = self.fusion(fused)
        return self.norm(out.transpose(1,2)).transpose(1,2)

class AttentionBlock(nn.Module):
    def __init__(self, d=192, nh=6, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nh, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Dropout(dropout),nn.Linear(d*4,d))
        self.drop = nn.Dropout(dropout)
    def forward(self,x):
        a,_ = self.attn(x,x,x); x = self.n1(x+self.drop(a))
        return self.n2(x+self.drop(self.ffn(x)))

class SHRDLProV4(nn.Module):
    """Multi-scale TCN + Attention for RUL prediction."""
    def __init__(self, n_features, d_model=192, n_heads=6, attn_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.ms_tcn = MultiScaleTCN(d_model, d_model, dropout)
        self.attn = nn.ModuleList([AttentionBlock(d_model,n_heads,dropout) for _ in range(attn_layers)])
        pe = torch.zeros(1000,d_model)
        pos = torch.arange(0,1000).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.pe = nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        B,T,_ = x.shape
        h = self.input_proj(x).transpose(1,2)  # (B,D,T)
        h = self.ms_tcn(h)
        h = h.transpose(1,2) + self.pe[:,:T,:]  # (B,T,D)
        for a in self.attn: h = a(h)
        return h.mean(dim=1)

class RULPredictorV4(nn.Module):
    def __init__(self, nf, d=192, nh=6, al=2, dropout=0.1):
        super().__init__()
        self.enc = SHRDLProV4(nf,d,nh,al,dropout)
        self.head = nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Dropout(dropout),
                                   nn.Linear(128,64),nn.GELU(),nn.Dropout(dropout),
                                   nn.Linear(64,1))
    def forward(self,x):return self.head(self.enc(x))

def train_v4(model,Xt,yt,Xv,yv,epochs=200,lr=5e-4,wd=3e-3,alpha=0.3,bs=256,device='cuda'):
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    warmup=10; spe=max(1,len(Xt)//bs)
    def lr_fn(step):
        if step<warmup*spe: return step/(warmup*spe)
        p=(step-warmup*spe)/(epochs*spe-warmup*spe);return 0.5*(1+math.cos(math.pi*p))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_fn)
    Xtt=torch.FloatTensor(Xt).to(device);ytt=torch.FloatTensor(yt).to(device)
    Xvv=torch.FloatTensor(Xv).to(device)
    best_rmse,best_score=1e9,1e9;best_state=None
    for ep in range(epochs):
        idx=np.random.permutation(len(Xtt));model.train()
        for i in range(0,len(Xt),bs):
            bi=idx[i:i+bs];pred=model(Xtt[bi]).squeeze()
            loss=combined_loss(pred,ytt[bi],alpha)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step();sched.step()
        model.eval()
        with torch.no_grad():
            p=model(Xvv).squeeze().cpu().numpy()
            rmse=np.sqrt(np.mean((yv-p)**2));score=compute_score(yv,p)
            if rmse<best_rmse:best_rmse=rmse;best_score=score;best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        if (ep+1)%30==0 or ep==0:print(f'  E{ep+1:3d}: RMSE={rmse:.1f} Score={score:.0f} best={best_rmse:.1f}')
    if best_state:model.load_state_dict(best_state)
    return best_rmse,best_score

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda:1');p.add_argument('--fd',default='FD002')
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--wd',type=float,default=3e-3);p.add_argument('--alpha',type=float,default=0.3)
    args=p.parse_args()
    device=args.device
    for fd in ([args.fd] if args.fd!='all' else ['FD002','FD004']):
        print(f'\n{"="*60}\n  {fd}: V4 Multi-scale TCN\n{"="*60}')
        Xt,yt,Xv,yv,nf=load_cmapss_std('data/processed',fd)
        print(f'  Train:{Xt.shape} Test:{Xv.shape}')
        m=RULPredictorV4(nf).to(device);print(f'  Params:{sum(p.numel() for p in m.parameters()):,}')
        r,s=train_v4(m,Xt,yt,Xv,yv,epochs=args.epochs,lr=args.lr,wd=args.wd,alpha=args.alpha,device=device)
        print(f'  => RMSE={r:.1f} Score={s:.0f}  (V2 was ~54)')
