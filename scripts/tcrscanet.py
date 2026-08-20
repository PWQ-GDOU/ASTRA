import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, math, argparse

def load_data(dd, fd, seq_len=30):
    cd = dd + '/cmapss/6. Turbofan Engine Degradation Simulation Data Set'
    tr = pd.read_csv(cd+'/train_'+fd+'.txt', sep=r'\s+', header=None).values
    te = pd.read_csv(cd+'/test_'+fd+'.txt', sep=r'\s+', header=None).values
    tru = pd.read_csv(cd+'/RUL_'+fd+'.txt', sep=r'\s+', header=None).values.flatten()
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

class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.se = nn.Sequential(nn.Linear(ch,ch//r),nn.GELU(),nn.Linear(ch//r,ch),nn.Sigmoid())
    def forward(self,x):
        B,C,T=x.shape; w=self.se(x.mean(-1)).view(B,C,1); return x*w

class MSBlock(nn.Module):
    def __init__(self,ch,ks,dropout=0.1):
        super().__init__()
        super().__init__()
        dil=2; pad=(ks-1)*dil
        self.conv=nn.Sequential(nn.Conv1d(ch,ch,ks,dilation=dil,padding=pad),nn.GELU(),nn.Dropout(dropout))
        self.se=SEBlock(ch); self.ln=nn.LayerNorm(ch)
    def forward(self,x):
        B,C,T=x.shape; h=self.conv(x)
        h=h[:,:,:T] if h.shape[-1]>=T else F.pad(h,(0,T-h.shape[-1]))
        h=x+h; h=self.se(h); return self.ln(h.transpose(1,2)).transpose(1,2)

class TCRSCANet(nn.Module):
    def __init__(self, nf, d=128, dropout=0.1):
        super().__init__()
        super().__init__()
        self.proj=nn.Linear(nf,d)
        self.b3=nn.ModuleList([MSBlock(d,3,dropout) for _ in range(3)])
        self.b5=nn.ModuleList([MSBlock(d,5,dropout) for _ in range(3)])
        self.b7=nn.ModuleList([MSBlock(d,7,dropout) for _ in range(3)])
        self.fuse=nn.Conv1d(d*3,d,1); self.ln1=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,4,dropout=dropout,batch_first=True)
        self.ln2=nn.LayerNorm(d)
        self.head=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
        for p in self.parameters():
            if p.dim()>1: nn.init.xavier_uniform_(p)
    def forward(self,x):
        B,T,_=x.shape; h=self.proj(x).transpose(1,2)
        h3=h;h5=h;h7=h
        for blk in self.b3:h3=blk(h3)
        for blk in self.b5:h5=blk(h5)
        for blk in self.b7:h7=blk(h7)
        h=torch.cat([h3,h5,h7],dim=1); h=self.fuse(h).transpose(1,2); h=self.ln1(h)
        a,_=self.attn(h,h,h); h=self.ln2(h+a)
        return self.head(h.mean(1))

def train_model(model, Xt, yt, Xv, yv, epochs=500, lr=5e-4, bs=256, dev='cuda'):
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.005)
    total_steps=epochs*max(1,len(Xt)//bs)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=total_steps)
    Xtt=torch.FloatTensor(Xt).to(dev); ytt=torch.FloatTensor(yt).to(dev)
    Xvv=torch.FloatTensor(Xv).to(dev)
    best_r,best_s=1e9,1e9;best_st=None;pat=0
    for ep in range(epochs):
        model.train();idx=np.random.permutation(len(Xtt))
        for i in range(0,len(Xt),bs):
            bi=idx[i:i+bs];p=model(Xtt[bi]).squeeze()
            d=p-ytt[bi];loss=F.mse_loss(p,ytt[bi])+0.02*F.relu(d).mean()
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step();sched.step()
        model.eval()
        with torch.no_grad():
            p=model(Xvv).squeeze().cpu().numpy()
            r=np.sqrt(np.mean((yv-p)**2));s=compute_score(yv,p)
            if r<best_r-0.01:best_r=r;best_s=s;best_st={k:v.cpu().clone() for k,v in model.state_dict().items()};pat=0
            else:pat+=1
        if (ep+1)%50==0 or ep==0:print(f'  E{ep+1}: R={r:.1f} S={s:.0f} best={best_r:.1f}')
        if pat>=100:print(f'Early stop E{ep+1}');break
    if best_st:model.load_state_dict(best_st)
    return best_r,best_s

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda:0');p.add_argument('--fd',default='FD001');p.add_argument('--epochs',type=int,default=500)
    args=p.parse_args()
    ds=['FD001','FD002','FD003','FD004'] if args.fd=='all' else [args.fd]
    for fd in ds:
        print(f'\n=== {fd} TCRSCANet (SOTA) ===')
        Xt,yt,Xv,yv,nf=load_data('data/processed',fd)
        print(f'Train:{Xt.shape} Test:{Xv.shape}')
        m=TCRSCANet(nf).to(args.device)
        print(f'Params:{sum(p.numel() for p in m.parameters()):,}')
        r,s=train_model(m,Xt,yt,Xv,yv,epochs=args.epochs,dev=args.device)
        print(f'=> RMSE={r:.1f} Score={s:.0f}')
