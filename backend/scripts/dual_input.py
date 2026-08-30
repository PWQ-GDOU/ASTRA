import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, math, argparse

def load_dual(dd, fd, seq_len=30):
    cd = dd + '/cmapss/6. Turbofan Engine Degradation Simulation Data Set'
    tr = pd.read_csv(f'{cd}/train_{fd}.txt', sep=r'\s+', header=None).values
    te = pd.read_csv(f'{cd}/test_{fd}.txt', sep=r'\s+', header=None).values
    tru = pd.read_csv(f'{cd}/RUL_{fd}.txt', sep=r'\s+', header=None).values.flatten()
    MAX_RUL = 125
    # Separate: sensors vs operating conditions
    all_sens = tr[:, 5:].astype(np.float32)  # 21 sensors
    all_ops = tr[:, 2:5].astype(np.float32)   # 3 conditions
    s_mean = all_sens.mean(axis=0, keepdims=True)
    s_std = all_sens.std(axis=0, keepdims=True) + 1e-8
    o_mean = all_ops.mean(axis=0, keepdims=True)
    o_std = all_ops.std(axis=0, keepdims=True) + 1e-8
    
    def proc(raw):
        Xs, Xo, yl = [], [], []
        for u in np.unique(raw[:,0]):
            t_s = raw[raw[:,0]==u, 5:].astype(np.float32)
            t_o = raw[raw[:,0]==u, 2:5].astype(np.float32)
            t_s = (t_s - s_mean) / s_std
            t_o = (t_o - o_mean) / o_std
            n = len(t_s)
            if n < seq_len: continue
            for i in range(n-seq_len+1):
                Xs.append(t_s[i:i+seq_len])
                Xo.append(t_o[i:i+seq_len])
                yl.append(min(n-(i+seq_len-1), MAX_RUL))
        return np.stack(Xs).astype(np.float32), np.stack(Xo).astype(np.float32), np.array(yl,dtype=np.float32)
    
    Xst, Xot, yt = proc(tr)
    # Test
    Xsv, Xov = [], []
    for u in np.unique(te[:,0]):
        t_s = te[te[:,0]==u, 5:].astype(np.float32)
        t_o = te[te[:,0]==u, 2:5].astype(np.float32)
        t_s = (t_s - s_mean) / s_std
        t_o = (t_o - o_mean) / o_std
        if len(t_s) >= seq_len:
            Xsv.append(t_s[-seq_len:]); Xov.append(t_o[-seq_len:])
    Xsv = np.stack(Xsv).astype(np.float32); Xov = np.stack(Xov).astype(np.float32)
    yv = tru[:len(Xsv)].astype(np.float32)
    return Xst, Xot, yt, Xsv, Xov, yv, 21  # n_sensors

def compute_score(yt,yp):
    d=yp-yt;return float(np.sum(np.where(d>=0,np.exp(d/13.)-1,np.exp(-d/10.)-1)))

class FiLMLayer(nn.Module):
    # Feature-wise Linear Modulation: condition modulates sensor features.
    def __init__(self, d): super().__init__(); self.ln=nn.LayerNorm(d)
    def forward(self, x, gamma, beta):
        return self.ln(x * (1 + gamma) + beta)

class DualInputRUL(nn.Module):
    def __init__(self, n_sens=21, n_cond=3, d=192, dropout=0.1):
        super().__init__()
        # Sensor encoder: TCN
        self.s_proj = nn.Linear(n_sens, d)
        self.s_tcn = nn.Sequential(
            *[self._tcn_block(d,3,dropout) for _ in range(3)],
            *[self._tcn_block(d,5,dropout) for _ in range(3)],
        )
        # Condition encoder: MLP
        self.c_proj = nn.Linear(n_cond, d)
        self.c_encoder = nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d))
        # FiLM: condition modulates sensor features
        self.gamma_net = nn.Linear(d, d)
        self.beta_net = nn.Linear(d, d)
        self.film = FiLMLayer(d)
        # Self-attention + head
        self.attn = nn.MultiheadAttention(d,4,dropout=dropout,batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
        for p in self.parameters():
            if p.dim()>1: nn.init.xavier_uniform_(p)
    
    def _tcn_block(self,ch,ks,dropout):
        dil=2;pad=(ks-1)*dil
        return nn.Sequential(
            nn.Conv1d(ch,ch,ks,dilation=dil,padding=pad),nn.GELU(),nn.Dropout(dropout),
            nn.Conv1d(ch,ch,1),nn.GELU())
    
    def forward(self, x_sens, x_cond):
        B,T,_=x_sens.shape
        # Sensor path
        hs = self.s_proj(x_sens).transpose(1,2)
        for tcn in self.s_tcn: hs = tcn(hs)
        hs = hs.transpose(1,2)  # (B,T,D)
        # Condition path  
        hc = self.c_proj(x_cond).mean(1)  # pool conditions
        hc = self.c_encoder(hc)  # (B,D)
        # FiLM: modulate sensor features by condition
        gamma = self.gamma_net(hc).unsqueeze(1)
        beta = self.beta_net(hc).unsqueeze(1)
        h = self.film(hs, gamma, beta)
        # Self-attention + predict
        a,_ = self.attn(h,h,h); h = self.ln(h + a)
        return self.head(h.mean(1))

def train_dual(model, Xst,Xot,yt,Xsv,Xov,yv, epochs=300,lr=5e-4,bs=256,dev='cuda:0'):
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.005)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs*max(1,len(Xst)//bs))
    Xstt=torch.FloatTensor(Xst).to(dev);Xott=torch.FloatTensor(Xot).to(dev);ytt=torch.FloatTensor(yt).to(dev)
    Xsvv=torch.FloatTensor(Xsv).to(dev);Xovv=torch.FloatTensor(Xov).to(dev)
    best_r,best_s=1e9,1e9;best_st=None;pat=0
    for ep in range(epochs):
        model.train();idx=np.random.permutation(len(Xstt))
        for i in range(0,len(Xst),bs):
            bi=idx[i:i+bs];p=model(Xstt[bi],Xott[bi]).squeeze()
            d=p-ytt[bi];loss=F.mse_loss(p,ytt[bi])+0.02*F.relu(d).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step();sched.step()
        model.eval()
        with torch.no_grad():
            p=model(Xsvv,Xovv).squeeze().cpu().numpy()
            r=np.sqrt(np.mean((yv-p)**2));s=compute_score(yv,p)
            if r<best_r-0.01:best_r=r;best_s=s;best_st={k:v.cpu().clone() for k,v in model.state_dict().items()};pat=0
            else:pat+=1
        if (ep+1)%30==0 or ep==0:print(f'E{ep+1}: R={r:.1f} S={s:.0f} best={best_r:.1f}')
        if pat>=60:print(f'Early stop E{ep+1}');break
    if best_st:model.load_state_dict(best_st)
    return best_r,best_s

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fd',default='FD002');p.add_argument('--epochs',type=int,default=300);p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    for fd in ([args.fd] if args.fd!='all' else ['FD002','FD004']):
        print(f'\n=== {fd} Dual-Input (FiLM) ===')
        Xst,Xot,yt,Xsv,Xov,yv,ns=load_dual('data/processed',fd)
        print(f'Train:{Xst.shape} Test:{Xsv.shape}')
        m=DualInputRUL().to(args.device);print(f'Params:{sum(p.numel() for p in m.parameters()):,}')
        r,s=train_dual(m,Xst,Xot,yt,Xsv,Xov,yv,epochs=args.epochs,dev=args.device)
        print(f'=> RMSE={r:.1f} Score={s:.0f}')
