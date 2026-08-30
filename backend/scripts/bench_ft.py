"""Fine-tuning benchmark: pretrained vs random vs LSTM on all C-MAPSS subsets."""
import torch, torch.nn as nn, numpy as np, pandas as pd
from src.models.shrdl import SHRDL
from src.baselines.lstm_baseline import LSTMBaseline

device = torch.device('cuda:1')

def load_data(dd, fd, sl=64, st=4):
    cd = dd + '/cmapss/6. Turbofan Engine Degradation Simulation Data Set'
    tr = pd.read_csv(f'{cd}/train_{fd}.txt', sep=r'\s+', header=None).values
    te = pd.read_csv(f'{cd}/test_{fd}.txt', sep=r'\s+', header=None).values
    tru = pd.read_csv(f'{cd}/RUL_{fd}.txt', sep=r'\s+', header=None).values.flatten()
    def proc(raw):
        Xl,yl=[],[]
        for u in np.unique(raw[:,0]):
            t=raw[raw[:,0]==u,2:].astype(np.float32)
            if len(t)<sl:continue
            ws=[]; 
            for i in range(0,len(t)-sl+1,st): ws.append(t[i:i+sl])
            if not ws: continue
            X=np.stack(ws); X=(X-X.mean())/(X.std()+1e-8)
            y=np.clip(np.arange(len(X))[::-1]*st,0,130)
            Xl.append(X.astype(np.float32));yl.append(y.astype(np.float32))
        return np.concatenate(Xl),np.concatenate(yl)
    Xt,yt=proc(tr)
    Xv=[]; 
    for u in np.unique(te[:,0]):
        t=te[te[:,0]==u,2:].astype(np.float32)
        if len(t)<sl:continue
        w=t[-sl:];w=(w-w.mean())/(w.std()+1e-8);Xv.append(w)
    Xv=np.stack(Xv).astype(np.float32)
    yv=tru[:len(Xv)].astype(np.float32)
    return Xt,yt,Xv,yv,tr.shape[1]-2

class Head(nn.Module):
    def __init__(self,d=128):super().__init__();self.net=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Dropout(0.2),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1),nn.Softplus())
    def forward(self,x):return self.net(x)

def ft_eval(enc,Xt,yt,Xv,yv,ep=80):
    hd=Head().to(device);enc.train();hd.train()
    opt=torch.optim.AdamW(list(enc.parameters())+list(hd.parameters()),lr=1e-3)
    Xtt=torch.FloatTensor(Xt).to(device);ytt=torch.FloatTensor(yt).to(device)
    Xvv=torch.FloatTensor(Xv).to(device);best=1e9
    for e in range(ep):
        idx=np.random.permutation(len(Xtt))
        for i in range(0,len(Xt),128):
            bi=idx[i:i+128]
            f=enc.encode(Xtt[bi],project=False).mean(1)
            loss=nn.MSELoss()(hd(f).squeeze(),ytt[bi])
            opt.zero_grad();loss.backward();opt.step()
        enc.eval();hd.eval()
        with torch.no_grad():
            p=hd(enc.encode(Xvv,project=False).mean(1)).squeeze().cpu().numpy()
            rmse=np.sqrt(np.mean((yv-p)**2))
            if rmse<best:best=rmse
        enc.train();hd.train()
    return best

# Load checkpoint
ckpt=torch.load('checkpoints/shrdl_gpu1_best.pt',map_location='cpu',weights_only=False)
state=ckpt['model'] if 'model' in ckpt else ckpt
ckpt_n=None
for k,v in state.items():
    if 'input_proj.0.weight' in k:ckpt_n=v.shape[1];break

res={}
for fd in ['FD001','FD002','FD003','FD004']:
    Xt,yt,Xv,yv,nf=load_data('data/processed',fd)
    # Pretrained
    enc=SHRDL(128,4,2,n_features=ckpt_n,feature_dim=64,window_sizes=[2,4,8,16,32]).to(device)
    md=enc.state_dict();md.update({k:v for k,v in state.items() if k in md and v.shape==md[k].shape})
    enc.load_state_dict(md,strict=False)
    if nf!=ckpt_n:
        old=enc.input_proj;enc.input_proj=nn.Sequential(nn.Linear(nf,ckpt_n),old).to(device)
    r1=ft_eval(enc,Xt,yt,Xv,yv)
    # Random
    enc_r=SHRDL(128,4,2,n_features=nf,feature_dim=64,window_sizes=[2,4,8,16,32]).to(device)
    r2=ft_eval(enc_r,Xt,yt,Xv,yv)
    # LSTM
    lstm=LSTMBaseline(n_features=nf,hidden_dim=128,n_layers=2).to(device)
    opt=torch.optim.Adam(lstm.parameters(),lr=1e-3)
    Xtt=torch.FloatTensor(Xt).to(device);ytt=torch.FloatTensor(yt).to(device)
    Xvv=torch.FloatTensor(Xv).to(device);best3=1e9;lstm.train()
    for e in range(80):
        idx=np.random.permutation(len(Xtt))
        for i in range(0,len(Xt),128):
            bi=idx[i:i+128];loss=nn.MSELoss()(lstm(Xtt[bi]).squeeze(),ytt[bi])
            opt.zero_grad();loss.backward();opt.step()
        lstm.eval()
        with torch.no_grad():
            rmse=np.sqrt(np.mean((yv-lstm(Xvv).squeeze().cpu().numpy())**2))
            if rmse<best3:best3=rmse
        lstm.train()
    res[fd]={'PT-FT':r1,'Rand-FT':r2,'LSTM':best3}
    print(f'{fd}: PT-FT={r1:.1f}, Rand-FT={r2:.1f}, LSTM={best3:.1f}')

print('\n=== Fine-tuning Results ===')
for fd,r in res.items():
    g=(1-r['PT-FT']/r['Rand-FT'])*100
    print(f'{fd}: PT={r["PT-FT"]:.1f}  Rand={r["Rand-FT"]:.1f}  LSTM={r["LSTM"]:.1f}  Gain={g:.0f}%')
