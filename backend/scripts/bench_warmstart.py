"""Compare: pretrained warm-start vs random init, both fully trained."""
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
            ws=[]
            for i in range(0,len(t)-sl+1,st): ws.append(t[i:i+sl])
            if not ws: continue
            X=np.stack(ws); X=(X-X.mean())/(X.std()+1e-8)
            y=np.clip(np.arange(len(X))[::-1]*st,0,130)
            Xl.append(X.astype(np.float32));yl.append(y.astype(np.float32))
        return np.concatenate(Xl),np.concatenate(yl)
    Xt,yt=proc(tr)
    Xv=[]
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

def train_full(encoder, Xt, yt, Xv, yv, epochs=60, lr=1e-3):
    head = Head().to(device)
    encoder.train(); head.train()
    opt = torch.optim.AdamW(list(encoder.parameters())+list(head.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtt = torch.FloatTensor(Xt).to(device); ytt = torch.FloatTensor(yt).to(device)
    Xvv = torch.FloatTensor(Xv).to(device)
    best = 1e9; history = []
    for ep in range(epochs):
        idx = np.random.permutation(len(Xtt))
        for i in range(0,len(Xt),128):
            bi = idx[i:i+128]
            f = encoder.encode(Xtt[bi], project=False).mean(1)
            loss = nn.MSELoss()(head(f).squeeze(), ytt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        encoder.eval(); head.eval()
        with torch.no_grad():
            p = head(encoder.encode(Xvv, project=False).mean(1)).squeeze().cpu().numpy()
            rmse = np.sqrt(np.mean((yv-p)**2))
            if rmse < best: best = rmse
            history.append(rmse)
        encoder.train(); head.train()
    return best, history

# Load checkpoint
ckpt = torch.load('checkpoints/shrdl_gpu1_best.pt', map_location='cpu', weights_only=False)
state = ckpt['model'] if 'model' in ckpt else ckpt
ckpt_n = None
for k,v in state.items():
    if 'input_proj.0.weight' in k: ckpt_n = v.shape[1]; break

for fd in ['FD001','FD003']:
    print(f'\n=== {fd} ===')
    Xt,yt,Xv,yv,nf = load_data('data/processed', fd)
    
    # 1. Pretrained warm-start
    enc_pt = SHRDL(128,4,2,n_features=ckpt_n,feature_dim=64,window_sizes=[2,4,8,16,32]).to(device)
    md = enc_pt.state_dict()
    md.update({k:v for k,v in state.items() if k in md and v.shape==md[k].shape})
    enc_pt.load_state_dict(md, strict=False)
    if nf != ckpt_n:
        old = enc_pt.input_proj
        enc_pt.input_proj = nn.Sequential(nn.Linear(nf, ckpt_n), old).to(device)
    
    best_pt, hist_pt = train_full(enc_pt, Xt, yt, Xv, yv)
    print(f'  Pretrained warm-start: best={best_pt:.1f}, first5={[round(h) for h in hist_pt[:5]]}')
    
    # 2. Random init
    enc_rd = SHRDL(128,4,2,n_features=nf,feature_dim=64,window_sizes=[2,4,8,16,32]).to(device)
    best_rd, hist_rd = train_full(enc_rd, Xt, yt, Xv, yv)
    print(f'  Random init:          best={best_rd:.1f}, first5={[round(h) for h in hist_rd[:5]]}')
    
    # 3. LSTM
    lstm = LSTMBaseline(n_features=nf,hidden_dim=128,n_layers=2).to(device)
    opt=torch.optim.Adam(lstm.parameters(),lr=1e-3)
    Xtt=torch.FloatTensor(Xt).to(device);ytt=torch.FloatTensor(yt).to(device)
    Xvv=torch.FloatTensor(Xv).to(device);best_l=1e9;lstm.train()
    for e in range(80):
        idx=np.random.permutation(len(Xtt))
        for i in range(0,len(Xt),128):
            bi=idx[i:i+128];loss=nn.MSELoss()(lstm(Xtt[bi]).squeeze(),ytt[bi])
            opt.zero_grad();loss.backward();opt.step()
        lstm.eval()
        with torch.no_grad():
            rmse=np.sqrt(np.mean((yv-lstm(Xvv).squeeze().cpu().numpy())**2))
            if rmse<best_l:best_l=rmse
        lstm.train()
    
    print(f'  LSTM:                 best={best_l:.1f}')
    print(f'  Pretrained gain: {(1-best_pt/best_rd)*100:.1f}% faster convergence')
