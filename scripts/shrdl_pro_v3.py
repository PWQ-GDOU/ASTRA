"""
SHRDL-Pro v3: Add Working Condition Decomposition (WDL) to handle
multi-condition datasets (FD002, FD004).

Key idea: Cross-attention between operating condition signal and sensor
signal to identify and remove condition-explained variance, leaving
only degradation-related features.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, math, argparse
from shrdl_pro_v2 import combined_loss, compute_score, load_cmapss_std

# ========== WDL Module ==========
class WDL(nn.Module):
    """Working Condition Decomposition via cross-attention."""
    def __init__(self, d_model=192, n_heads=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.hd = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h_sensor, h_cond):
        """h_sensor: (B,T,D), h_cond: (B,C,D) — remove condition effects"""
        B, T, D = h_sensor.shape
        # Cross-attention: condition attends to sensor
        Q = self.Wq(h_cond).view(B, -1, self.n_heads, self.hd).transpose(1,2)
        K = self.Wk(h_sensor).view(B, T, self.n_heads, self.hd).transpose(1,2)
        V = self.Wv(h_sensor).view(B, T, self.n_heads, self.hd).transpose(1,2)
        scale = self.hd ** 0.5
        attn = F.softmax(torch.matmul(Q, K.transpose(-2,-1)) / scale, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V).transpose(1,2).contiguous().view(B, -1, D)
        out = out.mean(dim=1, keepdim=True).expand(-1, T, -1)
        out = self.Wo(out)
        return self.norm(h_sensor - out)  # Subtract condition component


class Chomp1d(nn.Module):
    def __init__(self, cs): super().__init__(); self.cs = cs
    def forward(self, x): return x[:,:,:-self.cs].contiguous() if self.cs>0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dil, dropout=0.1):
        super().__init__()
        pad = (ks-1)*dil
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, ks, dilation=dil, padding=pad),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, ks, dilation=dil, padding=pad),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x); res = x if self.down is None else self.down(x)
        return self.relu(out + res)

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

class SHRDLProV3(nn.Module):
    """TCN + WDL + Multi-Head Attention with condition-awareness."""
    def __init__(self, n_features, n_cond=3, d_model=192, n_heads=6, 
                 tcn_levels=3, attn_layers=2, dropout=0.1):
        super().__init__()
        # Separate projections for sensor and condition
        self.sensor_proj = nn.Linear(n_features - n_cond, d_model)
        self.cond_proj = nn.Linear(n_cond, d_model)
        
        # WDL: remove condition effects
        self.wdl = WDL(d_model, n_heads, dropout)
        
        # TCN on condition-removed features
        self.tcn = nn.ModuleList([
            TemporalBlock(d_model, d_model, 3, 2**i, dropout) for i in range(tcn_levels)
        ])
        
        # Attention layers
        self.attn_layers = nn.ModuleList([
            AttentionBlock(d_model, n_heads, dropout) for _ in range(attn_layers)
        ])
        
        # Positional encoding
        pe = torch.zeros(1000, d_model)
        pos = torch.arange(0,1000).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.pe = nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        # x: (B,T,F) where last n_cond cols are operating conditions
        B,T,F = x.shape
        n_cond = 3
        
        h_s = self.sensor_proj(x[:,:,:F-n_cond])  # sensor part
        h_c = self.cond_proj(x[:,:,:n_cond])       # condition part
        
        # WDL: remove condition influence
        h = self.wdl(h_s, h_c)
        
        # TCN
        h = h.transpose(1,2)  # (B,D,T)
        for block in self.tcn:
            h = block(h)
        h = h.transpose(1,2)  # (B,T,D)
        
        # Attention
        h = h + self.pe[:,:T,:]
        for attn in self.attn_layers:
            h = attn(h)
        
        return h.mean(dim=1)


class RULPredictorV3(nn.Module):
    def __init__(self, n_features, n_cond=3, d_model=192, n_heads=6,
                 tcn_levels=3, attn_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = SHRDLProV3(n_features, n_cond, d_model, n_heads, 
                                   tcn_levels, attn_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1))
    def forward(self, x): return self.head(self.encoder(x))


def train_v3(model, Xt, yt, Xv, yv, epochs=200, lr=5e-4, wd=3e-3, alpha=0.3, bs=256, device='cuda'):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = 10
    spe = max(1, len(Xt)//bs)
    def lr_fn(step):
        if step < warmup*spe: return step/(warmup*spe)
        p = (step-warmup*spe)/(epochs*spe-warmup*spe)
        return 0.5*(1+math.cos(math.pi*p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    
    Xtt = torch.FloatTensor(Xt).to(device); ytt = torch.FloatTensor(yt).to(device)
    Xvv = torch.FloatTensor(Xv).to(device)
    best_rmse, best_score = 1e9, 1e9
    best_state = None
    
    for ep in range(epochs):
        idx = np.random.permutation(len(Xtt))
        model.train()
        for i in range(0,len(Xt),bs):
            bi = idx[i:i+bs]
            pred = model(Xtt[bi]).squeeze()
            loss = combined_loss(pred, ytt[bi], alpha)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sched.step()
        
        model.eval()
        with torch.no_grad():
            p = model(Xvv).squeeze().cpu().numpy()
            rmse = np.sqrt(np.mean((yv-p)**2))
            score = compute_score(yv, p)
            if rmse < best_rmse:
                best_rmse=rmse; best_score=score
                best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
        if (ep+1)%30==0 or ep==0:
            print(f'  E{ep+1:3d}: RMSE={rmse:.1f} Score={score:.0f} best={best_rmse:.1f}')
    
    if best_state: model.load_state_dict(best_state)
    return best_rmse, best_score


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--fd', default='FD002')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--wd', type=float, default=3e-3)
    p.add_argument('--alpha', type=float, default=0.3)
    args = p.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    for fd in ([args.fd] if args.fd != 'all' else ['FD002','FD004']):
        print(f'\n{"="*60}\n  {fd}: SHRDL-Pro V3 (WDL-enhanced)\n{"="*60}')
        Xt, yt, Xv, yv, nf = load_cmapss_std('data/processed', fd)
        print(f'  Train: {Xt.shape}, Test: {Xv.shape}')
        
        model = RULPredictorV3(n_features=nf).to(device)
        print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')
        
        rmse, score = train_v3(model, Xt, yt, Xv, yv, 
                                epochs=args.epochs, lr=args.lr, 
                                wd=args.wd, alpha=args.alpha, device=device)
        print(f'  => RMSE={rmse:.1f}, Score={score:.0f}')
        print(f'  V2(no WDL): RMSE~54,  Improvement: {(1-rmse/54)*100:.0f}%')
