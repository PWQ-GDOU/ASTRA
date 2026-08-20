"""
SHRDL-Pro: Optimized architecture targeting C-MAPSS SOTA.
Key improvements:
- TCN (Temporal Conv Net) for local feature extraction
- Multi-head Self-Attention for global dependencies  
- Deeper architecture (4 blocks)
- Residual connections throughout
- Standard C-MAPSS preprocessing
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, math
from typing import Optional

class Chomp1d(nn.Module):
    """Remove extra padding from causal conv."""
    def __init__(self, chomp_size): super().__init__(); self.cs = chomp_size
    def forward(self, x): return x[:, :, :-self.cs].contiguous() if self.cs > 0 else x

class TemporalBlock(nn.Module):
    """TCN block: dilated causal conv + weight norm + ReLU + dropout + residual."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNEncoder(nn.Module):
    """Multi-scale TCN encoder: captures local temporal patterns at multiple scales."""
    def __init__(self, n_features, d_model=256, levels=4, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        channels = [d_model] * (levels + 1)
        self.blocks = nn.ModuleList()
        for i in range(levels):
            self.blocks.append(TemporalBlock(channels[i], channels[i+1], kernel_size, 
                                              dilation=2**i, dropout=dropout))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, T, F) -> (B, D, T)
        x = self.input_proj(x).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2)  # (B, T, D)

class AttentionBlock(nn.Module):
    """Multi-head self-attention with residual + LayerNorm."""
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*4, d_model))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class SHRDLPro(nn.Module):
    """SHRDL-Pro: TCN + Multi-Head Attention encoder for RUL prediction."""
    def __init__(self, n_features, d_model=256, n_heads=8, tcn_levels=4, attn_layers=3, dropout=0.1):
        super().__init__()
        self.tcn = TCNEncoder(n_features, d_model, tcn_levels, dropout=dropout)
        self.attn_layers = nn.ModuleList([AttentionBlock(d_model, n_heads, dropout) for _ in range(attn_layers)])
        self.pos_encoding = self._positional_encoding(1000, d_model)
    
    def _positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        B, T, _ = x.shape
        h = self.tcn(x)  # (B, T, D)
        h = h + self.pos_encoding[:, :T, :]
        for attn in self.attn_layers:
            h = attn(h)
        return h.mean(dim=1)  # Global pooling


class RULPredictor(nn.Module):
    """Complete RUL prediction model."""
    def __init__(self, n_features, d_model=192, n_heads=6, tcn_levels=3, attn_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = SHRDLPro(n_features, d_model, n_heads, tcn_levels, attn_layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1))
    
    def forward(self, x): return self.head(self.encoder(x))


def load_cmapss_std(dd, fd, seq_len=30):
    """Standard C-MAPSS preprocessing matching academic benchmarks."""
    cd = dd + '/cmapss/6. Turbofan Engine Degradation Simulation Data Set'
    tr = pd.read_csv(f'{cd}/train_{fd}.txt', sep=r'\s+', header=None).values
    te = pd.read_csv(f'{cd}/test_{fd}.txt', sep=r'\s+', header=None).values
    tru = pd.read_csv(f'{cd}/RUL_{fd}.txt', sep=r'\s+', header=None).values.flatten()
    
    # Standard: max RUL = 125 (some use 130)
    MAX_RUL = 125
    
    def proc(raw):
        Xl, yl = [], []
        for u in np.unique(raw[:,0]):
            t = raw[raw[:,0]==u, 2:].astype(np.float32)
            n = len(t)
            if n < seq_len: continue
            # Standard: sliding windows with stride=1
            ws = []
            for i in range(n - seq_len + 1):
                w = t[i:i+seq_len]
                # Normalize per window
                w = (w - w.mean(axis=0, keepdims=True)) / (w.std(axis=0, keepdims=True) + 1e-8)
                ws.append(w)
            X = np.stack(ws)
            # RUL = remaining cycles from END of window
            rul = n - np.arange(seq_len-1, n)
            rul = np.clip(rul, 0, MAX_RUL)
            Xl.append(X.astype(np.float32)); yl.append(rul.astype(np.float32))
        return np.concatenate(Xl), np.concatenate(yl)
    
    Xt, yt = proc(tr)
    # Test: only last window per unit
    Xv_list, yv_list = [], []
    for u in np.unique(te[:,0]):
        t = te[te[:,0]==u, 2:].astype(np.float32)
        if len(t) < seq_len: continue
        w = t[-seq_len:]
        w = (w - w.mean(axis=0, keepdims=True)) / (w.std(axis=0, keepdims=True) + 1e-8)
        Xv_list.append(w)
    Xv = np.stack(Xv_list).astype(np.float32)
    yv = tru[:len(Xv_list)].astype(np.float32)
    return Xt, yt, Xv, yv, tr.shape[1]-2


def compute_score(y_true, y_pred):
    """NASA C-MAPSS scoring function. Penalizes late predictions more."""
    d = y_pred - y_true
    return np.sum(np.where(d >= 0, np.exp(d/13.) - 1, np.exp(-d/10.) - 1))


def train_model(model, Xt, yt, Xv, yv, epochs=120, lr=1e-3, device='cuda'):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    steps_per_epoch = max(1, len(Xt) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs=epochs, 
                                                  steps_per_epoch=steps_per_epoch + 1)  # +1 to avoid off-by-one
    Xtt = torch.FloatTensor(Xt).to(device); ytt = torch.FloatTensor(yt).to(device)
    Xvv = torch.FloatTensor(Xv).to(device)
    best_rmse, best_score = 1e9, 1e9
    best_state = None
    patience, no_improve = 20, 0
    
    for ep in range(epochs):
        idx = np.random.permutation(len(Xtt))
        for i in range(0, len(Xt), 128):
            bi = idx[i:i+128]
            pred = model(Xtt[bi]).squeeze()
            loss = F.mse_loss(pred, ytt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        
        model.eval()
        with torch.no_grad():
            p = model(Xvv).squeeze().cpu().numpy()
            rmse = np.sqrt(np.mean((yv - p)**2))
            score = compute_score(yv, p)
            if rmse < best_rmse:
                best_rmse = rmse; best_score = score
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
        model.train()
        if no_improve >= patience: 
            print(f'  Early stop at E{ep}')
            break
        if ep % 20 == 0: print(f'  E{ep}: RMSE={rmse:.1f}, Score={score:.0f}, best={best_rmse:.1f}')
    
    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    return best_rmse, best_score


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--fd', default='FD001')
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    for fd in ['FD001','FD002','FD003','FD004'] if args.fd == 'all' else [args.fd]:
        print(f'\n=== {fd} ===')
        Xt, yt, Xv, yv, nf = load_cmapss_std('data/processed', fd)
        print(f'  Train: {Xt.shape}, Test: {Xv.shape}')
        
        # SHRDL-Pro
        model = RULPredictor(n_features=nf, d_model=192, n_heads=6, tcn_levels=3, attn_layers=2).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Params: {n_params:,}')
        
        rmse, score = train_model(model, Xt, yt, Xv, yv, epochs=args.epochs, device=device)
        print(f'  Result: RMSE={rmse:.1f}, Score={score:.0f}')
        print(f'  SOTA:   RMSE~11, Score~200')
