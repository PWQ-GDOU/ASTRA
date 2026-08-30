"""
Fixed SHRDL pretraining — all datasets padded to consistent feature dim.
Also saves evaluation-ready encoders for C-MAPSS and battery.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
import numpy as np, pandas as pd
from scipy.io import loadmat
import glob, argparse
from tqdm import tqdm

from src.config import CHECKPOINT_DIR
from src.models.shrdl import SHRDL, CIMCLLoss
from src.data.preprocess import extract_sliding_windows
from src.utils.train_utils import set_seed


def pad_to_dim(traj, target_dim):
    """Pad trajectory features to target_dim with zeros."""
    if traj.shape[1] >= target_dim:
        return traj[:, :target_dim]
    pad = np.zeros((len(traj), target_dim - traj.shape[1]), dtype=traj.dtype)
    return np.concatenate([traj, pad], axis=1)


def load_all_trajectories(data_dir, seq_len):
    """Load all datasets with consistent feature padding."""
    raw = []
    
    # Nozzle data
    for fname in ['喷管喉部退化时程.csv', '壁面烧蚀速率分布数据.csv']:
        fp = os.path.join(data_dir, 'raw', fname)
        if os.path.exists(fp):
            df = pd.read_csv(fp, encoding='utf-8-sig')
            d = df.values.astype(np.float32)
            if len(d) >= seq_len: raw.append(d)
            print(f"  Nozzle: {len(d)} timesteps x {d.shape[1]} features")
    
    # C-MAPSS
    cmapss_dir = os.path.join(data_dir, 'cmapss', '6. Turbofan Engine Degradation Simulation Data Set')
    for fd in ['FD001','FD002','FD003','FD004']:
        fp = os.path.join(cmapss_dir, f'train_{fd}.txt')
        if not os.path.exists(fp): continue
        df = pd.read_csv(fp, sep=r'\s+', header=None).values
        for u in np.unique(df[:,0]):
            t = df[df[:,0]==u, 2:]
            if len(t) >= seq_len: raw.append(t.astype(np.float32))
        print(f"  C-MAPSS {fd}: {len(np.unique(df[:,0]))} units x {df.shape[1]-2} features")
    
    # NASA Battery
    bat_dir = os.path.join(data_dir, 'nasa_battery', '5. Battery Data Set')
    for matf in sorted(glob.glob(os.path.join(bat_dir, 'B*.mat'))):
        try:
            m = loadmat(matf)
            batt = m[list(m.keys())[-1]]
            feats = []
            for ci in range(batt.shape[0]):
                cyc = batt[ci,0]
                if cyc is not None and len(cyc) > 0 and isinstance(cyc, np.ndarray) and cyc.ndim > 1:
                    feats.append([float(np.nanmean(cyc[:,c])) for c in range(min(cyc.shape[1],10))])
            if len(feats) >= seq_len: raw.append(np.array(feats, dtype=np.float32))
        except: pass
    print(f"  Battery: {len(raw) - 6} batteries loaded")
    
    # Pad all to same feature count
    max_feat = max(t.shape[1] for t in raw if len(t) > 0)
    padded = [pad_to_dim(t, max_feat) for t in raw if len(t) > 0 and t.shape[1] >= 2]
    print(f"  Total: {len(padded)} trajectories x {max_feat} features")
    
    return padded, max_feat


def prepare_windows(trajs, seq_len, stride):
    """Extract and normalize sliding windows."""
    all_w = []
    for t in trajs:
        X, _ = extract_sliding_windows(t, seq_len, stride, 
                                         feature_cols=list(range(t.shape[1])), label_col=0)
        if len(X) > 0:
            X = np.nan_to_num(X, 0)
            X = (X - X.mean(axis=(0,1), keepdims=True)) / (X.std(axis=(0,1), keepdims=True) + 1e-8)
            all_w.append(X)
    return np.concatenate(all_w).astype(np.float32) if all_w else np.array([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--stride', type=int, default=4)
    args = p.parse_args()
    
    set_seed(args.seed)
    device = f'cuda:{args.gpu}'
    
    # Load
    trajs, n_feat = load_all_trajectories('data/processed', args.seq_len)
    X = prepare_windows(trajs, args.seq_len, args.stride)
    print(f"  Windows: {X.shape}")
    
    if len(X) < 100:
        print("ERROR: too few windows")
        return
    
    # Model
    model = SHRDL(d_model=128, n_heads=4, n_adn_blocks=2, n_features=n_feat,
                  feature_dim=64, window_sizes=[2,4,8,16,32],
                  momentum=0.999, memory_bank_size=4096).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = CIMCLLoss(temperature=0.07)
    
    best_loss = 1e9
    n = len(X)
    nb = n // args.batch_size
    
    for ep in range(1, args.epochs + 1):
        model.train()
        idx = np.random.permutation(n)
        ep_loss = 0
        
        for bi in range(nb):
            start = (bi * args.batch_size) % n
            ix = idx[start:start + args.batch_size]
            xb = torch.FloatTensor(X[ix]).to(device)
            
            # Two augmented views
            n1 = 0.01 * torch.randn_like(xb)
            n2 = 0.01 * torch.randn_like(xb)
            
            q, k1, k2 = model(xb + n1, xb + n2)
            queue = model.memory_bank.get_queue().to(device)
            loss = crit(q, k2, queue)
            
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        
        avg = ep_loss / max(nb, 1)
        if avg < best_loss:
            best_loss = avg
            torch.save({'model': model.state_dict(), 'n_features': n_feat, 'loss': best_loss},
                       f'{CHECKPOINT_DIR}/shrdl_gpu{args.gpu}_best.pt')
        if ep % 50 == 0:
            print(f"  GPU{args.gpu} E{ep}: loss={avg:.4f} best={best_loss:.4f}")
    
    print(f"  GPU{args.gpu} DONE: best_loss={best_loss:.4f}")

if __name__ == '__main__':
    main()
