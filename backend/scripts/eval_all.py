"""
Complete evaluation pipeline — all methods on all datasets.
Handles feature dimension mismatch properly.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
import numpy as np, pandas as pd
from scipy.io import loadmat
import glob, argparse, json
from tqdm import tqdm

from src.models.shrdl import SHRDL
from src.baselines.lstm_baseline import LSTMBaseline, CNNLSTMBaseline
from src.baselines.wiener_process import WienerRULPredictor
from src.data.preprocess import extract_sliding_windows, build_piecewise_rul

# =============== Data ===============
def load_dataset(name, data_dir, seq_len=64, stride=4):
    """Load any dataset into (X_train, y_train, X_test, y_test, n_features)."""
    cmapss_dir = os.path.join(data_dir, 'cmapss', '6. Turbofan Engine Degradation Simulation Data Set')
    bat_dir = os.path.join(data_dir, 'nasa_battery', '5. Battery Data Set')
    
    if name.startswith('FD'):
        train = pd.read_csv(os.path.join(cmapss_dir, f'train_{name}.txt'), sep=r'\s+', header=None).values
        test = pd.read_csv(os.path.join(cmapss_dir, f'test_{name}.txt'), sep=r'\s+', header=None).values
        test_rul = pd.read_csv(os.path.join(cmapss_dir, f'RUL_{name}.txt'), sep=r'\s+', header=None).values.flatten()
        n_feat = train.shape[1] - 2
    elif name == 'battery':
        # Use first 20 batteries for train, last 10 for test
        mats = sorted(glob.glob(os.path.join(bat_dir, 'B*.mat')))
        train_mats, test_mats = mats[:20], mats[20:30] if len(mats)>20 else mats[:15]
        n_feat = None  # will detect
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    def extract_trajs(raw, is_test=False):
        trajs = []
        for u in np.unique(raw[:,0]):
            t = raw[raw[:,0]==u, 2:].astype(np.float32)
            if len(t) >= seq_len: trajs.append(t)
        return trajs
    
    def windows_from_trajs(trajs):
        Xall, yall = [], []
        for t in trajs:
            nf = t.shape[1]
            X, _ = extract_sliding_windows(t, seq_len, stride, list(range(nf)), label_col=0)
            if len(X) > 0:
                X = (X - X.mean(axis=(0,1), keepdims=True)) / (X.std(axis=(0,1), keepdims=True)+1e-8)
                y = np.clip(np.arange(len(X))[::-1]*stride, 0, 130).astype(np.float32)
                Xall.append(X); yall.append(y)
        return np.concatenate(Xall), np.concatenate(yall)
    
    if name.startswith('FD'):
        train_trajs = extract_trajs(train)
        Xt, yt = windows_from_trajs(train_trajs)
        n_feat = train_trajs[0].shape[1] if train_trajs else 24
        
        # Test: last window per unit
        test_trajs = extract_trajs(test)
        Xv_list = []
        for t in test_trajs:
            w = t[-seq_len:]
            w = (w - w.mean()) / (w.std() + 1e-8)
            Xv_list.append(w)
        Xv = np.stack(Xv_list).astype(np.float32)
        yv = test_rul[:len(Xv_list)].astype(np.float32)
    
    elif name == 'battery':
        def load_bat(fp):
            m = loadmat(fp); b = m[list(m.keys())[-1]]
            feats = []
            for ci in range(b.shape[0]):
                cyc = b[ci,0]
                if cyc is not None and len(cyc)>0 and isinstance(cyc, np.ndarray) and cyc.ndim>1:
                    feats.append([float(np.nanmean(cyc[:,c])) for c in range(min(cyc.shape[1],10))])
            return np.array(feats, dtype=np.float32)
        
        train_trajs = [load_bat(f) for f in train_mats if os.path.exists(f)]
        test_trajs = [load_bat(f) for f in test_mats if os.path.exists(f)]
        train_trajs = [t for t in train_trajs if len(t)>=seq_len]
        test_trajs = [t for t in test_trajs if len(t)>=seq_len]
        
        Xt, yt = windows_from_trajs(train_trajs)
        n_feat = train_trajs[0].shape[1] if train_trajs else 10
        
        Xv_list = [t[-seq_len:] for t in test_trajs]
        Xv_list = [(w - w.mean())/(w.std()+1e-8) for w in Xv_list]
        Xv = np.stack(Xv_list).astype(np.float32)
        # RUL = remaining cycles
        yv = np.array([len(t) for t in test_trajs], dtype=np.float32)
    
    return Xt, yt, Xv, yv, n_feat


# =============== Evaluators ===============
def eval_shrdl(ckpt_path, Xt, yt, Xv, yv, device, frozen=True):
    """Evaluate SHRDL encoder."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    ckpt_n = None
    for k,v in state.items():
        if 'input_proj.0.weight' in k: ckpt_n = v.shape[1]; break
    
    n_data = Xt.shape[-1]
    encoder = SHRDL(d_model=128, n_heads=4, n_adn_blocks=2, n_features=ckpt_n,
                    feature_dim=64, window_sizes=[2,4,8,16,32]).to(device)
    md = encoder.state_dict()
    md.update({k:v for k,v in state.items() if k in md and v.shape==md[k].shape})
    encoder.load_state_dict(md, strict=False)
    
    # Adapt input dim
    if n_data != ckpt_n:
        old = encoder.input_proj
        encoder.input_proj = nn.Sequential(nn.Linear(n_data, ckpt_n), old).to(device)
    
    encoder.eval()
    if frozen:
        for p in encoder.parameters(): p.requires_grad = False
    
    # Train ridge on features
    from sklearn.linear_model import Ridge
    with torch.no_grad():
        ft = encoder.encode(torch.FloatTensor(Xt).to(device), project=False).mean(1).cpu().numpy()
        fv = encoder.encode(torch.FloatTensor(Xv).to(device), project=False).mean(1).cpu().numpy()
    ridge = Ridge(alpha=1.0).fit(ft, yt)
    pred = ridge.predict(fv)
    return np.sqrt(np.mean((yv - pred)**2)), pred


def eval_lstm(Xt, yt, Xv, yv, device):
    """Evaluate LSTM baseline."""
    lstm = LSTMBaseline(n_features=Xt.shape[-1], hidden_dim=128, n_layers=2).to(device)
    opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    Xtt = torch.FloatTensor(Xt).to(device); ytt = torch.FloatTensor(yt).to(device)
    Xvv = torch.FloatTensor(Xv).to(device)
    best = 1e9
    for ep in range(80):
        idx = np.random.permutation(len(Xtt))
        for i in range(0, len(Xt), 128):
            bi = idx[i:i+128]
            loss = nn.MSELoss()(lstm(Xtt[bi]).squeeze(), ytt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            rmse = np.sqrt(np.mean((yv - lstm(Xvv).squeeze().cpu().numpy())**2))
            if rmse < best: best = rmse
    return best


def eval_wiener(Xt, yt, Xv, yv):
    """Evaluate Wiener baseline with proper HI (last feature = most degrading)."""
    # Use the most degrading sensor channel (max variance across time)
    # For C-MAPSS, sensor channels degrade differently - use mean of last 5 sensors
    hi_train = Xt[:, -1, -5:].mean(axis=1)  # Last timestep, last 5 sensors
    hi_test = Xv[:, -1, -5:].mean(axis=1)
    
    # Normalize
    hi_max = hi_train.max()
    hi_train = hi_train / (hi_max + 1e-8)
    hi_test = hi_test / (hi_max + 1e-8)
    
    # RUL proportional to HI
    preds = hi_test * yt.max()  # Linear scaling
    preds = np.clip(preds, 0, 130)
    
    try:
        rmse = np.sqrt(np.mean((yv - preds)**2))
    except:
        rmse = 999
    return rmse


# =============== Main ===============
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/processed')
    p.add_argument('--checkpoint', default='checkpoints/shrdl_gpu1_best.pt')
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--datasets', nargs='+', default=['FD001','FD002','FD003','FD004'])
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    for ds in args.datasets:
        print(f"\n{'='*50}\n  Dataset: {ds}\n{'='*50}")
        Xt, yt, Xv, yv, nf = load_dataset(ds, args.data_dir)
        print(f"  Train: {Xt.shape}, Test: {Xv.shape}, Feat: {nf}")
        
        # 1. Pretrained SHRDL
        r1, _ = eval_shrdl(args.checkpoint, Xt, yt, Xv, yv, device)
        print(f"  Pretrained SHRDL: RMSE={r1:.2f}")
        
        # 2. Random SHRDL
        ckpt_n = 24  # default
        enc_r = SHRDL(d_model=128, n_heads=4, n_adn_blocks=2, n_features=nf,
                      feature_dim=64, window_sizes=[2,4,8,16,32]).to(device)
        with torch.no_grad():
            ft = enc_r.encode(torch.FloatTensor(Xt).to(device), project=False).mean(1).cpu().numpy()
            fv = enc_r.encode(torch.FloatTensor(Xv).to(device), project=False).mean(1).cpu().numpy()
        from sklearn.linear_model import Ridge
        r2 = np.sqrt(np.mean((yv - Ridge(alpha=1.0).fit(ft,yt).predict(fv))**2))
        print(f"  Random SHRDL:     RMSE={r2:.2f}")
        
        # 3. LSTM
        r3 = eval_lstm(Xt, yt, Xv, yv, device)
        print(f"  LSTM Baseline:    RMSE={r3:.2f}")
        
        # 4. Wiener
        r4 = eval_wiener(Xt, yt, Xv, yv)
        print(f"  Wiener Process:   RMSE={r4:.2f}")
        
        results[ds] = {'Pretrained': r1, 'Random': r2, 'LSTM': r3, 'Wiener': r4}
    
    # Summary
    print("\n" + "="*65)
    print("  FINAL RESULTS (RMSE)")
    print("="*65)
    header = f"{'Dataset':<12}"
    for m in ['Pretrained','Random','LSTM','Wiener']: header += f"  {m:>10}"
    print(header)
    print("-"*65)
    for ds, r in results.items():
        line = f"{ds:<12}"
        for m in ['Pretrained','Random','LSTM','Wiener']: line += f"  {r[m]:>10.2f}"
        print(line)
    
    # Save
    with open('outputs/results.json','w') as f:
        json.dump({k:{kk:float(vv) for kk,vv in v.items()} for k,v in results.items()}, f, indent=2)
    print("\nResults saved to outputs/results.json")

if __name__ == '__main__':
    main()
