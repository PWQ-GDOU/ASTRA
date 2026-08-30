"""
Evaluate pretrained SHRDL encoder on C-MAPSS RUL prediction.
Simple linear probe comparison: pretrained vs random init vs LSTM.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, torch.nn as nn
import numpy as np, pandas as pd
import argparse
from sklearn.linear_model import Ridge

from src.models.shrdl import SHRDL
from src.baselines.lstm_baseline import LSTMBaseline

def load_eval_data(data_dir, fd='FD001', seq_len=64, stride=8):
    """Load C-MAPSS and prepare train/test windows."""
    cmapss_dir = os.path.join(data_dir, 'cmapss', '6. Turbofan Engine Degradation Simulation Data Set')
    train_raw = pd.read_csv(os.path.join(cmapss_dir, f'train_{fd}.txt'), sep=r'\s+', header=None).values
    test_raw = pd.read_csv(os.path.join(cmapss_dir, f'test_{fd}.txt'), sep=r'\s+', header=None).values
    test_rul = pd.read_csv(os.path.join(cmapss_dir, f'RUL_{fd}.txt'), sep=r'\s+', header=None).values.flatten()

    def process(raw, is_test=False):
        X_list, y_list = [], []
        for u in np.unique(raw[:, 0]):
            traj = raw[raw[:, 0] == u, 2:]
            if len(traj) < seq_len: continue
            # Extract windows
            n = len(traj)
            n_feat = traj.shape[1]
            windows = []
            for i in range(0, n - seq_len + 1, stride):
                w = traj[i:i+seq_len]
                windows.append(w)
            if not windows: continue
            X = np.stack(windows)
            X = (X - X.mean()) / (X.std() + 1e-8)
            y = np.arange(len(X))[::-1] * stride  # RUL = remaining steps
            y = np.clip(y, 0, 130).astype(np.float32)
            X_list.append(X.astype(np.float32))
            y_list.append(y)
        return np.concatenate(X_list), np.concatenate(y_list)
    
    X_train, y_train = process(train_raw)
    
    # Test: take last window of each unit
    X_test_list, y_test_list = [], []
    for u in np.unique(test_raw[:, 0]):
        traj = test_raw[test_raw[:, 0] == u, 2:]
        if len(traj) < seq_len: continue
        n_feat = traj.shape[1]
        w = traj[-seq_len:]
        w = (w - w.mean()) / (w.std() + 1e-8)
        X_test_list.append(w)
    X_test = np.stack(X_test_list).astype(np.float32)
    y_test = test_rul[:len(X_test_list)].astype(np.float32)
    
    return X_train, y_train, X_test, y_test, n_feat


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data/processed')
    p.add_argument('--checkpoint', default='checkpoints/shrdl_gpu1_best.pt')
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--fd', default='FD001')
    args = p.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    X_train, y_train, X_test, y_test, n_feat = load_eval_data(args.data_dir, args.fd)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, Features: {n_feat}")
    
    # Load checkpoint to get model feature dim
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    ckpt_n_feat = None
    for k, v in state.items():
        if 'input_proj.0.weight' in k:
            ckpt_n_feat = v.shape[1]
            break
    print(f"Checkpoint features: {ckpt_n_feat}")
    
    # ---- Method 1: Pretrained ----
    print("\n[1] Pretrained SHRDL + Ridge")
    encoder = SHRDL(d_model=128, n_heads=4, n_adn_blocks=2, n_features=ckpt_n_feat,
                    feature_dim=64, window_sizes=[2,4,8,16,32]).to(device)
    model_dict = encoder.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered)
    encoder.load_state_dict(model_dict, strict=False)
    encoder.eval()
    
    # Adapt input if needed
    if n_feat != ckpt_n_feat:
        old = encoder.input_proj
        encoder.input_proj = nn.Sequential(nn.Linear(n_feat, ckpt_n_feat), old).to(device)
    
    with torch.no_grad():
        f_train = encoder.encode(torch.FloatTensor(X_train).to(device), project=False).mean(1).cpu().numpy()
        f_test = encoder.encode(torch.FloatTensor(X_test).to(device), project=False).mean(1).cpu().numpy()
    
    ridge = Ridge(alpha=1.0).fit(f_train, y_train)
    pred1 = ridge.predict(f_test)
    rmse1 = np.sqrt(np.mean((y_test - pred1)**2))
    print(f"  RMSE: {rmse1:.2f}")
    
    # ---- Method 2: Random ----
    print("\n[2] Random SHRDL + Ridge")
    rand = SHRDL(d_model=128, n_heads=4, n_adn_blocks=2, n_features=n_feat,
                 feature_dim=64, window_sizes=[2,4,8,16,32]).to(device)
    rand.eval()
    with torch.no_grad():
        f_train_r = rand.encode(torch.FloatTensor(X_train).to(device), project=False).mean(1).cpu().numpy()
        f_test_r = rand.encode(torch.FloatTensor(X_test).to(device), project=False).mean(1).cpu().numpy()
    ridge2 = Ridge(alpha=1.0).fit(f_train_r, y_train)
    pred2 = ridge2.predict(f_test_r)
    rmse2 = np.sqrt(np.mean((y_test - pred2)**2))
    print(f"  RMSE: {rmse2:.2f}")
    
    # ---- Method 3: LSTM ----
    print("\n[3] LSTM baseline")
    lstm = LSTMBaseline(n_features=n_feat, hidden_dim=128, n_layers=2).to(device)
    opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_test).to(device)
    
    lstm.train()
    for ep in range(80):
        idx = np.random.permutation(len(Xt))
        for i in range(0, len(Xt), 64):
            bi = idx[i:i+64]
            loss = nn.MSELoss()(lstm(Xt[bi]).squeeze(), yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
    
    lstm.eval()
    with torch.no_grad():
        pred3 = lstm(Xv).squeeze().cpu().numpy()
    rmse3 = np.sqrt(np.mean((y_test - pred3)**2))
    print(f"  RMSE: {rmse3:.2f}")
    
    # ---- Summary ----
    print("\n" + "="*55)
    print(f"  C-MAPSS {args.fd} RUL Prediction")
    print("="*55)
    print(f"  {'Method':<30} {'RMSE':>8}")
    print(f"  {'-'*38}")
    print(f"  {'1. Pretrained SHRDL (ours)':<30} {rmse1:>8.2f}")
    print(f"  {'2. Random Init SHRDL':<30} {rmse2:>8.2f}")
    print(f"  {'3. LSTM Baseline':<30} {rmse3:>8.2f}")
    
    if rmse2 > 0:
        imp = (rmse2 - rmse1) / rmse2 * 100
        print(f"\n  Pretraining gain: {imp:.1f}% over random init")


if __name__ == '__main__':
    main()
