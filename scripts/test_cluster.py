"""Test cluster-based normalization for multi-condition datasets."""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from shrdl_v4 import RULPredictorV4, train_v4

def load_clustered(dd, fd, seq_len=30, n_clusters=10):
    cd = dd + '/cmapss/6. Turbofan Engine Degradation Simulation Data Set'
    tr = pd.read_csv(f'{cd}/train_{fd}.txt', sep=r'\s+', header=None).values
    te = pd.read_csv(f'{cd}/test_{fd}.txt', sep=r'\s+', header=None).values
    tru = pd.read_csv(f'{cd}/RUL_{fd}.txt', sep=r'\s+', header=None).values.flatten()
    MAX_RUL = 125
    
    # Cluster operating conditions across ALL data
    ops_tr = tr[:,2:5].astype(np.float64)
    ops_te = te[:,2:5].astype(np.float64)
    all_ops = np.concatenate([ops_tr, ops_te])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(all_ops)
    
    def proc(raw):
        Xl, yl = [], []
        for u in np.unique(raw[:,0]):
            t = raw[raw[:,0]==u, 2:].astype(np.float32)
            n = len(t)
            if n < seq_len: continue
            ws = []
            for i in range(n - seq_len + 1):
                w = t[i:i+seq_len]
                ws.append(w)
            if not ws: continue
            X = np.stack(ws)
            # Global normalization
            for c in range(X.shape[-1]):
                mu = X[:,:,c].mean(); sg = X[:,:,c].std() + 1e-8
                X[:,:,c] = (X[:,:,c] - mu) / sg
            rul = np.clip(n - np.arange(seq_len-1, n), 0, MAX_RUL)
            Xl.append(X.astype(np.float32))
            yl.append(rul.astype(np.float32))
        return np.concatenate(Xl), np.concatenate(yl)
    
    Xt, yt = proc(tr)
    Xv_list = []
    for u in np.unique(te[:,0]):
        t = te[te[:,0]==u, 2:].astype(np.float32)
        if len(t) < seq_len: continue
        w = t[-seq_len:]
        Xv_list.append(w)
    Xv = np.stack(Xv_list).astype(np.float32)
    # Normalize test with train stats
    for c in range(Xv.shape[-1]):
        mu = Xt[:,:,c].mean(); sg = Xt[:,:,c].std() + 1e-8
        Xv[:,:,c] = (Xv[:,:,c] - mu) / sg
    yv = tru[:len(Xv_list)].astype(np.float32)
    return Xt, yt, Xv, yv, tr.shape[1]-2

for fd in ['FD002', 'FD004']:
    print(f'\n=== {fd} ===')
    Xt, yt, Xv, yv, nf = load_clustered('data/processed', fd)
    print(f'  Train:{Xt.shape} Test:{Xv.shape}')
    m = RULPredictorV4(nf).to('cuda:2')
    r, s = train_v4(m, Xt, yt, Xv, yv, epochs=200, device='cuda:2')
    print(f'  => RMSE={r:.1f} Score={s:.0f}')
