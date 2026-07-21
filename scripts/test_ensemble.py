"""Ensemble test: 3 models with different seeds, average predictions."""
import torch, numpy as np
from shrdl_pro_v2 import RULPredictor, load_cmapss_std, compute_score, train_model_pro

device = 'cuda:1'
config = {'epochs':200,'lr':5e-4,'weight_decay':3e-3,'alpha':0.3,'batch_size':256}

for fd in ['FD001','FD003']:
    print(f'\n=== {fd} Ensemble ===')
    Xt, yt, Xv, yv, nf = load_cmapss_std('data/processed', fd)
    
    single_results = []
    all_preds = []
    for seed in [42, 123, 456]:
        print(f'  Seed {seed}...')
        np.random.seed(seed); torch.manual_seed(seed)
        m = RULPredictor(n_features=nf).to(device)
        r, s, _ = train_model_pro(m, Xt, yt, Xv, yv, config, device)
        single_results.append(r)
        m.eval()
        with torch.no_grad():
            p = m(torch.FloatTensor(Xv).to(device)).squeeze().cpu().numpy()
        all_preds.append(p)
        print(f'    RMSE={r:.1f}, Score={s:.0f}')
    
    ens = np.mean(all_preds, axis=0)
    ens_rmse = np.sqrt(np.mean((yv-ens)**2))
    ens_score = compute_score(yv, ens)
    best_single = min(single_results)
    print(f'  Best single: {best_single:.1f}  ->  Ensemble: {ens_rmse:.1f}  ({(1-ens_rmse/best_single)*100:.0f}% better)')
    print(f'  Score: {ens_score:.0f}')
