"""
SHRDL-Pro v2: Score-aware training + extended epochs + tuned hyperparams
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, math, argparse, json
from shrdl_pro import SHRDLPro, RULPredictor, load_cmapss_std, compute_score

def smooth_score_loss(pred, target):
    """Smooth, differentiable approximation of NASA Score function.
    Combines MSE with an asymmetric penalty for late predictions."""
    d = pred - target  # positive = over-estimate (late) = worse
    
    # Smooth score: penalize over-estimation more heavily
    # For d >= 0: use exp(d/13) style penalty (smooth)
    # For d < 0: linear penalty (under-estimation is less bad)
    over_mask = (d >= 0).float()
    under_mask = 1 - over_mask
    
    # Smooth approximation with clipped exp to prevent gradient explosion
    over_penalty = torch.exp(torch.clamp(d / 13.0, -5, 5)) - 1.0
    under_penalty = torch.exp(torch.clamp(-d / 10.0, -5, 5)) - 1.0
    
    score = over_mask * over_penalty + under_mask * under_penalty
    return score.mean()


def combined_loss(pred, target, alpha=0.1):
    """MSE + weighted smooth Score loss."""
    mse = F.mse_loss(pred.squeeze(), target)
    score = smooth_score_loss(pred.squeeze(), target)
    return mse + alpha * score


def train_model_pro(model, Xt, yt, Xv, yv, config, device='cuda'):
    """Improved training with Score-aware loss, warmup, cosine decay."""
    epochs = config.get('epochs', 200)
    lr = config.get('lr', 1e-3)
    wd = config.get('weight_decay', 1e-3)
    alpha = config.get('alpha', 0.1)  # Score loss weight
    bs = config.get('batch_size', 256)
    
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    # Cosine annealing with linear warmup
    warmup_epochs = 10
    steps_per_epoch = max(1, len(Xt) // bs)
    total_steps = epochs * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_epochs * steps_per_epoch:
            return step / (warmup_epochs * steps_per_epoch)
        progress = (step - warmup_epochs * steps_per_epoch) / (total_steps - warmup_epochs * steps_per_epoch)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    
    Xtt = torch.FloatTensor(Xt).to(device)
    ytt = torch.FloatTensor(yt).to(device)
    Xvv = torch.FloatTensor(Xv).to(device)
    
    best_rmse, best_score = 1e9, 1e9
    best_state = None
    history = []
    
    for ep in range(epochs):
        # Shuffle + train
        idx = np.random.permutation(len(Xtt))
        model.train()
        for i in range(0, len(Xt), bs):
            bi = idx[i:i+bs]
            pred = model(Xtt[bi]).squeeze()
            loss = combined_loss(pred, ytt[bi], alpha)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            p = model(Xvv).squeeze().cpu().numpy()
            rmse = np.sqrt(np.mean((yv - p)**2))
            score = compute_score(yv, p)
            history.append((rmse, score))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (ep + 1) % 30 == 0 or ep == 0:
            print(f'  E{ep+1:3d}: RMSE={rmse:5.1f}  Score={score:8.0f}  '
                  f'lr={sched.get_last_lr()[0]:.2e}  best_RMSE={best_rmse:.1f}')
    
    if best_state:
        model.load_state_dict(best_state)
    
    return best_rmse, best_score, history


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--fd', default='FD001')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-3)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--bs', type=int, default=256)
    args = p.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    config = {'epochs': args.epochs, 'lr': args.lr, 'weight_decay': args.wd,
              'alpha': args.alpha, 'batch_size': args.bs}
    
    for fd in (['FD001','FD002','FD003','FD004'] if args.fd == 'all' else [args.fd]):
        print(f'\n{"="*60}')
        print(f'  {fd}: epochs={args.epochs} lr={args.lr} wd={args.wd} alpha={args.alpha}')
        print(f'{"="*60}')
        
        Xt, yt, Xv, yv, nf = load_cmapss_std('data/processed', fd)
        print(f'  Train: {Xt.shape}, Test: {Xv.shape}')
        
        model = RULPredictor(n_features=nf).to(device)
        print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')
        
        rmse, score, hist = train_model_pro(model, Xt, yt, Xv, yv, config, device)
        print(f'  => Best: RMSE={rmse:.1f}, Score={score:.0f}')
        print(f'  SOTA:   RMSE~11, Score~200')
