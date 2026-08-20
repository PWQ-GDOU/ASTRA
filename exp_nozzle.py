"""Experiment: Physics Parameter Fitting + PCG-TCN on COMSOL Nozzle Data"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, sys
sys.path.insert(0, '.')
from src.models.greybox import PCGTCN, PhysicsConstrainedLoss, AblationPhysicsLayer

# ═══════════════════ 1. Load Nozzle COMSOL Data ═══════════════════
data = np.load('data/processed/nozzle_ablation_features.npy')
print(f"Data: {data.shape} (56 time steps, 6 features)")

train_data = data[:40]
val_data = data[-20:]

def create_windows(data, window_size=10):
    X, y = [], []
    n = len(data)
    for i in range(n - window_size):
        X.append(data[i:i+window_size])
        rul = data[-1, 0] - data[i+window_size-1, 0]
        y.append(rul)
    return np.stack(X), np.array(y, dtype=np.float32)

X_train, y_train = create_windows(train_data, 10)
X_val, y_val = create_windows(val_data, 10)
print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Val:   X={X_val.shape}, y={y_val.shape}")

m = X_train.mean((0,1), keepdims=True)
s = X_train.std((0,1), keepdims=True) + 1e-8
X_train = (X_train - m) / s
X_val = (X_val - m) / s

# ═══════════════════ 2. Physics Parameter Fitting ═══════════════════
print("\n=== Physics Parameter Fitting ===")
physics = AblationPhysicsLayer(learn_physics=True)

T_data = torch.FloatTensor(data[:, 1:2])
q_data = torch.FloatTensor(data[:, 3:4])
P_data = torch.FloatTensor(data[:, 2:3])
v_data = torch.FloatTensor(data[:, 4:5])

opt = torch.optim.Adam(physics.parameters(), lr=0.01)
for step in range(5001):
    T_next, v_pred, q_net = physics(T_data[:-1], q_data[:-1], P_data[:-1],
                                     torch.full_like(T_data[:-1], 1147.0))
    loss_T = F.mse_loss(T_next, T_data[1:])
    loss_v = F.mse_loss(v_pred, v_data[1:])
    loss = loss_T + 10.0 * loss_v
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 1000 == 0:
        print(f"  step {step:4d}: loss_T={loss_T.item():.4f}, loss_v={loss_v.item():.2e}")

with torch.no_grad():
    A = torch.exp(physics.log_A).item()
    Ea = physics.Ea.item()
    h = torch.exp(physics.log_h).item()
    eps = torch.sigmoid(physics.epsilon).item()
    print(f"\nFitted Physics Parameters:")
    print(f"  A = {A:.2e} (pre-factor)")
    print(f"  Ea = {Ea:.1f} kJ/mol")
    print(f"  h = {h:.1f} W/m2.K (convection)")
    print(f"  epsilon = {eps:.4f} (emissivity)")

# ═══════════════════ 3. Train PCG-TCN vs Pure TCN ═══════════════════
print("\n=== Training PCG-TCN vs Pure TCN ===")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

for model_name, use_physics in [('Pure TCN', False), ('PCG-TCN', True)]:
    print(f"\n--- {model_name} ---")
    torch.manual_seed(42)
    model = PCGTCN(n_features=6, d_model=64, n_blocks=3, use_physics=use_physics).to(device)

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.FloatTensor(y_train).to(device)
    Xv = torch.FloatTensor(X_val).to(device)
    yv = torch.FloatTensor(y_val).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = PhysicsConstrainedLoss(lambda_phys=0.1 if use_physics else 0)

    best_rmse = 1e9
    for ep in range(2001):
        model.train()
        phys = None
        if use_physics:
            pred, phys = model(Xt, return_physics=True)
        else:
            pred = model(Xt)

        loss, loss_dict = criterion(pred.squeeze(), yt, phys)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 500 == 0:
            model.eval()
            with torch.no_grad():
                if use_physics:
                    pv, _ = model(Xv, return_physics=True)
                else:
                    pv = model(Xv)
                rmse = torch.sqrt(F.mse_loss(pv.squeeze(), yv)).item()
                if rmse < best_rmse:
                    best_rmse = rmse
                print(f"  ep {ep:4d}: loss={loss.item():.4f}, val_rmse={rmse:.4f}, best={best_rmse:.4f}")

    print(f"  => Best RMSE: {best_rmse:.4f} s")

print("\n=== Done ===")
