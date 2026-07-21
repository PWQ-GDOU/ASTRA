import numpy as np, pandas as pd
cd = 'data/processed/cmapss/6. Turbofan Engine Degradation Simulation Data Set'

for fd in ['FD001','FD002']:
    tr = pd.read_csv(f'{cd}/train_{fd}.txt', sep=r'\s+', header=None).values
    ops = tr[:,2:5]
    sensors = tr[:,5:]
    
    # Check: after global normalization, does sensor still vary with condition?
    s_mean = sensors.mean(axis=0); s_std = sensors.std(axis=0) + 1e-8
    sensors_norm = (sensors - s_mean) / s_std
    
    # For FD002, check sensor7 vs op1
    if '002' in fd:
        s7 = sensors[:, 7-1]
        op1 = ops[:,0]
        # Correlation: sensor7 with op1 (condition) vs sensor7 with time (degradation proxy)
        time = tr[:,1]
        print(f'{fd}: sensor7-op1 corr={np.corrcoef(s7,op1)[0,1]:.3f}, sensor7-time corr={np.corrcoef(s7,time)[0,1]:.3f}')
        
        # Within a narrow op1 band, check sensor7 range
        for band, (lo,hi) in enumerate([(0,1),(10,11),(20,21),(30,31),(41,42)]):
            mask = (op1>=lo)&(op1<hi)
            if mask.sum()>10:
                s7b = s7[mask]
                time_b = time[mask]
                # Degradation trend within this band
                if len(s7b)>100:
                    print(f'  op1=[{lo},{hi}): {mask.sum()} samples, s7=[{s7b.min():.0f},{s7b.max():.0f}], s7-time corr={np.corrcoef(s7b,time_b)[0,1]:.3f}')
