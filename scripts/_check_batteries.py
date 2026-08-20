import os, glob, sys
sys.path.insert(0, '/data/disk3/spacecraft_rul')
from src.data.battery_strict import load_battery_mat, materialize_battery
base = 'data/processed/nasa_battery/5. Battery Data Set'
mats = sorted(glob.glob(os.path.join(base, 'B0*.mat')))
print(f'Found {len(mats)} batteries')
event_cells, censored_cells = [], []
for m in mats:
    try:
        raw = load_battery_mat(m)
        s = materialize_battery(raw, protocol='rel80')
        name = os.path.basename(m).replace('.mat','')
        print(name, f'init={raw.init_capacity:.3f}Ah', f'eol={s.eol_ah:.3f}Ah',
              f'cycles={len(s)}', s.status)
        if s.status == 'event_observed': event_cells.append(name)
        else: censored_cells.append(name)
    except Exception as e:
        print(os.path.basename(m), 'ERROR:', e)
print(f'\nEvent-observed: {len(event_cells)}, Right-censored: {len(censored_cells)}')
print('Event cells:', event_cells[:10], '...' if len(event_cells)>10 else '')
print('Censored:', censored_cells)
