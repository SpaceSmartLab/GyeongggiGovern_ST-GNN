"""Verify identical long/short experimental conditions, without rewriting long outputs."""
import json,hashlib,argparse
from pathlib import Path
import numpy as np,pandas as pd,torch
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('runs/matched_long_term'));a=p.parse_args()
long=a.project/'long_term'/'runs'/'initial';short=a.output;results={}
for exp in ['core','extended']:
    l=long/exp;s=short/exp;ld=json.loads((l/'design.json').read_text());sd=json.loads((s/'design.json').read_text())
    for key in ['nodes','bounds','history_hours','origin_stride_minutes','input_encoding','features','windows','scaler_train_only','coordinates','graph_weights_prior']:assert ld[key]==sd[key],(exp,key)
    for f in ['adjacency_prior.csv','adjacency_message_passing.csv']:assert (l/f).read_bytes()==(s/f).read_bytes()
    lw=pd.read_csv(l/'split_windows.csv');sw=pd.read_csv(s/'split_windows.csv');pd.testing.assert_frame_equal(lw[['split','origin','input_start']],sw[['split','origin','input_start']]);assert (sw.comparison_guard_end==lw.target_end).all()
    for model in ['STGNN','GRU','LSTM']:
        for seed in [17,29,43]:
            lc=torch.load(l/f'{model}_{seed}.pt',weights_only=False,map_location='cpu');sc=torch.load(s/f'{model}_{seed}.pt',weights_only=False,map_location='cpu')
            assert lc['features']==sc['features'];assert lc['history_steps']==sc['history_steps'];assert sc['horizon_minutes']==[10,30,60]
            for k,v in lc['state_dict'].items():
                if not k.startswith('head.'):assert v.shape==sc['state_dict'][k].shape
            lz=np.load(l/f'{model}_{seed}_test.npz');sz=np.load(s/f'{model}_{seed}_test.npz');np.testing.assert_array_equal(lz['origins'],sz['origins']);np.testing.assert_array_equal(lz['actual'][:,0,:],sz['actual'][:,2,:])
    results[exp]={'identical_input_scaler_graph_dates_and_origins':True,'shared_60min_target_identical':True,'windows':sd['windows']}
la=json.loads((long/'source_audit.json').read_text(encoding='utf-8'));sa=json.loads((short/'source_audit.json').read_text(encoding='utf-8'))
for key in ['files','duplicates_removed','unique_valid_rows','seeds','max_epochs','patience']:assert la[key]==sa[key],key
before=Path(__file__).parent/'long_term_before.json'
if before.exists():
    for rel,h in json.loads(before.read_text()).items():assert hashlib.sha256((a.project/'long_term'/rel).read_bytes()).hexdigest()==h,rel
results['status']='PASS';results['long_term_files_unchanged']=before.exists()
(short/'protocol_comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results))
