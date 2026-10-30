"""Independent checks of saved metrics, chronology, inference and source hashes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from forecast import ingest,inputs,network,NODES,MODELS
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path('..'));p.add_argument('--output',type=Path,default=Path('runs/initial'));a=p.parse_args();torch.set_num_threads(2)
audit=json.loads((a.output/'source_audit.json').read_text(encoding='utf-8'));clean,_,_=ingest(a.project)
for item in audit['files']:assert hashlib.sha256((a.project/item['file']).read_bytes()).hexdigest()==item['sha256']
checked=0
for exp in ['core','extended']:
    out=a.output/exp;design=json.loads((out/'design.json').read_text());metrics=pd.read_csv(out/'metrics.csv');windows=pd.read_csv(out/'split_windows.csv',parse_dates=['origin','input_start','target_end'])
    for split,start,end in zip(['train','validation','test'],design['bounds'][:-1],design['bounds'][1:]):
        q=windows[windows.split==split];assert len(q)==design['windows'][split];assert (q.input_start>=pd.Timestamp(start)).all() and (q.target_end<pd.Timestamp(end)).all()
    meta=json.loads((out/'training_metadata.json').read_text());chosen=json.loads((out/'selected_seeds.json').read_text());latest=pd.read_csv(out/'latest_forecast.csv')
    for name in MODELS:
        assert chosen[name]==min([r for r in meta if r['name']==name],key=lambda r:r['validation_mae_c'])['seed']
        for seed in audit['seeds']:
            z=np.load(out/f'{name}_{seed}_test.npz');assert np.isfinite(z['predicted']).all();assert z['predicted'].shape==z['actual'].shape
            for i,h in enumerate(z['horizon_minutes']):
                row=metrics[(metrics.model==name)&(metrics.seed==seed)&(metrics.horizon_minutes==h)&(metrics.node=='ALL')&(metrics.group=='all')].iloc[0]
                error=z['predicted'][:,i].astype('float64')-z['actual'][:,i].astype('float64');np.testing.assert_allclose(np.abs(error).mean(),row.mae_c,rtol=1e-6);np.testing.assert_allclose(np.sqrt((error**2).mean()),row.rmse_c,rtol=1e-6)
            checked+=1
        ck=torch.load(out/f'{name}_{chosen[name]}.pt',map_location='cpu',weights_only=False);nodes=ck['nodes'];serials=[NODES[n][0] for n in nodes];q=latest[latest.model==name];end=pd.Timestamp(q.origin.iloc[0]);ts=pd.date_range(end-pd.Timedelta(minutes=1430),end,freq='10min');vals=np.stack([clean.pivot(index='time',columns='serial',values=c).reindex(index=ts,columns=serials).to_numpy(dtype='float32') for c in ['temp','rh']],-1);mu=np.array(ck['mean']);sd=np.array(ck['std']);x=torch.from_numpy(inputs(vals,ts,np.array([143]),mu,sd));m=network(name,len(nodes),np.array(ck['graph'],dtype='float32'));m.load_state_dict(ck['state_dict']);m.eval()
        with torch.no_grad(): pred=m(x).numpy()[0]*sd[:,0]+mu[:,0]
        for i,h in enumerate(ck['horizon_minutes']):np.testing.assert_allclose(pred[i],q[q.horizon_minutes==h].set_index('node').loc[nodes].predicted_c,atol=1e-5)
    assert len(list((out/'plots').glob('*.png')))==21
result={'status':'PASS','source_hashes':len(audit['files']),'test_prediction_files_recomputed':checked,'checkpoint_inference_cases':6,'checks':['split containment','validation-only seed selection','all horizon MAE/RMSE from saved arrays','checkpoint reload equals saved operational forecast','42 plots exist','raw source hashes unchanged']}
(a.output/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
