"""Short-horizon temperature forecasting; raw inputs and legacy outputs are read-only.
python forecast.py --project .. --output runs/initial
python forecast.py --project .. --output runs/initial --stage predict --experiment extended --as-of '2026-09-21 13:00'
"""
import argparse, copy, hashlib, json, os, random, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
NODES={
 'S04':('20896478',.680,.625,'open'), 'S05':('20896479',.277,.625,'open'),
 'S31':('21221269',.817,.625,'open'), 'S37':('21221256',.192,.625,'open'),
 'S41':('21221267',.369,.625,'open'), 'N01':('20846624',.378,.444,'open'),
 'N03':('20867860',.296,.445,'open'), 'N05':('20846631',.680,.444,'open'),
 'S11':('21864661',.516,.625,'open'), 'N02':('20867861',.478,.483,'closed'),
 'N15':('20846620',.258,.483,'doorless'), 'N25':('20846621',.773,.480,'doorless')}
EXPERIMENTS={
 'core':{'nodes':list(NODES)[:8], 'bounds':['2026-08-01','2026-09-07','2026-09-14','2026-09-21']},
 'extended':{'nodes':list(NODES), 'bounds':['2026-09-03','2026-09-14','2026-09-17','2026-09-21']}}
MODELS=['STGNN','GRU','LSTM']
MINUTES=np.array([10,30,60]); HORIZONS=MINUTES//10; HISTORY=144
# Preserve the exact long-term origins for a controlled comparison.
COMPARISON_GUARD=144
FEATURES=['temperature_hour_mean','temperature_hour_last','rh_hour_mean','rh_hour_last','hour_sin','hour_cos','weekday_sin','weekday_cos']
def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')
def ingest(project):
    frames=[]; inventory=[]
    for p in sorted((project/'rawdata').rglob('*.csv')):
        m=re.search(r'\((\d+)\)',p.name)
        if not m: continue
        raw=pd.read_csv(p,encoding='utf-8-sig'); t=pd.to_datetime(raw.iloc[:,0],format='mixed',errors='coerce')
        q=pd.DataFrame({'time':t,'serial':m[1],'temp':pd.to_numeric(raw.iloc[:,1],errors='coerce'),'rh':pd.to_numeric(raw.iloc[:,2],errors='coerce'),'phase':2 if '2차' in str(p) else 1})
        valid=q.time.notna() & q.temp.notna() & q.rh.notna()
        inventory.append({'file':str(p.relative_to(project)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(q),'excluded':int((~valid).sum())})
        q=q[valid]; assert (q.time==q.time.dt.floor('10min')).all(),p
        frames.append(q)
    raw=pd.concat(frames,ignore_index=True)
    # Empty event rows have already been removed, so cannot overwrite measurements.
    clean=raw.sort_values('phase',kind='stable').drop_duplicates(['serial','time'],keep='last')
    return clean, inventory, len(raw)-len(clean)
def graph(nodes):
    xy=np.array([NODES[n][1:3] for n in nodes]); dist=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    adj=np.eye(len(nodes),dtype=np.float32)
    for i in range(len(nodes)):
        for j in np.argsort(dist[i])[1:4]:
            kind='closed' if 'N02' in (nodes[i],nodes[j]) else ('doorless' if {'N15','N25'}&{nodes[i],nodes[j]} else 'open')
            # Disjoint prior ranges ensure requested hierarchy for every drawn edge.
            lo,hi={'closed':(.01,.02),'doorless':(.15,.25),'open':(.5,1.)}[kind]
            adj[i,j]=adj[j,i]=lo+(hi-lo)*np.exp(-(dist[i,j]/.25)**2)
    return adj
def make_data(clean,nodes,bounds):
    ts=pd.date_range(bounds[0],pd.Timestamp(bounds[-1])-pd.Timedelta(minutes=10),freq='10min')
    serials=[NODES[n][0] for n in nodes]
    arrays=[clean.pivot(index='time',columns='serial',values=c).reindex(index=ts,columns=serials).to_numpy(dtype=np.float32) for c in ['temp','rh']]
    values=np.stack(arrays,-1)
    if not np.isfinite(values).all(): raise ValueError('Incomplete observations: no interpolation is allowed in these fixed-node experiments.')
    train=values[ts<pd.Timestamp(bounds[1])]; mean=train.mean(0); std=train.std(0).clip(.05)
    return ts,values,mean,std
def origins_for(ts,bounds):
    result={}
    for name,start,end in zip(['train','validation','test'],bounds[:-1],bounds[1:]):
        # Same input and complete horizon set remain strictly within each split.
        result[name]=np.array([i for i in range(HISTORY-1,len(ts)-COMPARISON_GUARD,6)
          if ts[i-HISTORY+1]>=pd.Timestamp(start) and ts[i+COMPARISON_GUARD]<pd.Timestamp(end)],dtype=int)
        if not len(result[name]): raise ValueError('Not enough complete windows for '+name)
    return result
def inputs(values,ts,origins,mean,std):
    scaled=(values-mean)/std
    result=[]
    for i in origins:
        block=scaled[i-HISTORY+1:i+1].reshape(24,6,len(mean),2)
        local=np.concatenate([block.mean(1),block[:,-1]],axis=-1)[:,:, [0,2,1,3]]
        bt=ts[i-HISTORY+6:i+1:6]; hr=bt.hour.to_numpy()+bt.minute.to_numpy()/60
        cal=np.stack([np.sin(2*np.pi*hr/24),np.cos(2*np.pi*hr/24),np.sin(2*np.pi*bt.dayofweek/7),np.cos(2*np.pi*bt.dayofweek/7)],-1)
        result.append(np.concatenate([local,np.repeat(cal[:,None,:],len(mean),axis=1)],-1))
    return np.asarray(result,dtype=np.float32)
class STGNN(nn.Module):
    def __init__(self,n,adj,hidden=24):
        super().__init__(); self.register_buffer('adj',torch.tensor(adj/adj.sum(1,keepdims=True)))
        self.proj=nn.Conv2d(8,hidden,1)
        self.temporal=nn.ModuleList([nn.Conv2d(hidden,hidden,(3,1),dilation=(d,1)) for d in [1,2,4,8]])
        self.spatial=nn.ModuleList([nn.Conv2d(hidden,hidden,1) for _ in range(4)])
        self.head=nn.Conv2d(hidden,len(MINUTES),1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self,x):
        z=torch.relu(self.proj(x.permute(0,3,1,2)))
        for d,t,s in zip([1,2,4,8],self.temporal,self.spatial):
            temporal=torch.relu(t(nn.functional.pad(z,(0,0,2*d,0))))
            z=torch.relu(z+s(torch.einsum('ij,bctj->bcti',self.adj,temporal)))
        return self.head(z[:,:,-1:]).squeeze(2)+x[:,-1,:,1][:,None,:]
class Recurrent(nn.Module):
    def __init__(self,n,kind,hidden=48):
        super().__init__(); self.n=n; self.rnn=getattr(nn,kind)(n*8,hidden,batch_first=True);self.head=nn.Linear(hidden,n*len(MINUTES))
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        z,_=self.rnn(x.flatten(2));return self.head(z[:,-1]).reshape(-1,len(MINUTES),self.n)+x[:,-1,:,1][:,None,:]
def network(name,n,adj): return STGNN(n,adj) if name=='STGNN' else Recurrent(n,name)
def predict(model,x):
    model.eval()
    with torch.no_grad(): return np.concatenate([model(x[i:i+128]).numpy() for i in range(0,len(x),128)])
def fit(name,seed,x,y,mean,std,nodes,adj,out,epochs,patience):
    random.seed(seed); np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    model=network(name,len(nodes),adj);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x['train'],y['train']),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf');state=None;stale=0;history=[];t=time.time()
    for ep in range(1,epochs+1):
        model.train(); total=0
        for xx,yy in loader:
            opt.zero_grad(set_to_none=True);loss=(model(xx)-yy).square().mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item()*len(xx)
        pv=predict(model,x['validation']);score=float(np.abs((pv-y['validation'].numpy())*std[:,0]).mean())
        history.append({'epoch':ep,'train_scaled_mse':total/len(x['train']),'validation_mae_c':score})
        if score<best-1e-5: best=score;state=copy.deepcopy(model.state_dict());best_epoch=ep;stale=0
        else:stale+=1
        if ep==1 or ep%5==0:print(f'{out.name} {name} seed={seed} epoch={ep} validation_MAE={score:.4f} elapsed={time.time()-t:.0f}s',flush=True)
        if stale>=patience:break
    model.load_state_dict(state)
    meta={'name':name,'seed':seed,'best_epoch':best_epoch,'epochs_run':ep,'validation_mae_c':best,'parameters':sum(p.numel() for p in model.parameters()),'elapsed_seconds':time.time()-t}
    torch.save({'state_dict':state,'name':name,'nodes':nodes,'graph':adj.tolist(),'mean':mean.tolist(),'std':std.tolist(),'features':FEATURES,'history_steps':144,'horizon_minutes':MINUTES.tolist(),'metadata':meta},out/f'{name}_{seed}.pt')
    pd.DataFrame(history).to_csv(out/f'{name}_{seed}_training.csv',index=False)
    return model,meta
def evaluate(pred,actual,nodes,model,seed,ts,origins):
    rows=[]
    for hi,h in enumerate(MINUTES):
        target_times=ts[origins+HORIZONS[hi]]
        for group,selector in [('all',np.ones(len(origins),bool)),('weekday',target_times.dayofweek<5),('weekend',target_times.dayofweek>=5),('09-18', (target_times.hour>=9)&(target_times.hour<18)),('other_hours',(target_times.hour<9)|(target_times.hour>=18))]:
            for ni,node in [(-1,'ALL')]+list(enumerate(nodes)):
                e=(pred-actual)[selector,hi];e=e if ni==-1 else e[:,ni]
                if e.size:rows.append({'model':model,'seed':seed,'horizon_minutes':int(h),'node':node,'group':group,'n':e.size,'mae_c':float(np.abs(e).mean()),'rmse_c':float(np.sqrt((e**2).mean()))})
    return rows
def run_experiment(clean,exp,args):
    spec=EXPERIMENTS[exp]; nodes=spec['nodes']; out=args.output/exp;out.mkdir(parents=True,exist_ok=True)
    ts,values,mean,std=make_data(clean,nodes,spec['bounds']); origins=origins_for(ts,spec['bounds']);adj=graph(nodes)
    x={s:torch.from_numpy(inputs(values,ts,o,mean,std)) for s,o in origins.items()}
    actual={s:values[o[:,None]+HORIZONS[None],:,0] for s,o in origins.items()}
    y={s:torch.from_numpy(((v-mean[:,0])/std[:,0]).astype(np.float32)) for s,v in actual.items()}
    dump(out/'design.json',{'experiment':exp,**spec,'history_hours':24,'comparison_guard_hours':24,'matched_long_term_origins':True,'horizon_minutes':MINUTES.tolist(),'origin_stride_minutes':60,'input_encoding':'24 hourly blocks; each retains mean and final temperature/RH plus cyclical hour/weekday','features':FEATURES,'windows':{s:len(o) for s,o in origins.items()},'scaler_train_only':{'mean':mean.tolist(),'std':std.tolist()},'coordinates':{n:NODES[n] for n in nodes},'graph_weights_prior':'N02=.01-.02; N15/N25=.15-.25; open=.5-1; self=1; not physically calibrated'})
    pd.DataFrame(adj,index=nodes,columns=nodes).to_csv(out/'adjacency_prior.csv')
    pd.DataFrame(adj/adj.sum(1,keepdims=True),index=nodes,columns=nodes).to_csv(out/'adjacency_message_passing.csv')
    splitrows=[{'split':s,'origin':ts[i],'input_start':ts[i-143],'target_end':ts[i+max(HORIZONS)],'comparison_guard_end':ts[i+COMPARISON_GUARD]} for s,oo in origins.items() for i in oo]
    pd.DataFrame(splitrows).to_csv(out/'split_windows.csv',index=False)
    metrics=[];meta=[];choices={}
    for name in MODELS:
        records=[]
        for seed in args.seeds:
            model,m=fit(name,seed,x,y,mean,std,nodes,adj,out,args.epochs,args.patience)
            pred=predict(model,x['test'])*std[:,0]+mean[:,0]
            metrics.extend(evaluate(pred,actual['test'],nodes,name,seed,ts,origins['test']));meta.append(m);records.append((m,pred))
            np.savez_compressed(out/f'{name}_{seed}_test.npz',predicted=pred,actual=actual['test'],origins=ts[origins['test']].astype(str).to_numpy(dtype=str),horizon_minutes=MINUTES,nodes=np.array(nodes))
        chosen,pp=min(records,key=lambda t:t[0]['validation_mae_c']); choices[name]=chosen['seed']
        # Fixed chronological midpoint, not selected by test error.
        sample=len(pp)//2;origin=ts[origins['test'][sample]]
        pd.DataFrame([{'model':name,'seed':chosen['seed'],'origin':origin,'target_time':origin+pd.Timedelta(minutes=int(h)),'horizon_minutes':int(h),'node':n,'predicted_c':float(pp[sample,hi,ni]),'actual_c':float(actual['test'][sample,hi,ni])} for hi,h in enumerate(MINUTES) for ni,n in enumerate(nodes)]).to_csv(out/f'{name}_map_values.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);dump(out/'training_metadata.json',meta);dump(out/'selected_seeds.json',choices)
    # Operational forecast beyond latest common observations; no future labels required.
    forecast_latest(clean,exp,args,None)
    print('COMPLETED',exp,flush=True)
def forecast_latest(clean,exp,args,asof):
    out=args.output/exp; choices=json.loads((out/'selected_seeds.json').read_text());rows=[]
    for name in MODELS:
        ck=torch.load(out/f'{name}_{choices[name]}.pt',weights_only=False,map_location='cpu');nodes=ck['nodes']; serials=[NODES[n][0] for n in nodes]
        temp=clean.pivot(index='time',columns='serial',values='temp')[serials]
        end=pd.Timestamp(asof) if asof else temp.dropna().index.max();ts=pd.date_range(end-pd.Timedelta(minutes=1430),end,freq='10min')
        vv=np.stack([clean.pivot(index='time',columns='serial',values=c).reindex(index=ts,columns=serials).to_numpy(dtype=np.float32) for c in ['temp','rh']],-1)
        if not np.isfinite(vv).all():raise ValueError('Forecast input must contain 24 complete hours for all selected sensors')
        mean=np.array(ck['mean']);std=np.array(ck['std']);xx=torch.from_numpy(inputs(vv,ts,np.array([143]),mean,std));model=network(name,len(nodes),np.array(ck['graph'],dtype=np.float32));model.load_state_dict(ck['state_dict']);pr=predict(model,xx)[0]*std[:,0]+mean[:,0]
        for hi,h in enumerate(MINUTES):
            for ni,n in enumerate(nodes): rows.append({'model':name,'seed':choices[name],'origin':end,'target_time':end+pd.Timedelta(minutes=int(h)),'horizon_minutes':int(h),'node':n,'predicted_c':float(pr[hi,ni])})
    target=out/('latest_forecast.csv' if asof is None else 'forecast_'+pd.Timestamp(asof).strftime('%Y%m%d_%H%M')+'.csv');pd.DataFrame(rows).to_csv(target,index=False)
    print('FORECAST',target,flush=True)
def main():
    pa=argparse.ArgumentParser();pa.add_argument('--project',type=Path,default=Path(__file__).resolve().parent.parent);pa.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'runs'/'initial');pa.add_argument('--epochs',type=int,default=30);pa.add_argument('--patience',type=int,default=7);pa.add_argument('--seeds',default='17,29,43');pa.add_argument('--experiment',choices=['core','extended','both'],default='both');pa.add_argument('--stage',choices=['all','predict','report'],default='all');pa.add_argument('--as-of');args=pa.parse_args();args.seeds=[int(s) for s in args.seeds.split(',')];args.project=args.project.resolve();args.output=args.output.resolve();torch.set_num_threads(2)
    if args.stage=='report':
        from render_results import render
        render(args.output);return
    clean,inventory,duplicates=ingest(args.project)
    if args.stage=='predict':
        for exp in (list(EXPERIMENTS) if args.experiment=='both' else [args.experiment]):forecast_latest(clean,exp,args,args.as_of)
        return
    if (args.output/'source_audit.json').exists():raise FileExistsError('Choose a new output folder to preserve previous runs')
    args.output.mkdir(parents=True,exist_ok=True)
    dump(args.output/'source_audit.json',{'files':inventory,'duplicates_removed':duplicates,'unique_valid_rows':len(clean),'timezone_assumption':'Asia/Seoul; source timestamps have no timezone','excluded_nodes':{'S07':'retired before later experiment','S18':'position unverified'},'software':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__},'seeds':args.seeds,'max_epochs':args.epochs,'patience':args.patience})
    for exp in (list(EXPERIMENTS) if args.experiment=='both' else [args.experiment]):run_experiment(clean,exp,args)
    assert all(hashlib.sha256((args.project/r['file']).read_bytes()).hexdigest()==r['sha256'] for r in inventory),'Source files changed during run'
    from render_results import render
    render(args.output)
if __name__=='__main__':main()
