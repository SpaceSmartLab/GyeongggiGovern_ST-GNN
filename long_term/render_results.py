"""Plan overlay and zone-separated RBF visualization. Not a physical CFD solver."""
from pathlib import Path
import json,base64,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import Rectangle
from PIL import Image
from forecast import NODES,MODELS,HOURS
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10,'savefig.dpi':130})
ASSET=Path(__file__).parent/'assets'/'floorplan.png'
# Approximate image-coordinate zone envelopes, not a surveyed CAD reconstruction.
ROOMS={'N02':(.445,.385,.115,.125),'N15':(.245,.465,.030,.045),'N25':(.760,.460,.035,.050)}
FOOTPRINT=np.array([[.045,.390],[.953,.390],[.865,.855],[.80,.810],[.70,.780],[.60,.760],[.50,.760],[.40,.770],[.30,.790],[.20,.820],[.158,.835],[.105,.610],[.033,.550]])
def hull(points):
    pts=sorted(set(map(tuple,points)))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo=[];up=[]
    for p in pts:
        while len(lo)>=2 and cross(lo[-2],lo[-1],p)<=0:lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up)>=2 and cross(up[-2],up[-1],p)<=0:up.pop()
        up.append(p)
    return np.array(lo[:-1]+up[:-1])
def field(nodes,values):
    xx,yy=np.meshgrid(np.linspace(.025,.97,310),np.linspace(.375,.865,170));query=np.c_[xx.ravel(),yy.ravel()]
    indices=[i for i,n in enumerate(nodes) if n not in ROOMS];xy=np.array([NODES[nodes[i]][1:3] for i in indices]);v=np.asarray(values)[indices]
    scale=.19; k=np.exp(-((xy[:,None]-xy[None])**2).sum(-1)/(2*scale**2)); center=v.mean()
    w=np.linalg.solve(k+np.eye(len(k))*1e-5,v-center)
    z=center+np.exp(-((query[:,None]-xy[None])**2).sum(-1)/(2*scale**2))@w
    inside=MPath(FOOTPRINT).contains_points(query);supported=MPath(hull(xy)).contains_points(query)
    for node,(x,y,ww,hh) in ROOMS.items():
        region=(query[:,0]>=x)&(query[:,0]<=x+ww)&(query[:,1]>=y)&(query[:,1]<=y+hh)
        if node in nodes:z[region]=values[nodes.index(node)];supported[region]=True
        else:inside[region]=False
    # Stair/storage zones are not target occupied space.
    for x1,x2 in [(.17,.274),(.76,.87)]:inside[(query[:,0]>x1)&(query[:,0]<x2)&(query[:,1]<.465)]=False
    z[~inside]=np.nan
    return xx,yy,z.reshape(xx.shape),supported.reshape(xx.shape)
def base(ax):
    img=Image.open(ASSET);ax.imshow(img,extent=(0,1,1,0),alpha=.48);ax.set_aspect(img.height/img.width);ax.set_xlim(0,1);ax.set_ylim(1,0);ax.axis('off')
    # Replace old equipment terminology on the supplied plan.
    ax.add_patch(Rectangle((.003,.348),.33,.045,facecolor='white',edgecolor='none',zorder=5))
    ax.text(.012,.373,'녹색 표시: 천장형 팬코일 유니트(FCU)',color='green',fontsize=9,zorder=6)
def contour(nodes,values,path,title,limits):
    fig,ax=plt.subplots(figsize=(14,8.8));base(ax);xx,yy,z,supported=field(nodes,values)
    levels=np.linspace(*limits,17)
    # Outside the sensor convex hull is extrapolation, rendered with weaker opacity.
    cs=ax.contourf(xx,yy,z,levels=levels,cmap='coolwarm',alpha=.30,extend='both',zorder=2)
    ax.contourf(xx,yy,np.ma.masked_where(~supported,z),levels=levels,cmap='coolwarm',alpha=.60,extend='both',zorder=3)
    if np.nanmax(z)-np.nanmin(z)>.01:ax.contour(xx,yy,z,levels=levels[::2],colors='#555',linewidths=.35,alpha=.45,zorder=4)
    for n,v in zip(nodes,values):
        x,y=NODES[n][1:3];ax.scatter(x,y,c='black',s=22,edgecolors='white',zorder=7)
        ax.annotate(f'{n}\n{v:.2f}°C',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8,bbox={'facecolor':'white','alpha':.9,'edgecolor':'#888','pad':2},zorder=8)
    for n,(x,y,w,h) in ROOMS.items():ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#555',linestyle='--',linewidth=.8,zorder=6))
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(*limits),cmap='coolwarm'),ax=ax,fraction=.025,pad=.01,label='예측 온도 (°C)',extend='both')
    ax.set_title(title,fontsize=13,pad=10)
    fig.text(.06,.035,'센서점: 모델 예측 / 색상: 구획별 RBF 공간 추정 / 옅은 영역: 센서 배치 범위 밖 외삽\n점선 사각형: 임시 구획 경계(실측 CAD 아님). 단일 센서 구획은 균일 온도로 표시. 빨간 표시: 바닥공조.',fontsize=9)
    fig.savefig(path,bbox_inches='tight');plt.close(fig)
def draw_graph(nodes,adj,path):
    fig,ax=plt.subplots(figsize=(14,8.8));base(ax)
    for i,a in enumerate(nodes):
        for j in range(i):
            if adj[i,j]==0:continue
            b=nodes[j];closed='N02' in (a,b);doorless=bool({'N15','N25'}&{a,b})
            ax.plot([NODES[a][1],NODES[b][1]],[NODES[a][2],NODES[b][2]],ls=':' if closed else '-',lw=.8 if closed else (1.7 if doorless else 2.5),alpha=.4 if closed else (.75 if doorless else .85),color='#a26420' if closed else ('#377fa2' if doorless else '#264b72'),zorder=3)
    for n in nodes:
        x,y=NODES[n][1:3];ax.scatter(x,y,s=130,c='#e8f0f6',edgecolors='#244b70',zorder=4);ax.text(x,y,n,fontsize=7,ha='center',va='center',zorder=5)
    ax.set_title('공간관계 사전 가중치 · N02 가장 낮음 / 문 없는 N15·N25 중간',fontsize=14)
    fig.text(.08,.04,'N02 외부 연결 0.01–0.02 (옅은 점선)  |  N15·N25 외부 연결 0.15–0.25 (실선)  |  개방공간 0.50–1.00\n자기 연결=1(그림 생략). 학습 시 행 정규화. 선은 공기 흐름·덕트가 아닌 모델 연결이며, 물리 계측 가중치가 아닙니다.',fontsize=10)
    fig.savefig(path,bbox_inches='tight');plt.close(fig)
def embedded(path):return '<img loading="lazy" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'
def render(root):
    sections=[];summary=[]
    for exp in ['core','extended']:
        out=root/exp
        if not (out/'metrics.csv').exists():continue
        plots=out/'plots';plots.mkdir(exist_ok=True);design=json.loads((out/'design.json').read_text());nodes=design['nodes'];adj=pd.read_csv(out/'adjacency_prior.csv',index_col=0).to_numpy()
        draw_graph(nodes,adj,plots/'graph.png')
        metric=pd.read_csv(out/'metrics.csv');main=metric[(metric.node=='ALL')&(metric.group=='all')]
        agg=main.groupby(['model','horizon_hours']).agg(mae_mean=('mae_c','mean'),mae_sd=('mae_c','std'),rmse_mean=('rmse_c','mean'),rmse_sd=('rmse_c','std')).reset_index();agg.to_csv(out/'metrics_summary.csv',index=False)
        fig,ax=plt.subplots(figsize=(9,4.5))
        for name in MODELS:
            q=agg[agg.model==name];ax.errorbar(q.horizon_hours,q.mae_mean,yerr=q.mae_sd.fillna(0),marker='o',capsize=4,label=name)
        ax.set_xticks(HOURS);ax.set_xlabel('예측 시간 (시간)');ax.set_ylabel('시험 MAE (°C)');ax.set_title(f'{exp}: 초기값별 평균 ± 표준편차 (신뢰구간 아님)');ax.legend();ax.grid(alpha=.2);fig.savefig(plots/'metrics.png',bbox_inches='tight');plt.close(fig)
        maps={name:pd.read_csv(out/f'{name}_map_values.csv') for name in MODELS}; latest=pd.read_csv(out/'latest_forecast.csv')
        # Shared colour limits across every model/horizon, using train-only temperature scale.
        mu=np.array(design['scaler_train_only']['mean'])[:,0];sd=np.array(design['scaler_train_only']['std'])[:,0];limits=(float(np.floor((mu-2*sd).min()*2)/2),float(np.ceil((mu+2*sd).max()*2)/2))
        for name in MODELS:
            for h in HOURS:
                q=maps[name][maps[name].horizon_hours==h].set_index('node').loc[nodes]
                contour(nodes,q.predicted_c.to_numpy(),plots/f'{name}_{h}h.png',f'{name} / {exp} / +{h}시간 / 기준 {q.origin.iloc[0]} → {q.target_time.iloc[0]}',limits)
                q=latest[(latest.model==name)&(latest.horizon_hours==h)].set_index('node').loc[nodes]
                contour(nodes,q.predicted_c.to_numpy(),plots/f'latest_{name}_{h}h.png',f'{name} / 최신 관측 이후 예측(+{h}h) / {q.target_time.iloc[0]} / 정답 미확보',limits)
        # Actual/predicted time traces for validation-selected model seed, all test origins.
        chosen=json.loads((out/'selected_seeds.json').read_text());fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
        for ax,name in zip(axes,MODELS):
            z=np.load(out/f'{name}_{chosen[name]}_test.npz');times=pd.to_datetime(z['origins'])+pd.Timedelta(hours=24)
            ax.plot(times,z['actual'][:,4].mean(1),c='black',label='실측 센서 평균');ax.plot(times,z['predicted'][:,4].mean(1),label=f'{name} +24h 센서 평균');ax.set_ylabel('온도 °C');ax.legend();ax.grid(alpha=.2)
        fig.autofmt_xdate();fig.tight_layout();fig.savefig(plots/'test_traces_24h.png',bbox_inches='tight');plt.close(fig)
        section=f'<h2>{exp}: {len(nodes)}개 센서</h2><p>학습/검증/시험 경계: {design["bounds"]}<br>유효 창 수: {design["windows"]}. 24시간 입력·24시간 최대 목표가 각 분할 안에 모두 포함됩니다. 예측 기준 시각 간격 1시간.</p>'
        section+=embedded(plots/'metrics.png')+agg.round(4).to_html(index=False)+embedded(plots/'test_traces_24h.png')
        section+='<p>표의 평균·표준편차는 무작위 초기값별 시험 성능입니다. 지도는 검증 MAE로 선택한 초기값을 사용하며, 시험 오차로 시각이나 모델을 고르지 않았습니다. 공통 색상 범위는 학습 온도 통계로 고정했습니다.</p>'+embedded(plots/'graph.png')
        section+='<h3>ST-GNN 예측 contour map</h3>'
        for h in HOURS:section+=f'<details {"open" if h==24 else ""}><summary>+{h}시간 시험 예측</summary>'+embedded(plots/f'STGNN_{h}h.png')+'</details>'
        section+='<h3>최신 공통 관측 이후 24시간 예측</h3>'+embedded(plots/'latest_STGNN_24h.png')
        section+=f'<p><a href="{exp}/plots/">전체 모델·예측시점 지도 PNG</a> / <a href="{exp}/metrics.csv">센서·시간대별 시험 지표</a> / <a href="{exp}/latest_forecast.csv">최신 예측 수치</a> / <a href="{exp}/design.json">실험 설계</a></p>'
        sections.append(section)
        ss=agg.copy();ss['experiment']=exp;summary.append(ss)
    pd.concat(summary).to_csv(root/'comparison.csv',index=False)
    text='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>장기 온도 예측 결과</title><style>body{font-family:'Malgun Gothic',sans-serif;background:#eef2f5;color:#243447;line-height:1.8}main{max-width:1150px;margin:auto;background:white;padding:36px}img{max-width:100%}h1,h2{color:#163e5c}h2{border-top:2px solid #d5e2ec;padding-top:24px}table{border-collapse:collapse;font-size:13px}td,th{padding:7px 12px;border:1px solid #dce5ed}details{border:1px solid #dce5ed;padding:10px;margin:12px 0}a{color:#1671a5}.note{background:#fff4d7;padding:18px}</style><main><h1>ST-GNN·GRU·LSTM 장기 온도 예측</h1><p>동일 입력·동일 분할·동일 1/3/6/12/24시간 목표로 비교했습니다. 과거 24시간의 10분 자료를 시간별 평균과 마지막 값으로 요약한 24개 시점을 사용합니다. 온도·습도와 시각·요일만 입력하며 미래 관측값·재실 스냅샷·CO₂는 사용하지 않았습니다.</p><div class="note">12개 센서 실험은 18일 자료에 기반한 탐색 실험입니다. 엄격한 24시간 입력/목표 분리로 검증·시험 창이 적습니다. 초기값별 편차는 새로운 날짜에 대한 불확실성을 나타내지 않습니다. 기존에 탐색 분석한 기간이므로 완전한 블라인드 시험도 아닙니다. 별도 계절·외부 관측에서 추가 평가가 필요합니다.<br>색상장은 센서 예측의 공간 보간이며 전체 공간의 실측 정답이 아닙니다. 구획 경계와 센서 좌표는 도면을 읽은 근사치입니다. 실제 출입구 좌표·풍량·FCU 운전정보가 없어 공기 흐름을 재현하지 않습니다. 15·25번 문 없음은 그래프 가중치에 반영했습니다. 공조 위치는 배경 도면에 표시하되 수치 입력으로 추정해 넣지 않았습니다.</div>'''+''.join(sections)+'</main></html>'
    (root/'report.html').write_text(text,encoding='utf-8');print('REPORT',root/'report.html',flush=True)
