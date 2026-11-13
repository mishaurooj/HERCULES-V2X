#!/usr/bin/env python3
"""Frozen-checkpoint cross-dataset evaluation for GEM-CAN (external, not used for training).

Usage:
python evaluate_gemcan_external.py \
  --archive "results(3).zip" \
  --csv GEM_Attack_Scenario_Raw_Labeled.csv \
  --out gemcan_external_predictions_summary.csv

"""
from __future__ import annotations
import argparse
import io
import json
import time
import zipfile
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,precision_score,
                             recall_score,f1_score,fbeta_score,matthews_corrcoef,
                             roc_auc_score,average_precision_score,brier_score_loss,
                             log_loss,confusion_matrix)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.Sequential(
            torch.nn.Linear(12,32), torch.nn.ReLU(),
            torch.nn.Linear(32,16), torch.nn.ReLU(),
            torch.nn.Linear(16,1))
    def forward(self,x):
        return self.layers(x).reshape(-1)

def parse_hex(v):
    if pd.isna(v):return 0
    s=str(v).strip().replace('0x','').replace('0X','')
    if not s:return 0
    return int(s,16)

def features(df):
    # GEM-CAN labels are not used to construct any feature.
    needed=['timestamp','arbitration_id','dlc']+[f'data{i}' for i in range(8)]+['label']
    miss=[c for c in needed if c not in df.columns]
    if miss:raise ValueError('Missing GEM-CAN columns: '+','.join(miss))
    df=df.copy()
    df['timestamp']=pd.to_numeric(df['timestamp'],errors='raise')
    df['dlc']=pd.to_numeric(df['dlc'],errors='raise')
    df['label']=pd.to_numeric(df['label'],errors='raise')
    if not df['dlc'].between(0,8).all():raise ValueError('DLC outside 0..8')
    if not df['label'].isin([0,1]).all():raise ValueError('Labels outside 0,1')
    # Stable order ensures frequency/interarrival features remain causal.
    df=df.sort_values('timestamp',kind='stable').reset_index(drop=True)
    t=df['timestamp'].to_numpy(dtype=np.float64)
    cols=[df['arbitration_id'].map(parse_hex).to_numpy(dtype=np.float32),
          df['dlc'].to_numpy(dtype=np.float32)]
    for i in range(8):
        values=df[f'data{i}'].map(parse_hex).to_numpy(dtype=np.float32)
        values=np.where(cols[1]>i,values,0)
        cols.append(values)
    base=np.column_stack(cols).astype(np.float32)
    dt=np.r_[0,np.maximum(0,np.diff(t))].astype(np.float32)
    counts={};freq=np.empty(len(base),dtype=np.float32)
    for j,cid in enumerate(base[:,0].astype(np.int64)):
        counts[cid]=counts.get(cid,0)+1
        freq[j]=min(counts[cid],10000)
    x=np.column_stack([base,np.log1p(dt),np.log1p(freq)]).astype(np.float32)
    return df,x,df['label'].to_numpy(dtype=np.int8)

def ece(y,p,nbins=15):
    edges=np.linspace(0,1,nbins+1);bins=np.minimum(np.searchsorted(edges,p,side='right')-1,nbins-1)
    return sum(np.mean(bins==i)*abs(np.mean(p[bins==i])-np.mean(y[bins==i]))
               for i in range(nbins) if (bins==i).any())

def score(y,p,th):
    pred=(p>=th).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    out=dict(n=len(y),normal=int((y==0).sum()),attack=int((y==1).sum()),
             prevalence=float(np.mean(y)),threshold=th,tn=int(tn),fp=int(fp),
             fn=int(fn),tp=int(tp),
             accuracy=float(accuracy_score(y,pred)),
             balanced_accuracy=float(balanced_accuracy_score(y,pred)) if len(set(y))==2 else np.nan,
             precision=float(precision_score(y,pred,zero_division=0)) if (y==1).any() else np.nan,
             recall=float(recall_score(y,pred,zero_division=0)) if (y==1).any() else np.nan,
             f1=float(f1_score(y,pred,zero_division=0)) if (y==1).any() else np.nan,
             f2=float(fbeta_score(y,pred,beta=2,zero_division=0)) if (y==1).any() else np.nan,
             specificity=float(tn/(tn+fp)) if (tn+fp) else np.nan,
             npv=float(tn/(tn+fn)) if (tn+fn) else np.nan,
             fpr=float(fp/(tn+fp)) if (tn+fp) else np.nan,
             fnr=float(fn/(fn+tp)) if (fn+tp) else np.nan,
             false_alarms_per_10k=float(10000*fp/(tn+fp)) if (tn+fp) else np.nan,
             mcc=float(matthews_corrcoef(y,pred)) if len(set(y))==2 else np.nan,
             ece=float(ece(y,p)),
             brier=float(brier_score_loss(y,p)),
             log_loss=float(log_loss(y,p,labels=[0,1])),
             roc_auc=float(roc_auc_score(y,p)) if len(set(y))==2 else np.nan,
             average_precision=float(average_precision_score(y,p)) if len(set(y))==2 else np.nan)
    return out

def predict(model,x):
    a=torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32))
    model.eval()
    with torch.inference_mode():
        return np.concatenate([torch.sigmoid(model(c)).cpu().numpy() for c in a.split(65536)])

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--archive',required=True)
    a.add_argument('--csv',required=True)
    a.add_argument('--out',default='gemcan_external_metrics.csv')
    a.add_argument('--max-rows',type=int,default=0,
                   help='If >0, systematic chronological sample of this many rows; full data by default')
    args=a.parse_args()
    source=pd.read_csv(args.csv,dtype=str)
    if args.max_rows and len(source)>args.max_rows:
        idx=np.linspace(0,len(source)-1,args.max_rows,dtype=int)
        source=source.iloc[idx].copy()
    source,x,y=features(source)
    out=[]
    with zipfile.ZipFile(args.archive) as z:
        scaler=joblib.load(io.BytesIO(z.read('results/models/train_only_scaler.joblib')))
        xs=scaler.transform(x).astype(np.float32)
        ood_rows=int((np.abs(xs)>5).any(axis=1).sum())
        ref=pd.read_csv(io.BytesIO(z.read('results/tables/federated.csv')))
        for algo in ('central','equal','fedavg','fedprox'):
            for seed in (11,22,33,44,55):
                net=Net()
                if algo=='central':
                    ck=torch.load(io.BytesIO(z.read(f'results/models/central_seed{seed}.pt')),
                                  weights_only=True,map_location='cpu')
                    state=ck['state'];th=float(ck['threshold'])
                else:
                    ckname=f'results/models/fl_{algo}_k32_a1.0_s{seed}.pt'
                    state=torch.load(io.BytesIO(z.read(ckname)),
                                     weights_only=True,map_location='cpu')
                    matches=ref.loc[(ref['setting']==f'{algo}_k32_a1.0')&(ref['seed']==seed),'threshold']
                    if len(matches)!=1:raise ValueError(f'No saved threshold: {algo} {seed}')
                    th=float(matches.iloc[0])
                net.load_state_dict(state)
                t0=time.perf_counter();p=predict(net,xs);seconds=time.perf_counter()-t0
                metrics=score(y,p,th)
                metrics.update(dataset='GEM-CAN external test',
                               model=algo,seed=seed,train_dataset='HCRL Car-Hacking',
                               feature_dim=12,ood_gt5sigma_rows=ood_rows,
                               ood_gt5sigma_fraction=ood_rows/len(x),
                               inference_seconds=seconds,
                               inference_frames_per_sec=len(x)/max(seconds,1e-12))
                out.append(metrics)
                if 'Attack_Type' in source.columns:
                    for attack_type,g in source.groupby('Attack_Type',sort=False):
                        ix=g.index.to_numpy()
                        m=score(y[ix],p[ix],th)
                        m.update(dataset=f'GEM-CAN ({attack_type})',model=algo,seed=seed,
                                 train_dataset='HCRL Car-Hacking',feature_dim=12,
                                 ood_gt5sigma_rows=int((np.abs(xs[ix])>5).any(axis=1).sum()))
                        out.append(m)
    result=pd.DataFrame(out)
    result.to_csv(args.out,index=False)
    print('Wrote',args.out,'rows',len(result),'overall evals',len(result[result.dataset=='GEM-CAN external test']))
    print(result[result.dataset=='GEM-CAN external test'].groupby('model')[['accuracy','f1','recall','specificity','fpr']].mean().to_string())
    print('OOD feature diagnostic: rows with any |training-standardized feature| > 5:',
          ood_rows,'/',len(x))
if __name__=='__main__':
    main()
