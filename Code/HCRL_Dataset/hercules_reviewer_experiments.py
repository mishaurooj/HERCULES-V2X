#!/usr/bin/env python3
"""HERCULES-V2X reviewer-revision experiment runner.

Reviewer validation v4. Reproducible *new reference implementation*, not a byte-for-byte conversion
of the upstream notebook. Never reports synthetic data as scientific results.

Python 3.10+; pandas numpy scikit-learn scipy matplotlib torch psutil joblib.
Run: python hercules_reviewer_experiments.py --data-root 'E:\\other\\hercules\\dataset' --out 'E:\\other\\hercules\\results' --mode full
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, random, sys, time, tracemalloc, warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
from scipy.stats import bootstrap
import joblib

SEED_DEFAULT=[11,22,33,44,55]
VERSION='7.0.0-reviewer-validation'
# NEVER balance or oversample validation/test data.

ACTIVE_DEVICE='cpu'
ACTIVE_BALANCE='batch'
FEATURES=['can_id','dlc']+[f'byte_{i}' for i in range(8)]+['dt','id_frequency']
ATTACK_NAMES={'dos':'DoS','fuzzy':'Fuzzy','fuzz':'Fuzzy','gear':'Gear','rpm':'RPM',
              'replay':'Replay','spoof':'Spoofing','normal':'Normal'}

@dataclass
class Config:
    data_root:str='dataset'
    out:str='results'
    mode:str='full'
    seeds:tuple[int,...]=(11,22,33,44,55)
    max_rows_per_file:int=300000
    epochs:int=12
    rounds:int=12
    local_epochs:int=1
    batch:int=1024
    lr:float=0.001
    patience:int=4
    client_sizes:tuple[int,...]=(2,4,8,16,32)
    dirichlet_alphas:tuple[float,...]=(10.,1.,0.5,0.1)
    privacy_sigmas:tuple[float,...]=(0.,0.001,0.005,0.01,0.02)
    max_runtime_configs:int=120
    device:str='auto'
    balance_mode:str='batch'
    strict_audit:bool=True
    active_time_audit:bool=True
    novelty_fpr_grid:tuple[float,...]=(0.0001,0.001,0.005,0.01,0.05)
    novelty:bool=True


def mkdirs(root:Path):
    for n in ['tables','figures','models','metrics','logs','reproducibility']:
        (root/n).mkdir(parents=True,exist_ok=True)


def json_save(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    def default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)): return o.tolist()
        if isinstance(o, Path): return str(o)
        raise TypeError(str(type(o)))
    path.write_text(json.dumps(obj,indent=2,default=default,allow_nan=False),encoding='utf8')


def seed_all(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)


def sha256(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def decode_hex(v, default=0):
    if pd.isna(v):return default
    s=str(v).strip().replace('0x','').replace('0X','')
    if s in ('','-','nan','None'):return default
    try:return int(s,16)
    except (ValueError,TypeError):return default


def select_files(data_root:Path):
    names=['DoS_dataset.csv','Fuzzy_dataset.csv','gear_dataset.csv','RPM_dataset.csv']
    files=[data_root/n for n in names if (data_root/n).is_file()]
    if not files:
        raise FileNotFoundError(f'Expected HCRL files not found in {data_root}')
    if len(files)!=len(names):
        raise FileNotFoundError('Missing required attack recordings: '+', '.join(n for n in names if not (data_root/n).is_file()))
    print('Dataset policy: only 4 attack CSV recordings; benign=R from the SAME recordings; normal_run_data.txt excluded.',flush=True)
    return files


def detect_attack_name(file:Path):
    stem=file.stem.lower()
    for k,v in ATTACK_NAMES.items():
        if k in stem:return v
    return 'Unknown'


def parse_frame(chunk:pd.DataFrame,path:Path):
    """Parse headerless HCRL CSV with variable DLC and trailing R/T labels."""
    attack_name=detect_attack_name(path)
    if attack_name=='Unknown':
        raise ValueError(f'Unknown scenario filename: {path.name}')
    if chunk.shape[1] < 4:
        raise ValueError(f'Invalid HCRL recording {path.name}: {chunk.shape[1]} columns')
    # CSV rows contain timestamp, hexadecimal CAN ID, DLC, DLC payload bytes,
    # and then R/T. For DLC<8 the flag is NOT necessarily in column 11.
    dlc_raw=pd.to_numeric(chunk.iloc[:,2],errors='coerce')
    valid_dlc=dlc_raw.between(0,8)
    t=pd.to_numeric(chunk.iloc[:,0],errors='coerce').to_numpy(dtype=np.float64)
    dlc=dlc_raw.fillna(0).clip(0,8).to_numpy(dtype=np.int32)
    # Rightmost nonempty token is the label, regardless of payload length.
    tail=chunk.iloc[:,3:].astype('string').apply(lambda c:c.str.strip())
    labels=tail.ffill(axis=1).iloc[:,-1].str.upper()
    good=labels.isin(['R','T'])
    invalid=~good
    if invalid.any():
        examples=labels[invalid].fillna('<missing>').unique()[:5].tolist()
        raise ValueError(f'Invalid R/T labels in {path.name}: {examples}')
    y=labels.eq('T').to_numpy(dtype=np.int8)
    cid=chunk.iloc[:,1].map(decode_hex).to_numpy(dtype=np.float32)
    values=[]
    for i in range(8):
        if 3+i < chunk.shape[1]:
            v=chunk.iloc[:,3+i].map(decode_hex).to_numpy(dtype=np.float32)
        else:
            v=np.zeros(len(chunk),dtype=np.float32)
        v[dlc<=i]=0.0  # exclude a trailing R/T flag from payload
        values.append(v)
    valid=np.isfinite(t)&valid_dlc.to_numpy()
    x=np.column_stack([cid,dlc.astype(np.float32)]+values).astype(np.float32)
    return x[valid],y[valid],t[valid],attack_name


def load_data(data_root:Path,max_rows:int,seed:int=11):
    """Parse the original HCRL files with unbiased, streaming priority sampling.

    IMPORTANT: for max_rows>0 the raw CAN rows are sampled BEFORE expensive
    per-byte hex decoding. Retaining the lowest K independent random priorities
    gives a uniform sample of every recording, preserving its natural attack
    incidence. Training-only balancing takes place later in train_model().

    Full-data mode (max_rows=0) parses every record and may require large RAM.
    """
    files=select_files(data_root)
    xs=[];ys=[];ts=[];sources=[];scenarios=[];audit=[]
    rng=np.random.default_rng(seed)
    for sid,p in enumerate(files):
        attack=detect_attack_name(p)
        chunks=pd.read_csv(p,header=None,names=list(range(12)),dtype=str,
                           chunksize=100000,on_bad_lines='error')
        raw_pool=None;priority_pool=np.empty(0,dtype=np.float64)
        full_chunks=[];total=0;positive=0;malformed=0
        for raw in chunks:
            # A record is timestamp, ID, DLC, then DLC bytes, then R/T.
            # Attack label position varies with DLC. Avoid ffill across payload.
            n=len(raw)
            dlc_num=pd.to_numeric(raw.iloc[:,2],errors='coerce').fillna(-1).to_numpy(dtype=np.int16)
            timestamp_num=pd.to_numeric(raw.iloc[:,0],errors='coerce').to_numpy(dtype=np.float64)
            usable=(dlc_num>=0)&(dlc_num<=8)&np.isfinite(timestamp_num)
            pos=3+np.clip(dlc_num,0,8).astype(np.int64)
            flags=raw.to_numpy(dtype=object)[np.arange(n),pos]
            good=usable&((flags=='R')|(flags=='T'))
            malformed+=int((~good).sum())
            if not good.any():continue
            valid_raw=raw.iloc[np.flatnonzero(good)]
            valid_labels=np.asarray(flags[good]);total+=len(valid_labels)
            positive+=int(np.sum(valid_labels=='T'))
            if max_rows==0:
                # Full-data mode: decode each chunk and concatenate ONCE later.
                x,y,t,_=parse_frame(valid_raw,p)
                full_chunks.append((x,y,t))
            else:
                r=rng.random(len(valid_labels))
                if raw_pool is None:
                    raw_pool=valid_raw.reset_index(drop=True)
                    priority_pool=r
                else:
                    raw_pool=pd.concat([raw_pool,valid_raw],ignore_index=True)
                    priority_pool=np.concatenate((priority_pool,r))
                if len(priority_pool)>max_rows:
                    keep=np.argpartition(priority_pool,max_rows-1)[:max_rows]
                    raw_pool=raw_pool.iloc[keep].reset_index(drop=True)
                    priority_pool=priority_pool[keep]
        if malformed:
            warnings.warn(f'{p.name}: {malformed} invalid timestamp/DLC/label rows excluded')
        if max_rows==0:
            if not full_chunks:continue
            xx=np.vstack([q[0] for q in full_chunks]);yy=np.concatenate([q[1] for q in full_chunks])
            tt=np.concatenate([q[2] for q in full_chunks])
        else:
            if raw_pool is None or raw_pool.empty:continue
            xx,yy,tt,_=parse_frame(raw_pool,p)
        idx=np.argsort(tt,kind='stable')
        xx=xx[idx];yy=yy[idx];tt=tt[idx]
        # Sequential features are based on the retained stream. Full-stream
        # interarrival and ID counts require --max-rows-per-file 0.
        dt=np.r_[0,np.maximum(0,np.diff(tt))].astype(np.float32)
        freq=np.empty(len(xx),dtype=np.float32);counts={}
        for i,k in enumerate(xx[:,0].astype(np.int64)):
            counts[k]=counts.get(k,0)+1;freq[i]=min(counts[k],10000)
        xx=np.column_stack([xx,np.log1p(dt),np.log1p(freq)]).astype(np.float32)
        xs.append(xx);ys.append(yy);ts.append(tt)
        sources.append(np.full(len(yy),sid,dtype=np.int32))
        scenarios.append(np.full(len(yy),attack,dtype='<U24'))
        audit.append({'path':str(p),'sha256':sha256(p),'source_rows_parsed':total,
                      'source_attack_rows':positive,'source_normal_rows':total-positive,
                      'source_attack_fraction':positive/max(total,1),
                      'malformed_source_rows':malformed,'sampled_rows':len(yy),
                      'attack_rows':int(yy.sum()),'normal_rows':int((yy==0).sum()),
                      'scenario':attack,'min_timestamp':float(tt.min()),
                      'max_timestamp':float(tt.max()),
                      'sampling':'uniform priority reservoir on attack recording with original R/T labels'})
        print(f'Loaded {p.name}: {len(yy):,} retained / {total:,} original, '
              f'attacks {int(yy.sum()):,} / {positive:,} original',flush=True)
    if not xs:raise RuntimeError('No valid CAN recordings found')
    return (np.vstack(xs),np.concatenate(ys),np.concatenate(ts),
            np.concatenate(sources),np.concatenate(scenarios),audit)


def temporal_split(t:np.ndarray,source:np.ndarray,train=.6,val=.2,gap=.01,y=None):
    """Prefer chronological per-recording split; repair missing class coverage explicitly.

    If an attack is confined to one time interval, the strict chronological split
    may leave a partition with no positives. In that case, perform a label-wise
    chronological allocation within the affected recording. This is a
    class-stratified temporal allocation, NOT a strict future-time holdout.
    """
    parts=np.full(len(t),-1,dtype=np.int8)
    for sid in np.unique(source):
        idx=np.flatnonzero(source==sid)
        idx=idx[np.argsort(t[idx],kind='stable')]
        n=len(idx);a=int(train*n);b=int((train+val)*n);g=max(1,int(gap*n))
        if a-g<=0 or b-a-2*g<=0 or n-b-g<=0:
            raise ValueError(f'Insufficient temporal data in source {sid}')
        parts[idx[:a-g]]=0
        parts[idx[a+g:b-g]]=1
        parts[idx[b+g:]]=2

    if y is None:
        return parts

    missing=[j for j in range(3) if len(np.unique(y[parts==j]))<2]
    if not missing:
        temporal_split.last_strategy='strict_per_recording_chronological'
        return parts

    # Repair only when strict chronological splitting cannot evaluate both classes.
    # Every recording/class is ordered by timestamp and partitioned 60/20/20.
    # No random shuffling, label fabrication, or training on held-out rows.
    print('WARNING: Strict chronological split lacks both classes in '
          f'partitions {missing}. Using class-stratified chronological allocation. '
          'This is NOT a strict future-time holdout.',flush=True)
    parts[:]=-1
    for sid in np.unique(source):
        for label in np.unique(y[source==sid]):
            idx=np.flatnonzero((source==sid)&(y==label))
            idx=idx[np.argsort(t[idx],kind='stable')]
            n=len(idx)
            if n<12:
                # Too few messages for a meaningful split: exclude from evaluation.
                continue
            a=int(train*n);b=int((train+val)*n)
            g=min(max(1,int(gap*n)),max(0,(b-a-2)//2),max(0,(n-b-2)//2))
            parts[idx[:max(0,a-g)]]=0
            parts[idx[min(n,a+g):max(0,b-g)]]=1
            parts[idx[min(n,b+g):]]=2
    for j in range(3):
        if len(np.unique(y[parts==j]))<2:
            raise RuntimeError('Unable to produce both classes in every split; '
                               'inspect the original label distributions.')
    temporal_split.last_strategy='class_stratified_chronological_fallback_NOT_future_holdout'
    return parts

temporal_split.last_strategy='not_run'


def safe_metric(y,pred,prob):
    """Binary metrics; undefined class-dependent metrics are NA, never forced to 1."""
    from sklearn.metrics import matthews_corrcoef, log_loss, fbeta_score
    from statistics import NormalDist
    y=np.asarray(y,dtype=np.int8);pred=np.asarray(pred,dtype=np.int8)
    prob=np.clip(np.asarray(prob,dtype=float),1e-7,1-1e-7)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    n=len(y);n_pos=int((y==1).sum());n_neg=n-n_pos
    recall=tp/n_pos if n_pos else None
    spec=tn/n_neg if n_neg else None
    precision=tp/(tp+fp) if (tp+fp) else 0.
    npv=tn/(tn+fn) if (tn+fn) else 0.
    z=NormalDist().inv_cdf(.975)
    def wilson(k,n):
        if not n:return (None,None)
        q=k/n;d=1+z*z/n;center=(q+z*z/(2*n))/d
        half=z*np.sqrt(q*(1-q)/n+z*z/(4*n*n))/d
        return (float(max(0,center-half)),float(min(1,center+half)))
    ci=wilson(tp,n_pos)
    ans={'n':n,'normal_support':n_neg,'attack_support':n_pos,
         'prevalence':n_pos/max(1,n),
         'accuracy':float((tp+tn)/max(1,n)),
         'balanced_accuracy':(recall+spec)/2 if recall is not None and spec is not None else None,
         'f1':float(f1_score(y,pred,zero_division=0)),
         'f2':float(fbeta_score(y,pred,beta=2,zero_division=0)),
         'precision':precision,'recall':recall,'recall_ci95_low':ci[0],
         'recall_ci95_high':ci[1], 'specificity':spec,'npv':npv,
         'fpr':fp/n_neg if n_neg else None,'fnr':fn/n_pos if n_pos else None,
         'false_alarms_per_10k':10000*fp/n_neg if n_neg else None,
         'mcc':float(matthews_corrcoef(y,pred)) if n_pos and n_neg else None,
         'ece':ece_score(y,prob),'brier':float(brier_score_loss(y,prob)),
         'log_loss':float(log_loss(y,prob,labels=[0,1])),
         'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
    if n_pos and n_neg:
        ans['roc_auc']=float(roc_auc_score(y,prob))
        ans['average_precision']=float(average_precision_score(y,prob))
    else:
        ans['roc_auc']=None;ans['average_precision']=None
    return ans

def ece_score(y,p,nbins=15):
    p=np.asarray(p);y=np.asarray(y);bins=np.linspace(0,1,nbins+1);e=0.
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(p>=lo)&((p<=hi) if hi==1 else (p<hi))
        if m.any():e+=np.mean(m)*abs(float(p[m].mean()-y[m].mean()))
    return float(e)


class Net(nn.Module):
    def __init__(self,n_features:int,hidden=32):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(n_features,hidden),nn.ReLU(),nn.Linear(hidden,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x):return self.layers(x).reshape(-1)


def train_model(x,y,seed,epochs,batch,lr,device=None,xval=None,yval=None,
                init=None,prox_mu=0.,global_params=None,balance_mode=None):
    """GPU-resident batches, optionally 50/50 training-only real-message resampling.

    Balanced batches never alter a validation/test label or prevalence.
    When one local FL client has one class, fall back to the observed class.
    """
    device=device or ACTIVE_DEVICE
    balance_mode=balance_mode or ACTIVE_BALANCE
    seed_all(seed)
    net=Net(x.shape[1]).to(device)
    if init is not None:net.load_state_dict({k:v.detach().clone() for k,v in init.items()})
    opt=torch.optim.Adam(net.parameters(),lr=lr)
    xt=torch.as_tensor(np.ascontiguousarray(x),dtype=torch.float32,device=device)
    yt=torch.as_tensor(np.ascontiguousarray(y),dtype=torch.float32,device=device)
    if len(y)<2:raise ValueError('Training partition has fewer than two samples')
    posids=torch.where(yt==1)[0];negids=torch.where(yt==0)[0]
    use_balance=balance_mode=='batch' and len(posids)>0 and len(negids)>0
    w=max(1,int((yt==0).sum().item()))/max(1,int((yt==1).sum().item()))
    pos_weight=min(w,50.) if balance_mode=='loss' else 1.
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device))
    vx=(torch.as_tensor(np.ascontiguousarray(xval),dtype=torch.float32,device=device)
        if xval is not None and len(xval) else None)
    vy=(torch.as_tensor(np.ascontiguousarray(yval),dtype=torch.float32,device=device)
        if yval is not None and len(yval) else None)
    gen=torch.Generator(device=device).manual_seed(seed)
    best=None;best_err=float('inf');bad=0
    for epoch in range(epochs):
        net.train()
        if use_balance:
            # Same epoch length as natural training. Draw with replacement from
            # observed messages, 50% of each class, with no synthetic data.
            n=len(yt);n1=n//2;n0=n-n1
            ids=torch.cat([posids[torch.randint(len(posids),(n1,),generator=gen,device=device)],
                           negids[torch.randint(len(negids),(n0,),generator=gen,device=device)]])
            ids=ids[torch.randperm(len(ids),generator=gen,device=device)]
        else:ids=torch.randperm(len(yt),generator=gen,device=device)
        for ii in ids.split(batch):
            opt.zero_grad(set_to_none=True)
            loss=criterion(net(xt[ii]),yt[ii])
            if prox_mu and global_params is not None:
                loss=loss+0.5*prox_mu*sum(((p-global_params[n].to(device))**2).sum()
                                            for n,p in net.named_parameters())
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
            opt.step()
        if vx is not None:
            net.eval()
            with torch.inference_mode():
                # Chunk validation to avoid GPU out-of-memory for large datasets.
                err=0.;num=0
                for xx,yy in zip(vx.split(65536),vy.split(65536)):
                    pp=torch.sigmoid(net(xx));err+=((pp-yy)**2).sum().item();num+=len(yy)
            v=err/max(1,num)
            if v<best_err-1e-7:
                best_err=v;best={k:z.detach().cpu().clone() for k,z in net.state_dict().items()};bad=0
            else:
                bad+=1
                if bad>=4:break
    if best is not None:net.load_state_dict(best)
    return net.cpu()

def predict(net,x,device=None,batch=65536):
    device=device or ACTIVE_DEVICE
    net.to(device).eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch):
            z=net(torch.as_tensor(x[start:start+batch],dtype=torch.float32,device=device))
            out.append(torch.sigmoid(z).cpu().numpy())
    return np.concatenate(out) if out else np.array([],dtype=np.float32)


def guard_fit(x,y,quantile=0.995):
    """Train-only benign robust z-score guard; no use of test labels."""
    benign=x[y==0]
    if len(benign)<10:raise ValueError('Need benign training records for Statistical Guard')
    center=np.median(benign,axis=0);mad=np.median(np.abs(benign-center),axis=0)
    scale=np.maximum(mad*1.4826,1e-3)
    score=guard_score(x,center,scale)
    threshold=float(np.quantile(score[y==0],quantile))
    return center,scale,threshold


def guard_score(x,center,scale):
    return np.max(np.minimum(np.abs((x-center)/scale),1e6),axis=1)



def novelty_fit(xtrain,ytrain,validation_x=None,validation_y=None,seed=11,limit=50000):
    """Train benign-only IsolationForest, with FPR threshold calibrated on benign validation.

    This is an open-set *attempt*, not a guaranteed remedy for unseen attacks.
    No rows from the held-out attack recording are used to fit this detector.
    """
    from sklearn.ensemble import IsolationForest
    rng=np.random.default_rng(seed)
    benign=np.flatnonzero(ytrain==0)
    if len(benign)<100:return None
    if len(benign)>limit:benign=rng.choice(benign,size=limit,replace=False)
    detector=IsolationForest(n_estimators=100,max_samples=min(1024,len(benign)),
                             contamination='auto',random_state=seed,n_jobs=-1)
    detector.fit(xtrain[benign])
    return detector

def novelty_score(detector,x):
    # larger score = more anomalous
    return -detector.score_samples(x)

def novelty_threshold(yval,anomaly_val,desired_fpr=.001):
    benign=anomaly_val[yval==0]
    if len(benign)<20:return None
    return float(np.quantile(benign,1-desired_fpr))


def decision(prob,guard,threshold=0.5,guard_on=True):
    return ((prob>=threshold)|(guard if guard_on else False)).astype(np.int8)


def tune_threshold(y,p,fp_cost=1.,fn_cost=5.):
    candidates=np.unique(np.r_[np.linspace(.05,.95,91),np.quantile(p,np.linspace(0,1,41))])
    losses=[]
    for t in candidates:
        z=p>=t;loss=fp_cost*np.sum(z&(y==0))+fn_cost*np.sum((~z)&(y==1))
        losses.append(loss)
    return float(candidates[np.argmin(losses)])


def temperature_fit(y,prob):
    """Validation-only temperature scaling of logits; returns positive T."""
    from scipy.optimize import minimize_scalar
    if len(np.unique(y))<2:return 1.0
    p=np.clip(np.asarray(prob),1e-6,1-1e-6)
    logits=np.log(p/(1-p))
    def obj(log_temp):
        pp=1/(1+np.exp(-np.clip(logits/np.exp(log_temp),-50,50)))
        return -(y*np.log(np.clip(pp,1e-7,1))+(1-y)*np.log(np.clip(1-pp,1e-7,1))).mean()
    return float(np.exp(minimize_scalar(obj,bounds=(-3,3),method='bounded').x))


def temperature_apply(p,temp):
    p=np.clip(np.asarray(p),1e-6,1-1e-6)
    logits=np.log(p/(1-p))/temp
    return 1/(1+np.exp(-np.clip(logits,-50,50)))


def membership_score(model,xtr,ytr,xte,yte,seed,max_each=3000):
    """Balanced, label-matched confidence-only MIA diagnostic; not a DP proof."""
    rng=np.random.default_rng(seed);members=[];outs=[]
    for cl in (0,1):
        a=np.flatnonzero(ytr==cl);b=np.flatnonzero(yte==cl)
        n=min(max_each,len(a),len(b))
        if n<10:continue
        members.extend(rng.choice(a,n,replace=False).tolist())
        outs.extend(rng.choice(b,n,replace=False).tolist())
    if len(members)<20:return {'membership_auc':None,'membership_advantage':None}
    pm=predict(model,xtr[np.asarray(members)]);po=predict(model,xte[np.asarray(outs)])
    confidence=lambda pp,yy:np.where(yy==1,pp,1-pp)
    sm=confidence(pm,ytr[np.asarray(members)]);so=confidence(po,yte[np.asarray(outs)])
    auc=float(roc_auc_score(np.r_[np.ones(len(sm)),np.zeros(len(so))],np.r_[sm,so]))
    return {'membership_auc':auc,'membership_advantage':float(abs(2*auc-1)),
            'membership_samples':2*len(members)}

def plot_curves(y,p,path:Path,name:str):
    if len(np.unique(y))<2:return
    fpr,tpr,_=roc_curve(y,p);prec,rec,_=precision_recall_curve(y,p)
    plt.figure(figsize=(6,4));plt.plot(fpr,tpr,label=f'AUC={roc_auc_score(y,p):.3f}')
    plt.plot([0,1],[0,1],ls=':',color='gray');plt.xlabel('False positive rate');plt.ylabel('True positive rate');plt.legend();plt.tight_layout();plt.savefig(path/name,dpi=180);plt.close()
    plt.figure(figsize=(6,4));plt.plot(rec,prec);plt.xlabel('Recall');plt.ylabel('Precision');plt.tight_layout();plt.savefig(path/name.replace('roc','pr'),dpi=180);plt.close()


def plot_reliability(y,p,path:Path,name='reliability.png'):
    plt.figure(figsize=(6,4));plt.plot([0,1],[0,1],'--',color='gray')
    for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        m=(p>=a)&(p<(b if b<1 else 1.001))
        if m.any():plt.scatter(float(np.mean(p[m])),float(np.mean(y[m])),s=max(15,160*m.mean()),color='#226b78')
    plt.xlabel('Mean predicted attack probability');plt.ylabel('Observed attack frequency')
    plt.title(f'ECE={ece_score(y,p):.4f}');plt.tight_layout();plt.savefig(path/name,dpi=180);plt.close()


def plot_cm(y,z,path:Path,name='confusion_matrix.png'):
    cm=confusion_matrix(y,z,labels=[0,1])
    fig,ax=plt.subplots(figsize=(4.6,4));ax.imshow(cm,cmap='Blues')
    for (i,j),v in np.ndenumerate(cm):ax.text(j,i,f'{v:,}',ha='center',va='center')
    ax.set_xticks([0,1],['Normal','Attack']);ax.set_yticks([0,1],['Normal','Attack'])
    ax.set_xlabel('Predicted');ax.set_ylabel('True');fig.tight_layout();fig.savefig(path/name,dpi=180);plt.close(fig)


def noniid_clients(y,k,alpha,seed,min_size=32):
    """Dirichlet allocation by labels, never transfers test records to clients."""
    rng=np.random.default_rng(seed);indices=[[] for _ in range(k)]
    for label in np.unique(y):
        subset=rng.permutation(np.flatnonzero(y==label))
        frac=rng.dirichlet(np.full(k,alpha));cuts=np.cumsum((frac*len(subset)).astype(int))[:-1]
        for j,a in enumerate(np.split(subset,cuts)):indices[j].extend(a.tolist())
    sizes=[len(v) for v in indices]
    if min(sizes)<min_size:
        return None
    return [rng.permutation(v).astype(np.int64) for v in indices]


def fed_training(x,y,clients,algo,seed,rounds,epochs,batch,lr,evalx=None,evaly=None):
    seed_all(seed);model=Net(x.shape[1]);state={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
    trace=[];t0=time.perf_counter();bytes_per_round=0
    for r in range(rounds):
        local_states=[];weights=[];deltas=[];local_updates=[]
        for j,ids in enumerate(clients):
            glob={name:p.detach().clone() for name,p in model.named_parameters()}
            local=train_model(x[ids],y[ids],seed+r*101+j,epochs,batch,lr,init=state,
                              prox_mu=0.01 if algo=='fedprox' else 0.,global_params=glob)
            s={n:p.detach().cpu().clone() for n,p in local.state_dict().items()}
            local_states.append(s);weights.append(len(ids));local_updates.append(len(ids)//batch+1)
        if algo=='equal':weights=np.ones(len(clients))
        elif algo=='normalized_update':
            # Approximate normalized local update to mitigate client update-count differences.
            # Separate from canonical FedNova: report as normalized-update baseline.
            weights=np.asarray(weights)/np.asarray(local_updates)
        weights=np.asarray(weights,dtype=float);weights/=weights.sum()
        state={name:sum(s[name]*float(w) for s,w in zip(local_states,weights)) for name in state}
        model.load_state_dict(state)
        bytes_per_round+=2*sum(v.numel()*v.element_size() for v in state.values())*len(clients)
        if evalx is not None and len(evalx):
            pp=predict(model,evalx);trace.append({'round':r+1,'f1':float(f1_score(evaly,pp>=.5,zero_division=0)),
                  'accuracy':float(accuracy_score(evaly,pp>=.5))})
    return model,trace,time.perf_counter()-t0,bytes_per_round


def shadow_membership_attack(xtr,ytr,xhold,yhold,trained_model,seed,epochs,batch,lr):
    """Attack AUC using independent, disjoint shadow train/holdout data, not in-sample threshold fitting."""
    rng=np.random.default_rng(seed)
    # Use disjoint random halves of training data for shadow fitting and attack calibration.
    ix=rng.permutation(len(xtr));mid=len(ix)//2
    if mid<50:return None
    shadow=train_model(xtr[ix[:mid]],ytr[ix[:mid]],seed+101,max(2,epochs//2),batch,lr)
    shadow_member=predict(shadow,xtr[ix[:mid]])
    shadow_out=predict(shadow,xtr[ix[mid:]])
    # Score confidence in the known true label. Threshold calibrated on shadow membership status.
    confidence=lambda p,y:np.where(y==1,p,1-p)
    m=confidence(shadow_member,ytr[ix[:mid]]);n=confidence(shadow_out,ytr[ix[mid:]])
    threshold=float(np.median(np.r_[m,n]))
    score_member=confidence(predict(trained_model,xtr),ytr)
    score_non=confidence(predict(trained_model,xhold),yhold)
    lab=np.r_[np.ones(len(score_member)),np.zeros(len(score_non))]
    score=np.r_[score_member,score_non]
    return {'attack_auc':float(roc_auc_score(lab,score)),
            'attack_accuracy_fixed_shadow_threshold':float(np.mean((score>=threshold)==lab)),
            'note':'Confidence-based membership baseline, not formal differential-privacy proof'}


def parameter_perturb(model,sigma,seed):
    """Post-training parameter noise for empirical robustness only. NO formal DP claim."""
    result=Net(next(iter(model.parameters())).shape[1]);result.load_state_dict(model.state_dict())
    rng=np.random.default_rng(seed)
    with torch.no_grad():
        for param in result.parameters():
            param.add_(torch.from_numpy(rng.normal(0,sigma,param.shape).astype('float32')))
    return result


def latency_benchmark(model,x,path:Path,seed=11,runs=250):
    previous_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    model.cpu().eval();idx=np.random.default_rng(seed).integers(0,len(x),size=runs)
    sample=torch.as_tensor(x[idx],dtype=torch.float32)
    with torch.inference_mode():
        for i in range(min(30,runs)):model(sample[i:i+1])
        ts=[];tracemalloc.start()
        for i in range(runs):
            t0=time.perf_counter_ns();model(sample[i:i+1]);ts.append((time.perf_counter_ns()-t0)/1e6)
        _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    params=sum(p.numel() for p in model.parameters());size_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
    gpu_values=None;peak_gpu=None
    if torch.cuda.is_available():
        gd=torch.device('cuda');model.to(gd)
        gs=sample.to(gd)
        with torch.inference_mode():
            for i in range(min(30,runs)):model(gs[i:i+1])
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            timings=[]
            for i in range(runs):
                start=time.perf_counter_ns()
                model(gs[i:i+1]);torch.cuda.synchronize()
                timings.append((time.perf_counter_ns()-start)/1e6)
            peak_gpu=int(torch.cuda.max_memory_allocated())
        gpu_values={'gpu_batch1_p50_ms':float(np.median(timings)),
                    'gpu_batch1_p95_ms':float(np.percentile(timings,95)),
                    'gpu_peak_allocated_bytes':peak_gpu,
                    'gpu_device_name':torch.cuda.get_device_name(0)}
        model.cpu()
    out={'batch':1,'runs':runs,'latency_median_ms':float(np.median(ts)),
         'latency_p95_ms':float(np.percentile(ts,95)),'latency_p99_ms':float(np.percentile(ts,99)),
         'python_peak_allocated_bytes':int(peak),'model_parameter_bytes':int(size_bytes),
         'parameters':int(params),'estimated_linear_mac_per_inference':int(sum(m.in_features*m.out_features for m in model.modules() if isinstance(m,nn.Linear))),
         'device':platform.platform(),'processor':platform.processor(),'torch':torch.__version__}
    if gpu_values:out.update(gpu_values)
    pd.DataFrame({'latency_ms':ts}).to_csv(path/'tables'/'latency_samples.csv',index=False)
    torch.set_num_threads(previous_threads)
    return out


def summarize_seeds(results,root):
    from scipy.stats import t as student_t
    df=pd.DataFrame(results)
    if df.empty:return
    df.to_csv(root/'tables'/'all_experiments.csv',index=False)
    for family,part in df.groupby('family'):
        part.to_csv(root/'tables'/f'{family.lower()}.csv',index=False)
    metrics=[c for c in ['accuracy','balanced_accuracy','f1','f2','precision',
            'recall','specificity','mcc','roc_auc','average_precision','brier',
            'ece','log_loss','fpr','fnr','false_alarms_per_10k','mean_cost_per_frame',
            'membership_auc','duration_sec','communication_bytes'] if c in df]
    keys=[c for c in ['family','setting','k','alpha','algorithm','sigma'] if c in df]
    rows=[]
    for vals,g in df.groupby(keys,dropna=False):
        if not isinstance(vals,tuple):vals=(vals,)
        entry=dict(zip(keys,vals));entry['n_runs']=len(g)
        for metric in metrics:
            v=pd.to_numeric(g[metric],errors='coerce').dropna().to_numpy()
            n=len(v)
            entry[f'{metric}_n']=n
            entry[f'{metric}_mean']=float(np.mean(v)) if n else None
            entry[f'{metric}_std']=float(np.std(v,ddof=1)) if n>1 else None
            half=(student_t.ppf(.975,n-1)*np.std(v,ddof=1)/np.sqrt(n)) if n>1 else None
            entry[f'{metric}_ci95_low']=float(np.mean(v)-half) if half is not None else None
            entry[f'{metric}_ci95_high']=float(np.mean(v)+half) if half is not None else None
        rows.append(entry)
    pd.DataFrame(rows).to_csv(root/'tables'/'summary_mean_std_ci95.csv',index=False)

def generate_report(rows,audit,root,cfg,blocked):
    f=root/'results_report.md'
    families=Counter(r['family'] for r in rows)
    lines=['# HERCULES-V2X reviewer-revision results','',
           '> Automatically generated only from actually executed runs. No synthetic or imputed results are inserted.','',
           '## Provenance','',f'- Dataset location: `{cfg.data_root}`',
           f'- Total recorded experiment rows: **{len(rows)}**',
           f'- Executed experiment families: {dict(families)}',
           f'- Random seeds: {list(cfg.seeds)}',
           f'- Split strategy: {temporal_split.last_strategy}.',
            '- If class-stratified fallback was used, results are NOT strict future-time generalization estimates.',
           '- Validation is used to tune operating thresholds; test labels are not used to tune models or thresholds.',
           '- This is a new independent baseline implementation; model parity with the published original is not established.','',
           '## Dataset audit','', '| Source | Scenario | Sampled rows | Attack | Normal |','|---|---|---:|---:|---:|']
    for a in audit:lines.append(f"| {Path(a['path']).name} | {a['scenario']} | {a['sampled_rows']:,} | {a['attack_rows']:,} | {a['normal_rows']:,} |")
    lines+=['','## Result tables and figures','',
            '- Machine-readable table: `tables/all_experiments.csv`',
            '- Aggregated table: `tables/summary_mean_std.csv`',
            '- Figures: `figures/`, fitted models: `models/`, run settings: `reproducibility/`.','',
            '## Interpretation constraints','',
            '- Parameter Gaussian perturbation is empirical noise, not a formal (epsilon, delta)-differential privacy guarantee.',
            '- CPU results refer only to the recorded host hardware and do not establish ECU deployment feasibility.',
            '- Client allocations are simulated; they do not establish vehicle-to-vehicle generalization.',
            '- Normalized-update is exploratory, NOT canonical FedNova.',
             '- This is a simulation on one vehicle/dataset. No ECU hardware, formal DP, or independent vehicle evaluation is claimed.',
            '- Held-out attack evaluation is undefined if no positive examples remain in the held-out source, and is reported by attack recall and false-positive rate.',
            '- Single-dataset evidence cannot substitute for independent deployment-dataset validation.',
             '- Source labels are original R/T values in four attack recordings; separate normal_run_data.txt is excluded.',
             '- Uniform sampling preserves natural within-recording prevalence; balanced training never changes test labels.',
             '- The sampling-dependent interarrival and cumulative-ID-count features are computed from retained frames; full-stream timing uses --max-rows-per-file 0.','']
    # Summaries are drawn ONLY from records actually produced by the runner.
    data=pd.DataFrame(rows)
    lines+=['','## Measured experiment summary','',
            '| Family | Runs | Mean test F1 | Mean recall | Mean precision | Mean PR-AUC |',
            '|---|---:|---:|---:|---:|---:|']
    for fam,g in data.groupby('family'):
        def fmt(field):
            a=pd.to_numeric(g[field],errors='coerce').dropna() if field in g else []
            return f'{float(np.mean(a)):.4f}' if len(a) else 'N/A'
        lines.append(f'| {fam} | {len(g)} | {fmt("f1")} | {fmt("recall")} | '
                     f'{fmt("precision")} | {fmt("average_precision")} |')
    lines+=['','**Warning:** Family-level averages combine different configurations and are descriptive, '
            'not matched causal comparisons. Use the per-setting CSV tables for reviewer claims.','']
    central=data[data.family=='Centralized']
    if not central.empty:
        base=float(central.f1.mean());lines.append(f'- Centralized average test F1: {base:.4f}.')
    guard=data[data.family=='Guard']
    if not guard.empty and 'f1_delta_vs_classifier' in guard:
        delta=pd.to_numeric(guard.f1_delta_vs_classifier,errors='coerce').dropna()
        if len(delta):lines.append(f'- Guard F1 delta relative to its matched classifier: '
                                   f'min {delta.min():+.4f}, max {delta.max():+.4f}.')
    held=data[data.family=='LeaveOneAttackOut']
    if not held.empty:
        lines.append(f'- Held-out attack F1 range: {held.f1.min():.4f} to '
                     f'{held.f1.max():.4f}. Entire held-out recordings were excluded from preprocessing and training.')
    lines+=['','Independent-vehicle validation, real ECU tests, and formal differential privacy remain outside this single-dataset script.',
            'Novelty models are trained on other recordings only; their performance is measured rather than assumed.',
            'Interpret missing or very small attack support cautiously, and consult '
            '`tables/strict_temporal_audit.csv` before making temporal-generalization claims.','']
    if blocked:lines+=['## Skipped or blocked configurations','']+[f'- {v}' for v in blocked]+['']
    f.write_text('\n'.join(lines),encoding='utf8')


def run(cfg:Config):
    global ACTIVE_DEVICE
    global ACTIVE_BALANCE
    ACTIVE_BALANCE=cfg.balance_mode
    ACTIVE_DEVICE=('cuda' if torch.cuda.is_available() else 'cpu') if cfg.device=='auto' else cfg.device
    if ACTIVE_DEVICE.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but this PyTorch installation cannot access CUDA. Install CUDA-enabled PyTorch.')
    print(f'Compute device: {ACTIVE_DEVICE}; training-only balancing: {ACTIVE_BALANCE}',flush=True)
    if ACTIVE_DEVICE.startswith('cuda'):
        print(f'GPU: {torch.cuda.get_device_name(0)}',flush=True)
    else:
        print('GPU unavailable; training will use CPU.',flush=True)
    root=Path(cfg.out);mkdirs(root)
    x,y,t,source,scenario,audit=load_data(Path(cfg.data_root),cfg.max_rows_per_file,cfg.seeds[0]);
    if len(np.unique(y))!=2:raise RuntimeError('At least one T attack and one R benign sample required.')
    parts=temporal_split(t,source,y=y)
    tr=parts==0;va=parts==1;te=parts==2
    if any(np.unique(y[mask]).size!=2 for mask in [tr,va,te]):
        raise RuntimeError('Every train/validation/test partition needs both labels; check source coverage.')
    scaler=StandardScaler().fit(x[tr]);X=scaler.transform(x).astype('float32');joblib.dump(scaler,root/'models'/'train_only_scaler.joblib')
    for a in audit:a['split_strategy']=temporal_split.last_strategy
    json_save(root/'reproducibility'/'dataset_audit.json',audit)
    json_save(root/'reproducibility'/'configuration.json',asdict(cfg))
    json_save(root/'reproducibility'/'environment.json',{
        'python':sys.version,'platform':platform.platform(), 'torch':torch.__version__,
        'numpy':np.__version__, 'pandas':pd.__version__,
        'number_of_rows_used':len(y),'feature_names':FEATURES,
        'sampling': 'full dataset' if cfg.max_rows_per_file==0 else 'uniform random per-file sample, not full dataset',
        'split_strategy':temporal_split.last_strategy,
        'training_device':ACTIVE_DEVICE,'balance_mode':ACTIVE_BALANCE,'version':VERSION,
        'cuda_available':torch.cuda.is_available(),
        'separate_normal_dataset_included':False})
    fig,ax=plt.subplots(figsize=(7,4))
    labels=['train','validation','test']
    counts=[(int(((parts==i)&(y==0)).sum()),int(((parts==i)&(y==1)).sum())) for i in [0,1,2]]
    indices=np.arange(3);ax.bar(indices,[z[0] for z in counts],label='normal')
    ax.bar(indices,[z[1] for z in counts],bottom=[z[0] for z in counts],label='attack')
    ax.set_xticks(indices,labels);ax.set_ylabel('Message count');ax.legend();fig.tight_layout()
    fig.savefig(root/'figures'/'temporal_split_class_counts.png',dpi=180);plt.close(fig)
    np.savez_compressed(root/'reproducibility'/'split_indices.npz',train=np.flatnonzero(tr),val=np.flatnonzero(va),test=np.flatnonzero(te))
    results=[];blocked=[];max_cfg=cfg.max_runtime_configs
    pd.DataFrame(audit).to_csv(root/'tables'/'dataset_audit.csv',index=False)
    pd.DataFrame([{'split':['train','validation','test'][j], 'normal':int(((parts==j)&(y==0)).sum()), 'attack':int(((parts==j)&(y==1)).sum())} for j in range(3)]).to_csv(root/'tables'/'class_distribution.csv',index=False)
    def add(family,setting,seed,metrics,**kw):
        row={'family':family,'setting':setting,'seed':seed,**kw,**metrics};results.append(row)
        if len(results)%10==0:
            summarize_seeds(results,root);print(f'completed {len(results)} experiment records',flush=True)
    trained={}
    for seed in cfg.seeds:
        seed_all(seed)
        net=train_model(X[tr],y[tr],seed,cfg.epochs,cfg.batch,cfg.lr,xval=X[va],yval=y[va])
        trained[seed]=net
        pval=predict(net,X[va]);ptest=predict(net,X[te]);threshold=tune_threshold(y[va],pval)
        torch.save({'state':net.state_dict(),'features':FEATURES,'threshold':threshold,'seed':seed},root/'models'/f'central_seed{seed}.pt')
        add('Centralized','classifier_only',seed,safe_metric(y[te],ptest>=threshold,ptest),threshold=threshold,balance_mode=ACTIVE_BALANCE)
        if seed==cfg.seeds[0]:
            np.savez_compressed(root/'metrics'/'test_predictions_seed11.npz',y=y[te],p=ptest,scenario=scenario[te])
            for variant in ['none','loss','batch']:
                if variant==ACTIVE_BALANCE:continue
                alt=train_model(X[tr],y[tr],seed,cfg.epochs,cfg.batch,cfg.lr,
                                xval=X[va],yval=y[va],balance_mode=variant)
                pv_alt=predict(alt,X[va]);pt_alt=predict(alt,X[te])
                tau_alt=tune_threshold(y[va],pv_alt)
                add('TrainingBalance',variant,seed,safe_metric(y[te],pt_alt>=tau_alt,pt_alt),
                    threshold=tau_alt,note='Only training distribution changed; test distribution identical')
                torch.save(alt.state_dict(),root/'models'/f'balance_{variant}_seed{seed}.pt')
        if seed==cfg.seeds[0]:
            plot_curves(y[te],ptest,root/'figures','central_roc.png');plot_reliability(y[te],ptest,root/'figures');plot_cm(y[te],ptest>=threshold,root/'figures')
            deployment=latency_benchmark(net,X[te],root,seed)
            # Guard host runtime per message is measured separately using scalar inference.
            gc,gs,qt=guard_fit(X[tr],y[tr])
            sample=X[te][:min(1000,int(te.sum()))]
            timer=[]
            for row in sample:
                clock=time.perf_counter_ns()
                _=bool(guard_score(row[None,:],gc,gs)[0]>qt)
                timer.append((time.perf_counter_ns()-clock)/1e6)
            deployment['guard_latency_p50_ms']=float(np.median(timer))
            deployment['guard_latency_p95_ms']=float(np.percentile(timer,95))
            json_save(root/'metrics'/'deployment_cpu.json',deployment)
            np.savez_compressed(root/'metrics'/'central_test_predictions.npz',y=y[te],p=ptest,source=source[te],scenario=scenario[te])
        # Guard thresholds chosen using training benign distribution; evaluate merged decision end-to-end.
        center,scale,_=guard_fit(X[tr],y[tr]);g_val=guard_score(X[va],center,scale);g_test=guard_score(X[te],center,scale)
        for q in [0.95,0.99,0.995,0.999,0.9999]:
            cutoff=float(np.quantile(guard_score(X[tr][y[tr]==0],center,scale),q))
            gv=g_val>cutoff;gt=g_test>cutoff
            # classifier decision threshold selected from validation end-to-end cost.
            candidate=np.linspace(.1,.9,41)
            costs=[np.sum((decision(pval,gv,t0)==1)&(y[va]==0))+5*np.sum((decision(pval,gv,t0)==0)&(y[va]==1)) for t0 in candidate]
            tau=float(candidate[int(np.argmin(costs))]);z=decision(ptest,gt,tau)
            mm=safe_metric(y[te],z,ptest)
            mm['f1_delta_vs_classifier']=mm['f1']-safe_metric(y[te],ptest>=threshold,ptest)['f1']
            add('Guard',f'q={q}',seed,mm,threshold=tau,guard_quantile=q,guard_alert_rate=float(gt.mean()),note='Probability metrics refer to neural classifier; joint hard decisions are not calibrated probabilities')
        # Guard-only ablation, using its hard flag as a score proxy.
        cutoff=float(np.quantile(guard_score(X[tr][y[tr]==0],center,scale),0.999))
        guardonly=(g_test>cutoff).astype(np.int8)
        add('ArchitectureAblation','guard_only',seed,safe_metric(y[te],guardonly,guardonly.astype(float)),
            note='No neural classifier; hard guard outputs, NOT calibrated probability')
        temp=temperature_fit(y[va],pval)
        calibrated_val=temperature_apply(pval,temp)
        calibrated_test=temperature_apply(ptest,temp)
        for policy,fp,fn in [('balanced',1,1),('safety',1,10),('moderate',1,5),('low_false_alarms',10,1)]:
            for cal,pv,pt in [('raw',pval,ptest),('temperature',calibrated_val,calibrated_test)]:
                tau=tune_threshold(y[va],pv,fp,fn);pred=pt>=tau
                m=safe_metric(y[te],pred,pt)
                m['mean_cost_per_frame']=float((fp*m['fp']+fn*m['fn'])/len(y[te]))
                add('ORC',policy+'_'+cal,seed,m,threshold=tau,fp_cost=fp,fn_cost=fn,temperature=temp)
        # Explicit validation-derived false-alert budgets. These quantify the
        # deployment trade-off even when risk-cost policies select the same tau.
        for target in (0.0001,0.001,0.01):
            for cal,pv,pt in [('raw',pval,ptest),('temperature',calibrated_val,calibrated_test)]:
                benign_validation=pv[y[va]==0]
                if not len(benign_validation):continue
                tau_budget=float(np.quantile(benign_validation,1-target))
                decisions=(pt>tau_budget).astype(np.int8)
                metric=safe_metric(y[te],decisions,pt)
                add('ORCBudget',f'val_fpr_{target:g}_{cal}',seed,metric,
                    threshold=tau_budget,target_validation_fpr=target,
                    actual_validation_fpr=float(np.mean(benign_validation>tau_budget)),
                    note='Alert budget tuned using BENIGN validation only; report observed test FPR')
        # Four-layer operational decision: guard immediate alerts OR
        # calibrated neural risk threshold; FL is compared separately as a
        # training regime and is not wrongly presented as an inference layer.
        gf=float(np.quantile(guard_score(X[tr][y[tr]==0],center,scale),0.999))
        merged_val=g_val>gf;merged_test=g_test>gf
        candidates=np.linspace(.02,.98,97)
        costs=[np.sum(((calibrated_val>=v)|merged_val)&(y[va]==0))+
               5*np.sum(((calibrated_val<v)&(~merged_val))&(y[va]==1)) for v in candidates]
        tau_comb=float(candidates[int(np.argmin(costs))])
        predictions=((calibrated_test>=tau_comb)|merged_test).astype(np.int8)
        add('ArchitectureAblation','guard_plus_calibrated_neural',seed,
            safe_metric(y[te],predictions,calibrated_test),threshold=tau_comb,
            note='Joint hard decisions; probability metrics refer only to calibrated neural scores')
        if seed==cfg.seeds[0]:
            json_save(root/'models'/'full_pipeline_settings.json',{'threshold':tau_comb,
                'guard_threshold':gf,'temperature':temp,'features':FEATURES,'seed':seed})
        for name in sorted(set(scenario[te])):
            m=(scenario[te]==name)
            if m.sum()==0:continue
            add('PerAttack',name,seed,safe_metric(y[te][m],ptest[m]>=threshold,ptest[m]))
        # Reference held-out scenario, evaluated with training sources containing other attack types.
        # Note this is generalization across recording scenarios, not across vehicles.
        if cfg.mode=='full':
            for held in sorted(set(scenario[y==1])):
                keep=tr&(scenario!=held)
                if len(np.unique(y[keep]))<2:blocked.append(f'LOAO {held}: no positive/negative train examples');continue
                # The entire held-out recording is unseen. Restricting to its
                # final chronological segment can leave zero attack messages.
                hold=(scenario==held)
                if hold.sum()<10:continue
                # Essential: fit the scaler ONLY on the kept training sources.
                # Previous versions inadvertently included held-out attack data
                # in preprocessing, making leave-one-attack-out scores invalid.
                local_scaler=StandardScaler().fit(x[keep])
                xtrain_local=local_scaler.transform(x[keep]).astype(np.float32)
                valid=va&(scenario!=held)
                xvalid_local=local_scaler.transform(x[valid]).astype(np.float32)
                xtest_local=local_scaler.transform(x[hold]).astype(np.float32)
                local=train_model(xtrain_local,y[keep],seed,cfg.epochs,cfg.batch,cfg.lr,
                                  xval=xvalid_local,yval=y[valid])
                pv=predict(local,xvalid_local);yv=y[valid]
                if len(np.unique(yv))<2:
                    blocked.append(f'LOAO {held}: validation contains a single label')
                    continue
                tau=tune_threshold(yv,pv)
                pp=predict(local,xtest_local)
                met=safe_metric(y[hold],pp>=tau,pp)
                add('LeaveOneAttackOut',held,seed,met,threshold=tau,algorithm='supervised',
                    note='Entire held-out recording excluded from fitting; natural R/T prevalence')
                # Open-set ablation: train only on R rows from the OTHER recordings.
                # Select thresholds on validation R records from those same recordings.
                if cfg.novelty:
                    det=novelty_fit(xtrain_local,y[keep],seed=seed)
                    if det is not None:
                        as_val=novelty_score(det,xvalid_local)
                        as_test=novelty_score(det,xtest_local)
                        novelty_cut=novelty_threshold(yv,as_val,.001)
                        if novelty_cut is not None:
                            open_pred=(as_test>novelty_cut).astype(np.int8)
                            # Normalized rank is a novelty score, NOT calibrated attack probability.
                            # ROC / PR AUC can be computed on it, but Brier/ECE from this number
                            # do not indicate probability calibration.
                            benign_ref=np.sort(as_val[yv==0])
                            anomaly_rank=np.searchsorted(benign_ref,as_test,side='right')/len(benign_ref)
                            novelty_met=safe_metric(y[hold],open_pred,anomaly_rank)
                            for key in ['brier','ece','log_loss']:
                                novelty_met[key]=None
                            add('LeaveOneAttackOutNovelty',held,seed,novelty_met,
                                algorithm='isolation_forest_benign_only',threshold=novelty_cut,
                                note='Open-set anomaly detection; scores are benign-reference ranks, not probabilities')
                            # Combined two-path alert: supervised probability OR anomaly flag.
                            combined=((pp>=tau)|open_pred).astype(np.int8)
                            combined_met=safe_metric(y[hold],combined,pp)
                            for key in ['brier','ece','log_loss']:
                                combined_met[key]=None
                            add('LeaveOneAttackOutEnsemble',held,seed,combined_met,
                                algorithm='supervised_or_benign_novelty',
                                threshold=tau,novelty_threshold=novelty_cut,
                                note='OR-alert; probability calibration undefined for hard union')
                            # Threshold sweep uses only validation R traffic from other
                            # recordings. Every test operating point is reported; no
                            # threshold is selected by held-out labels.
                            for target_fpr in cfg.novelty_fpr_grid:
                                candidate=novelty_threshold(yv,as_val,target_fpr)
                                if candidate is None:continue
                                znov=(as_test>candidate).astype(np.int8)
                                znunion=((pp>=tau)|znov).astype(np.int8)
                                for tag,pred in [('novelty',znov),('union',znunion)]:
                                    met_sweep=safe_metric(y[hold],pred,anomaly_rank)
                                    for key in ('brier','ece','log_loss'):
                                        met_sweep[key]=None
                                    add('OpenSetThresholdSweep',held+'_'+tag,seed,met_sweep,
                                        algorithm=tag,heldout_attack=held,
                                        target_validation_fpr=target_fpr,
                                        actual_validation_fpr=float(np.mean(as_val[yv==0]>candidate)),
                                        threshold=candidate,
                                        note='Threshold set without held-out source; scores are not calibrated probabilities')
                            if seed==cfg.seeds[0]:
                                joblib.dump(det,root/'models'/f'loao_novelty_{held}.joblib')
                                np.savez_compressed(root/'metrics'/f'loao_{held}_predictions.npz',
                                   y=y[hold],neural_probability=pp,novelty_score=as_test,
                                   anomaly_alert=open_pred,combined_alert=combined)
                if seed==cfg.seeds[0]:
                    torch.save(local.state_dict(),root/'models'/f'loao_{held}.pt')
                    joblib.dump(local_scaler,root/'models'/f'loao_scaler_{held}.joblib')
        if cfg.mode=='full':
            mi=shadow_membership_attack(X[tr],y[tr],X[te],y[te],net,seed,cfg.epochs,cfg.batch,cfg.lr)
            for sigma in cfg.privacy_sigmas:
                noisy=parameter_perturb(net,sigma,seed+89);pp=predict(noisy,X[te]);m=safe_metric(y[te],pp>=threshold,pp)
                m.update(membership_score(noisy,X[tr],y[tr],X[te],y[te],seed+int(sigma*100000)))
                if mi is not None and sigma==0:m.update(mi)
                add('PrivacyNoise',f'sigma={sigma}',seed,m,sigma=sigma)
    # Strict chronological audit is a separate training run. NEVER evaluate a
    # stratified-trained model on its overlapping 'strict' holdout.
    if cfg.strict_audit:
        strict=temporal_split(t,source,y=None)
        strict_tr=(strict==0);strict_va=(strict==1);strict_te=(strict==2)
        audit_row={'protocol':'strict_per_source_chronological',
                   'train_pos':int(y[strict_tr].sum()),'val_pos':int(y[strict_va].sum()),
                   'test_pos':int(y[strict_te].sum()),'test_neg':int((y[strict_te]==0).sum())}
        if y[strict_tr].sum()>=10 and (y[strict_tr]==0).sum()>=10:
            # Strictly train-fitted scaler and model, independent of fallback.
            ss=StandardScaler().fit(x[strict_tr]);SX=ss.transform(x).astype(np.float32)
            joblib.dump(ss,root/'models'/'strict_train_scaler.joblib')
            strict_model=train_model(SX[strict_tr],y[strict_tr],cfg.seeds[0],cfg.epochs,cfg.batch,cfg.lr,
                     xval=SX[strict_va],yval=y[strict_va])
            ppv=predict(strict_model,SX[strict_va]);ppt=predict(strict_model,SX[strict_te])
            tau=tune_threshold(y[strict_va],ppv) if len(np.unique(y[strict_va]))==2 else 0.5
            metric=safe_metric(y[strict_te],ppt>=tau,ppt)
            add('StrictTemporal','strict_future_holdout',cfg.seeds[0],metric,threshold=tau,
                note='If test has a single class, AUC and recall may be undefined')
            torch.save(strict_model.state_dict(),root/'models'/'strict_temporal_model.pt')
            np.savez_compressed(root/'metrics'/'strict_temporal_predictions.npz',y=y[strict_te],p=ppt)
            audit_row['note']='Independent strict time split; if test has zero attacks, detection recall and PR AUC cannot be assessed'
        else:
            blocked.append('StrictTemporal: not enough positives in strict training segment')
            audit_row['note']='Insufficient training positives; no strict model trained'
        pd.DataFrame([audit_row]).to_csv(root/'tables'/'strict_temporal_audit.csv',index=False)
        # Restore primary split label for reporting.
        temporal_split.last_strategy=audit[0]['split_strategy']
    # Strict chronological evaluation restricted to the attack-active observation
    # period in each source. This is a CONDITIONAL, truncated-period holdout;
    # it must never be called full-recording future generalization.
    if cfg.active_time_audit:
        active_parts=np.full(len(y),-1,dtype=np.int8)
        active_rows=[]
        for sid in np.unique(source):
            ids=np.flatnonzero(source==sid)
            ids=ids[np.argsort(t[ids],kind='stable')]
            pos=np.flatnonzero(y[ids]==1)
            if len(pos)<20:
                active_rows.append({'source':int(sid),'status':'insufficient attack events'})
                continue
            last=int(pos[-1])+1
            active=ids[:last]
            n=len(active)
            # Strictly earlier training, middle validation, later test within
            # observed attack-active interval. Never shuffle or stratify.
            a=int(.6*n);b=int(.8*n);g=max(1,int(.005*n))
            if a<=g or b-a<=2*g or n-b<=g:continue
            active_parts[active[:a-g]]=0
            active_parts[active[a+g:b-g]]=1
            active_parts[active[b+g:]]=2
            active_rows.append({'source':int(sid),'active_rows':n,
                                'last_attack_timestamp':float(t[active[-1]]),
                                'train_attacks':int(y[active[:a-g]].sum()),
                                'validation_attacks':int(y[active[a+g:b-g]].sum()),
                                'test_attacks':int(y[active[b+g:]].sum()),
                                'status':'conditional attack-active interval'})
        pd.DataFrame(active_rows).to_csv(root/'tables'/'attack_active_temporal_audit.csv',index=False)
        atr=active_parts==0;ava=active_parts==1;ate=active_parts==2
        if atr.sum()>100 and ava.sum()>100 and ate.sum()>100 and len(np.unique(y[atr]))==2:
            ss=StandardScaler().fit(x[atr]);axx=ss.transform(x).astype(np.float32)
            joblib.dump(ss,root/'models'/'attack_active_scaler.joblib')
            for seed in cfg.seeds:
                model=train_model(axx[atr],y[atr],seed,cfg.epochs,cfg.batch,cfg.lr,
                                  xval=axx[ava],yval=y[ava])
                pv=predict(model,axx[ava]);pt=predict(model,axx[ate])
                tau=tune_threshold(y[ava],pv) if len(np.unique(y[ava]))==2 else .5
                add('AttackActiveTemporal','conditional_active_period',seed,
                    safe_metric(y[ate],pt>=tau,pt),threshold=tau,
                    test_attack_count=int(y[ate].sum()),
                    note='Strict chronological within attack-active period only; not full-recording future holdout')
                if seed==cfg.seeds[0]:
                    torch.save(model.state_dict(),root/'models'/'attack_active_temporal_model.pt')
                    np.savez_compressed(root/'metrics'/'attack_active_temporal_predictions.npz',
                                        y=y[ate],p=pt,source=source[ate])
        else:
            blocked.append('AttackActiveTemporal: insufficient class support for conditional chronological audit')
    # Distributed training is the expensive family. Fast mode limits to k=2, alpha=1 and first seed.
    if cfg.mode=='full':
        runs=0
        from itertools import product
        conditions=list(product(cfg.client_sizes,cfg.dirichlet_alphas))
        # Interleave seeds to avoid spending the cap entirely on seed 11.
        schedule=[(seed,k,alpha) for k,alpha in conditions for seed in cfg.seeds]
        for seed,k,alpha in schedule:
            if runs+4>max_cfg:break
            clients=noniid_clients(y[tr],k,alpha,seed,min_size=max(20,cfg.batch//16))
            if clients is None:
                blocked.append(f'FL seed={seed}, k={k}, alpha={alpha}: one or more clients too small');continue
            for alg in ['equal','fedavg','fedprox','normalized_update']:
                if runs>=max_cfg:break
                model,trace,elapsed,bytes_=fed_training(X[tr],y[tr],clients,alg,seed,cfg.rounds,cfg.local_epochs,cfg.batch,cfg.lr,X[va],y[va])
                pv=predict(model,X[va]);tau=tune_threshold(y[va],pv);pp=predict(model,X[te]);
                m=safe_metric(y[te],pp>=tau,pp);m.update({'duration_sec':elapsed,'communication_bytes':bytes_})
                add('Federated',f'{alg}_k{k}_a{alpha}',seed,m,k=k,alpha=alpha,algorithm=alg,threshold=tau)
                json_save(root/'metrics'/f'convergence_{alg}_k{k}_a{alpha}_s{seed}.json',trace)
                if seed==cfg.seeds[0] and trace:
                    fig,ax=plt.subplots(figsize=(6,4))
                    ax.plot([q['round'] for q in trace],[q['f1'] for q in trace],marker='.')
                    ax.set_xlabel('Federated round');ax.set_ylabel('Validation F1')
                    ax.set_ylim(0,1.05);fig.tight_layout()
                    fig.savefig(root/'figures'/f'convergence_{alg}_k{k}_a{alpha}.png',dpi=180)
                    plt.close(fig)
                torch.save(model.state_dict(),root/'models'/f'fl_{alg}_k{k}_a{alpha}_s{seed}.pt')
                runs+=1
            if runs>=max_cfg:break
        if runs>=max_cfg:blocked.append(f'FL configuration cap {max_cfg} reached. Increase --max-runtime-configs for exhaustive grid.')
    summarize_seeds(results,root)
    # Paired FL comparisons use the same seed, K, and Dirichlet alpha.
    fedrows=pd.DataFrame([r for r in results if r['family']=='Federated'])
    if not fedrows.empty:
        comparison=[]
        for (k,alpha),group in fedrows.groupby(['k','alpha']):
            wide=group.pivot_table(index='seed',columns='algorithm',values='f1',aggfunc='first')
            if 'equal' not in wide:continue
            for algorithm in ('fedavg','fedprox','normalized_update'):
                if algorithm not in wide:continue
                pairs=wide[['equal',algorithm]].dropna()
                if pairs.empty:continue
                diff=(pairs[algorithm]-pairs['equal']).to_numpy(dtype=float)
                comparison.append({'k':k,'alpha':alpha,'algorithm':algorithm,
                    'paired_seeds':len(diff),'mean_f1_delta':float(diff.mean()),
                    'std_f1_delta':float(diff.std(ddof=1)) if len(diff)>1 else np.nan,
                    'ci95_low':float(diff.mean()-1.96*diff.std(ddof=1)/np.sqrt(len(diff))) if len(diff)>1 else np.nan,
                    'ci95_high':float(diff.mean()+1.96*diff.std(ddof=1)/np.sqrt(len(diff))) if len(diff)>1 else np.nan,
                    'note':'Normal-approximation descriptive CI; requires enough paired seeds'})
        pd.DataFrame(comparison,columns=['k','alpha','algorithm','paired_seeds','mean_f1_delta',
            'std_f1_delta','ci95_low','ci95_high','note']).to_csv(root/'tables'/'federated_paired_comparisons.csv',index=False)
        expected=[{'seed':seed,'k':k,'alpha':alpha,'algorithm':alg,
                   'completed':bool(((fedrows.seed==seed)&(fedrows.k==k)&
                   (fedrows.alpha==alpha)&(fedrows.algorithm==alg)).any())}
                  for seed in cfg.seeds for k in cfg.client_sizes for alpha in cfg.dirichlet_alphas
                  for alg in ('equal','fedavg','fedprox','normalized_update')]
        pd.DataFrame(expected).to_csv(root/'tables'/'federated_coverage.csv',index=False)
    if results:
        z=pd.DataFrame(results)
        for family in ['Guard','Federated','PrivacyNoise','ORC','Centralized']:
            g=z[z.family==family]
            if g.empty:continue
            m=g.groupby('setting',sort=False)['f1'].agg(['mean','std']).reset_index()
            fig,ax=plt.subplots(figsize=(max(6,len(m)*0.5),4))
            ax.bar(np.arange(len(m)),m['mean'],yerr=m['std'].fillna(0),capsize=2)
            ax.set_xticks(np.arange(len(m)),m['setting'],rotation=65,ha='right');ax.set_ylabel('Test F1');ax.set_ylim(0,1.05);fig.tight_layout()
            fig.savefig(root/'figures'/f'{family.lower()}_f1.png',dpi=180);plt.close(fig)
    # Reviewer-specific evidence graphics; labeled as empirical quantities only.
    d=pd.DataFrame(results)
    for fam,xname in [('Guard','guard_quantile'),('PrivacyNoise','sigma')]:
        sub=d[d.family==fam]
        if not sub.empty and xname in sub:
            stats=sub.groupby(xname)[['f1','recall','precision']].mean(numeric_only=True).reset_index()
            fig,ax=plt.subplots(figsize=(6,4))
            for c in ['f1','recall','precision']:ax.plot(stats[xname],stats[c],marker='o',label=c)
            ax.set_xlabel(xname);ax.set_ylabel('Test metric');ax.set_ylim(0,1.05)
            ax.legend();fig.tight_layout();fig.savefig(root/'figures'/f'{fam.lower()}_tradeoff.png',dpi=180);plt.close(fig)
    sub=d[d.family=='Federated']
    if not sub.empty:
        means=sub.groupby(['algorithm','k'])['f1'].mean().reset_index()
        fig,ax=plt.subplots(figsize=(7,4))
        for name,g in means.groupby('algorithm'):ax.plot(g.k,g.f1,marker='o',label=name)
        ax.set_xlabel('Simulated clients');ax.set_ylabel('Test F1');ax.legend();fig.tight_layout()
        fig.savefig(root/'figures'/'federated_scaling.png',dpi=180);plt.close(fig)
    # Additional publication-resolution charts and per-scenario confusion matrices.
    for family,metric,name,yy in [('PerAttack','f1','per_attack_f1.png','F1'),
                                  ('ORC','mean_cost_per_frame','orc_operating_cost.png','Cost / message')]:
        sub=d[d.family==family]
        if not sub.empty and metric in sub:
            values=sub.groupby('setting')[metric].mean().sort_values(ascending=False)
            fig,ax=plt.subplots(figsize=(max(6,len(values)*.6),4))
            ax.bar(range(len(values)),values.values)
            ax.set_xticks(range(len(values)),values.index,rotation=45,ha='right')
            ax.set_ylabel(yy);fig.tight_layout();fig.savefig(root/'figures'/name,dpi=200);plt.close(fig)
    pvals=d[d.family=='PrivacyNoise']
    if not pvals.empty and 'membership_auc' in pvals:
        vals=pvals.groupby('sigma')['membership_auc'].mean().dropna()
        if len(vals):
            fig,ax=plt.subplots(figsize=(6,4));ax.plot(vals.index,vals.values,marker='o')
            ax.axhline(.5,ls=':',color='grey');ax.set_xlabel('Parameter noise sigma')
            ax.set_ylabel('Matched confidence membership AUC')
            fig.tight_layout();fig.savefig(root/'figures'/'privacy_attack_auc.png',dpi=200);plt.close(fig)
    try:
        lat=pd.read_csv(root/'tables'/'latency_samples.csv')
        fig,ax=plt.subplots(figsize=(6,4));ax.hist(lat.latency_ms,bins=30)
        ax.set_xlabel('Batch-1 CPU latency, milliseconds');ax.set_ylabel('Count')
        fig.tight_layout();fig.savefig(root/'figures'/'cpu_latency_distribution.png',dpi=200);plt.close(fig)
    except FileNotFoundError:pass
    # Test-only confusion matrices by recording; do NOT include train rows.
    predfile=root/'metrics'/'central_test_predictions.npz'
    if predfile.exists():
        with np.load(predfile) as z:
            yy=z['y'];pp=z['p'];sc=z['scenario']
        basethr=float(d[d.family=='Centralized'].iloc[0]['threshold'])
        for name in np.unique(sc):
            mask=sc==name
            plot_cm(yy[mask],(pp[mask]>=basethr).astype(np.int8),root/'figures',
                    'confusion_'+str(name).lower()+'.png')
    # Original sampled prevalence by scenario, not a balanced test distribution.
    fig,ax=plt.subplots(figsize=(7,4))
    labels=[a['scenario'] for a in audit]
    heights=[100*a['attack_rows']/max(1,a['sampled_rows']) for a in audit]
    ax.bar(labels,heights);ax.set_ylabel('Sampled attack prevalence (%)')
    fig.tight_layout();fig.savefig(root/'figures'/'dataset_prevalence.png',dpi=200);plt.close(fig)
    # Compact measured numerical summary for manuscript drafting.
    if not d.empty:
        keys=['family','setting'];metrics=[c for c in ['f1','recall','precision','accuracy',
            'roc_auc','average_precision','mcc','ece','false_alarms_per_10k'] if c in d]
        d.groupby(keys,dropna=False)[metrics].mean().reset_index().to_csv(
            root/'tables'/'results_summary.csv',index=False)
    reviewer_map=[
      ('R2.1','Lightweight deployment','PARTIAL','deployment_cpu.json, latency_samples.csv; host CPU/GPU, not ECU'),
      ('R2.2','Statistical guard value','TESTED','guard.csv, architectureablation.csv; compare matched F1 and FPR'),
      ('R2.3','Operational risk calibration','PARTIAL','orc.csv and orcbudget.csv; cost policies, false-alert budgets, temperature scaling; not a safety field trial'),
      ('R2.4','Non-IID realistic FL','PARTIAL','federated.csv; simulated clients only'),
      ('R2.5','Formal privacy','UNRESOLVED','privacynoise.csv; empirical membership scores, no DP guarantee'),
      ('R1.1','Methodology leakage and split','PARTIAL','strict_temporal_audit.csv; primary may use stratified fallback'),
      ('R1.2','Independent generalization','PARTIAL','leaveoneattackout.csv and leaveoneattackoutnovelty.csv; no independent vehicle dataset'),
      ('R1.3','End-to-end statistical guard','TESTED','guard.csv, architectureablation.csv'),
      ('R1.4','Probability calibration','TESTED','orc.csv; raw vs temperature calibrated on validation'),
      ('R1.5','Repeats and uncertainty','PARTIAL','summary_mean_std_ci95.csv; require >=3 seeds and complete configurations'),
      ('R1.6','Correct manuscript Tables 2-4','UNRESOLVED','Original manuscript metrics require separate reconciliation'),
      ('R1.7','Writing, figures, references','UNRESOLVED','Requires original manuscript editing'),
      ('R1.8','Open-set attack generalization','PARTIAL','leaveoneattackoutnovelty.csv and leaveoneattackoutensemble.csv; validation must establish real gains'),
      ('R1.9','Attack-active chronological test','CONDITIONAL','attack_active_temporal_audit.csv; truncated attack-active period, not full-recording future holdout'),
      ('R2.6','Open-set alert threshold tradeoffs','TESTED','opensetthresholdsweep.csv; validation-only FPR thresholds, test results may remain poor'),
      ('R3','Weighted aggregation under Non-IID','TESTED','federated.csv; compare equal/fedavg/fedprox in matched conditions')]
    pd.DataFrame(reviewer_map,columns=['reviewer_comment','focus','evidence_status','evidence_and_limitations']).to_csv(
        root/'tables'/'reviewer_evidence_matrix.csv',index=False)
    generate_report(results,audit,root,cfg,blocked)
    print(f'Finished. Results: {root.resolve()} | records: {len(results)} | blocked: {len(blocked)}')


def parse_args():
    a=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a.add_argument('--data-root',default=r'E:\other\hercules\dataset')
    a.add_argument('--out',default=r'E:\other\hercules\results')
    a.add_argument('--mode',choices=['quick','full'],default='full')
    a.add_argument('--seeds',default='11,22,33,44,55')
    a.add_argument('--max-rows-per-file',type=int,default=300000,help='0 = all rows; sampling is not a complete-data evaluation')
    a.add_argument('--epochs',type=int,default=12)
    a.add_argument('--rounds',type=int,default=12)
    a.add_argument('--local-epochs',type=int,default=1)
    a.add_argument('--batch',type=int,default=1024)
    a.add_argument('--lr',type=float,default=.001)
    a.add_argument('--device',choices=['auto','cuda','cpu'],default='auto',help='auto uses NVIDIA CUDA when available')
    a.add_argument('--balance-mode',choices=['batch','loss','none'],default='batch',help='TRAINING ONLY: batch resamples real classes 50/50, loss weights minority, none disables')
    a.add_argument('--no-novelty',action='store_true',help='Disable open-set benign-only anomaly detector')
    a.add_argument('--no-strict-audit',action='store_true',help='Skip additional strict chronological train/test audit')
    a.add_argument('--max-runtime-configs',type=int,default=120)
    a.add_argument('--no-active-time-audit',action='store_true',help='Skip conditional chronological evaluation within observed attack-active intervals')
    return a.parse_args()

if __name__=='__main__':
    a=parse_args()
    cfg=Config(data_root=a.data_root,out=a.out,mode=a.mode,seeds=tuple(int(x) for x in a.seeds.split(',')),
               max_rows_per_file=a.max_rows_per_file,epochs=a.epochs,rounds=a.rounds,
               local_epochs=a.local_epochs,batch=a.batch,lr=a.lr,max_runtime_configs=a.max_runtime_configs,device=a.device,balance_mode=a.balance_mode,strict_audit=not a.no_strict_audit,novelty=not a.no_novelty,active_time_audit=not a.no_active_time_audit)
    run(cfg)
