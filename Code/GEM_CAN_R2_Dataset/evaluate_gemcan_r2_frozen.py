#!/usr/bin/env python3
"""GEM-CAN R2 frozen checkpoint external evaluation (HCRL-trained HERCULES-V2X).
No training, refitting, threshold selection, or test-data calibration. Inference features
reset across recording boundaries. Uses saved original HCRL thresholds and scaler.
"""
import io, os, json, zipfile, time, hashlib, sys, argparse
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import numpy as np
import pandas as pd
import joblib, torch
from sklearn import __version__ as sklearn_version

parser=argparse.ArgumentParser(description='Frozen HCRL model evaluation on GEM-CAN R2')
parser.add_argument('--models-zip',default='results(3).zip',help='ZIP containing HCRL trained models')
parser.add_argument('--dataset-zip',default='GEM_CAN_Dataset_R2.zip',help='GEM-CAN R2 input ZIP')
parser.add_argument('--outdir',default='GEM_CAN_R2_external_results')
args=parser.parse_args()
P=Path(__file__).resolve().parent
modelzip=Path(args.models_zip).resolve()
datasetzip=Path(args.dataset_zip).resolve()
if not modelzip.exists() or not datasetzip.exists():
    raise FileNotFoundError('Provide both --models-zip and --dataset-zip')
spec=spec_from_file_location('gem_old',P/'evaluate_gemcan_external.py')
gem=module_from_spec(spec);spec.loader.exec_module(gem)
torch.set_num_threads(2)

def fast_score(y,p,th):
    # All statistics computed directly from the saved, unmodified probability scores.
    y=np.asarray(y,dtype=np.int8);p=np.asarray(p,dtype=np.float64)
    positive=y==1;negative=~positive;pred=p>=th
    tp=int(np.count_nonzero(pred & positive)); fp=int(np.count_nonzero(pred & negative))
    tn=int(np.count_nonzero(~pred & negative));fn=int(np.count_nonzero(~pred & positive))
    n=len(y);pcount=tp+fn;ncount=tn+fp
    div=lambda a,b:float(a/b) if b else float('nan')
    acc=(tp+tn)/n;rec=div(tp,pcount);spec=div(tn,ncount);prec=div(tp,tp+fp)
    # Precision for all-negative prediction is set to zero only when both classes exist.
    if tp+fp==0 and pcount>0:prec=0.0
    f1=div(2*tp,2*tp+fp+fn) if pcount and ncount else float('nan')
    f2=div(5*tp,5*tp+4*fn+fp) if pcount and ncount else float('nan')
    npv=div(tn,tn+fn)
    if tn+fn==0 and ncount>0:npv=0.0
    pp=np.clip(p,1e-15,1-1e-15)
    bins=np.minimum(np.floor(pp*15).astype(int),14)
    counts=np.bincount(bins,minlength=15)
    sumprob=np.bincount(bins,weights=pp,minlength=15)
    sumlabel=np.bincount(bins,weights=y,minlength=15)
    ece=float(np.sum(np.abs(sumprob-sumlabel))/n)
    denom=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    mcc=float((tp*tn-fp*fn)/np.sqrt(float(denom))) if denom else float('nan')
    if pcount and ncount:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc=float(roc_auc_score(y,p));ap=float(average_precision_score(y,p))
    else:auc=float('nan');ap=float('nan')
    return dict(n=n,normal=ncount,attack=pcount,prevalence=pcount/n,threshold=float(th),
                tn=tn,fp=fp,fn=fn,tp=tp,accuracy=acc,
                balanced_accuracy=(rec+spec)/2 if pcount and ncount else float('nan'),
                precision=prec if pcount and ncount else float('nan'),recall=rec,f1=f1,f2=f2,
                specificity=spec,npv=npv,fpr=div(fp,ncount),fnr=div(fn,pcount),
                false_alarms_per_10k=div(10000*fp,ncount),mcc=mcc,ece=ece,
                brier=float(np.mean((p-y)**2)),
                log_loss=float(np.mean(-y*np.log(pp)-(1-y)*np.log1p(-pp))),
                roc_auc=auc,average_precision=ap,
                fdr=div(fp,tp+fp),false_omission_rate=div(fn,tn+fn),
                informedness=(rec+spec-1) if pcount and ncount else float('nan'),
                markedness=(prec+npv-1) if np.isfinite(prec) and np.isfinite(npv) and pcount and ncount else float('nan'),
                error_rate=1-acc,predicted_positive_rate=(tp+fp)/n)
gem.score=fast_score

outdir=Path(args.outdir).resolve()
outdir.mkdir(parents=True,exist_ok=True)
source_dir=P/'gemcan_r2_work/GEM_CAN_Dataset_R2'
with zipfile.ZipFile(datasetzip) as gd:
    expected=['GEM_Normal_Driving_Raw_Labeled.csv','GEM_Attack_Scenario_Raw_Labeled.csv']
    for file_basename in expected:
        matches=[nm for nm in gd.namelist() if nm.endswith('/'+file_basename)]
        if len(matches)!=1:raise FileNotFoundError('Missing or duplicated GEM-CAN CSV '+file_basename)
        gd.extract(matches[0],P/'gemcan_r2_work')

files={
 'normal_driving':source_dir/'GEM_Normal_Driving_Raw_Labeled.csv',
 'attack_session':source_dir/'GEM_Attack_Scenario_Raw_Labeled.csv'
}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

print('Reading GEM-CAN R2 independent records', flush=True)
frames=[];feats=[];ys=[];audit=[]
for name,p in files.items():
 df=pd.read_csv(p,dtype=str)
 cleaned,x,y=gem.features(df)
 # Attack categories are ground truth only; never features.
 cats=(cleaned['Attack_Type'].to_numpy(dtype=str) if 'Attack_Type' in cleaned.columns
       else np.full(len(y),'Normal driving',dtype='<U30'))
 audit.append(dict(file=p.name,sha256=sha(p),n=len(df),normal=int((y==0).sum()),attack=int((y==1).sum()),
                   time_min=float(pd.to_numeric(cleaned.timestamp).min()),
                   time_max=float(pd.to_numeric(cleaned.timestamp).max()),
                   categories=cleaned['Attack_Type'].value_counts().to_dict() if 'Attack_Type' in cleaned.columns else {'Normal driving':len(df)}))
 frames.append(pd.DataFrame({'recording':name,'label':y,'Attack_Type':cats}))
 feats.append(x);ys.append(y)
 print(name,len(y),'benign',int((y==0).sum()),'attack',int((y==1).sum()), flush=True)

meta=pd.concat(frames,ignore_index=True);x=np.concatenate(feats);y=np.concatenate(ys)
neg=(y==0);pos=(y==1)
assert meta.shape[0]==len(x)==len(y)

with zipfile.ZipFile(modelzip) as z:
 scaler=joblib.load(io.BytesIO(z.read('results/models/train_only_scaler.joblib')))
 xs=scaler.transform(x).astype(np.float32)
 print('Feature means, training:',np.round(scaler.mean_,4),flush=True)
 print('Feature stds, training:',np.round(scaler.scale_,4),flush=True)
 fnames=['can_id','dlc']+[f'byte_{i}' for i in range(8)]+['log1p_dt','log1p_cumulative_id_frequency']
 ood=pd.DataFrame({'feature':fnames,'training_mean':scaler.mean_,'training_std':scaler.scale_,
                   'external_min':x.min(axis=0),'external_max':x.max(axis=0),
                   'frac_abs_z_gt5':(np.abs(xs)>5).mean(axis=0),
                   'frac_abs_z_gt10':(np.abs(xs)>10).mean(axis=0)})
 ood.to_csv(outdir/'external_feature_shift.csv',index=False)
 stat_cols=['accuracy','balanced_accuracy','precision','recall','f1','f2','specificity','npv','fpr','fnr',
            'false_alarms_per_10k','mcc','ece','brier','log_loss','roc_auc','average_precision']
 reference=pd.read_csv(io.BytesIO(z.read('results/tables/federated.csv')))
 original=pd.read_csv(io.BytesIO(z.read('results/tables/centralized.csv')))
 meta['index']=np.arange(len(meta))
 group_index={
  'Pooled GEM-CAN R2':np.arange(len(y)),
  'Attack-session only':np.flatnonzero(meta.recording.to_numpy()=='attack_session'),
  'Normal-driving only':np.flatnonzero(meta.recording.to_numpy()=='normal_driving'),
  'Attack-session normal':np.flatnonzero((meta.recording.to_numpy()=='attack_session') & (y==0)),
  'DoS attack only':np.flatnonzero(meta.Attack_Type.to_numpy()=='DoS'),
  'Brake tampering only':np.flatnonzero(meta.Attack_Type.to_numpy()=='Brake Tampering'),
  'Steering tampering only':np.flatnonzero(meta.Attack_Type.to_numpy()=='Steering Tampering'),
 }
 # Type-specific one-v-normal test: all GEM benign frames + a single attack type
 for cat in ['DoS','Brake Tampering','Steering Tampering']:
  group_index[f'{cat} vs all benign']=np.flatnonzero(neg | (meta.Attack_Type.to_numpy()==cat))
 all_metrics=[]; type_rows=[]; masked_rows=[]; summary_preds=[]
 for algo in ('central','equal','fedavg','fedprox'):
  for seed in (11,22,33,44,55):
   net=gem.Net()
   if algo=='central':
    cp=torch.load(io.BytesIO(z.read(f'results/models/central_seed{seed}.pt')),weights_only=True,map_location='cpu')
    state=cp['state'];th=float(cp['threshold'])
   else:
    state=torch.load(io.BytesIO(z.read(f'results/models/fl_{algo}_k32_a1.0_s{seed}.pt')),weights_only=True,map_location='cpu')
    ths=reference.loc[(reference.setting==f'{algo}_k32_a1.0')&(reference.seed==seed),'threshold']
    assert len(ths)==1, (algo,seed)
    th=float(ths.iloc[0])
   net.load_state_dict(state)
   t0=time.perf_counter()
   p=gem.predict(net,xs)
   inf_seconds=time.perf_counter()-t0
   print(f'{algo} seed={seed} threshold={th:.5g} model_infer_s={inf_seconds:.4f}',flush=True)
   yp=(p>=th).astype(np.int8)
   type_recalls={}
   for name,idx in group_index.items():
    yy=y[idx]; pp=p[idx]
    mm=gem.score(yy,pp,th)
    mm.update(dataset=name,model=algo,seed=seed,train_dataset='HCRL Car-Hacking',
              threshold_policy='Saved HCRL validation threshold; frozen',
              feature_dim=12,seconds_full_batch=inf_seconds,
              infer_frames_per_sec=len(y)/max(inf_seconds,1e-10),
              no_refit=True,application='external GEM-CAN R2')
    mm['ood_frac_abs_z_gt5']=float((np.abs(xs[idx])>5).any(axis=1).mean())
    mm['cost_fp1_fn5_per_frame']=(mm['fp']+5*mm['fn'])/max(mm['n'],1)
    all_metrics.append(mm)
    if name in ['DoS attack only','Brake tampering only','Steering tampering only']:
     type_recalls[name]=mm['recall']
     type_rows.append(dict(model=algo,seed=seed,attack=name,attack_count=mm['attack'],
                           recall=mm['recall'],missed_attacks=mm['fn'],detected_attacks=mm['tp'],
                           mean_attack_prob=float(pp.mean()),median_attack_prob=float(np.median(pp))))
   summary_preds.append(dict(model=algo,seed=seed,threshold=th,
        macro_attack_recall=float(np.mean(list(type_recalls.values()))),
        dos_recall=type_recalls['DoS attack only'],brake_recall=type_recalls['Brake tampering only'],
        steering_recall=type_recalls['Steering tampering only'],
        total_model_inference_seconds=inf_seconds))
   for nf in [2,4,6,8,10,12]:
    if nf==12:
     p_mask=p;secs=inf_seconds
    else:
     mx=xs.copy();mx[:,nf:]=0.0
     mt0=time.perf_counter();p_mask=gem.predict(net,mx);secs=time.perf_counter()-mt0
    m=gem.score(y,p_mask,th)
    m.update(model=algo,seed=seed,retained_features=nf,masked_features=12-nf,
             setting='frozen model; trailing standardized features masked to training mean',
             inference_seconds=secs)
    # Attack-specific recall for each masked condition
    for cat in ['DoS','Brake Tampering','Steering Tampering']:
     ix=group_index[{'DoS':'DoS attack only','Brake Tampering':'Brake tampering only','Steering Tampering':'Steering tampering only'}[cat]]
     m[cat.replace(' ','_')+'_recall']=float((p_mask[ix]>=th).mean())
    masked_rows.append(m)

 pd.DataFrame(all_metrics).to_csv(outdir/'gemcan_r2_detailed_20model_metrics.csv',index=False)
 pd.DataFrame(type_rows).to_csv(outdir/'gemcan_r2_per_attack_recall.csv',index=False)
 pd.DataFrame(masked_rows).to_csv(outdir/'gemcan_r2_2to12_feature_masking.csv',index=False)
 pd.DataFrame(summary_preds).to_csv(outdir/'gemcan_r2_macro_attack_recall.csv',index=False)

 po=pd.DataFrame(all_metrics)
 ss=po[po.dataset=='Pooled GEM-CAN R2'].copy()
 summed=ss.groupby('model').agg({c:['mean','std','min','max'] for c in stat_cols+['tn','fp','fn','tp','cost_fp1_fn5_per_frame','ood_frac_abs_z_gt5']})
 summed.columns=['_'.join(k) for k in summed.columns]
 summed=summed.reset_index()
 types=pd.DataFrame(summary_preds).groupby('model').agg({'macro_attack_recall':['mean','std'],'dos_recall':['mean','std'],
                              'brake_recall':['mean','std'],'steering_recall':['mean','std']})
 types.columns=['_'.join(k) for k in types.columns]
 summed=summed.merge(types.reset_index(),on='model')
 summed.to_csv(outdir/'gemcan_r2_model_comparison_summary.csv',index=False)

 subset=po[po.dataset.isin(['Normal-driving only','Attack-session only','Attack-session normal'])]
 subset.groupby(['model','dataset'])[['accuracy','f1','recall','specificity','fpr','false_alarms_per_10k']].mean().to_csv(outdir/'gemcan_r2_recording_comparison.csv')
 orig_c=original.loc[original.setting=='classifier_only'].groupby('setting')[['accuracy','f1','recall','fpr','roc_auc']].mean().reset_index()
 orig_f=reference.loc[reference.setting.isin([f'{x}_k32_a1.0' for x in ['equal','fedavg','fedprox']])].groupby('setting')[['accuracy','f1','recall','fpr','roc_auc']].mean().reset_index()
 orig=pd.concat([orig_c,orig_f],ignore_index=True);orig['model']=orig.setting.map(lambda a:'central' if a=='classifier_only' else a.split('_')[0])
 comparison=orig.merge(summed,left_on='model',right_on='model')
 comparison.to_csv(outdir/'gemcan_r2_hcrl_vs_external.csv',index=False)
 audit_obj={ 'source_archive':str(datasetzip),'trained_archive':str(modelzip),
            'source_archive_sha256':sha(datasetzip),'trained_archive_sha256':sha(modelzip),
            'dataset':'GEM-CAN R2','models':20,'five_seeds':[11,22,33,44,55],
            'train':'HCRL Car-Hacking','training_new':False,'scaler_refit':False,
            'threshold_recalibration':False,'features':fnames,'notes':[
                'Each recording independently sorted and engineered, with temporal features reset at recording boundaries.',
                'Pooled evaluation combines two different recordings. Pooled prevalence is a sampling artifact; report both session-specific metrics.',
                '2,4,6,8,10 features are input mask stress tests, not trained smaller networks and not physical CAN signal subsets.',
                'Layer I robust guard fit center and scale not archived; unavailable as frozen out-of-domain test.',
                'Layer IV aggregation is training-time. Checkpoints evaluated individually with original saved thresholds.',
                'Layer III policy threshold frozen from HCRL validation, no calibration on GEM-CAN labels.',
                'No dynamic latency or ECU deployment measured; speed measured in the current host Python environment.',
                'Dataset CSV physically contains 43,444 attack-session frames vs README approximate 43,443.'
            ],'files':audit,'versions':{'torch':torch.__version__,'sklearn':sklearn_version,
                                       'pandas':pd.__version__,'numpy':np.__version__},
            'external_ood_fraction_z5':float((np.abs(xs)>5).any(axis=1).mean())}
 (outdir/'protocol_audit.json').write_text(json.dumps(audit_obj,indent=2))

 print('\nFINAL SUMMARY')
 print(summed[['model','accuracy_mean','f1_mean','recall_mean','specificity_mean','fpr_mean','roc_auc_mean','average_precision_mean','macro_attack_recall_mean','dos_recall_mean','brake_recall_mean','steering_recall_mean']].to_string(index=False))
 print('OUTPUTS',str(outdir),flush=True)
