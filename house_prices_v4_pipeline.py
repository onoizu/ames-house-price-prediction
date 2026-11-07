"""V4: a frozen, tested development recipe built from the group's V3.

The search record is under output/project/v4/search_*; this module reruns the
selected recipe, not the entire exploratory search. All original validation
rows remain in evaluation. No external sale prices or Id-specific rules exist.
"""
from pathlib import Path
import hashlib,importlib.metadata,json,warnings
from time import perf_counter
import house_prices_v4_core as h
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

ROOT=h.ROOT
OUT=h.OUT/'final'
CONFIG={
    'seeds':[42,2026], 'folds':5, 'target_rmse':.110,
    'components':[
        {'name':'ContextSVR','kind':'context_svr','weight':.4749247653507601,
         'params':{'C':3,'gamma':.0003,'epsilon':.02},
         'clip':.99,'metric_power':.5,'smoothing':2.,'context_strength':12.},
        {'name':'ClippedCatBoost','kind':'cat','weight':.23483327876275376,
         'clip':.99,'params':{'iterations':1800,'depth':6,'learning_rate':.03,
          'l2_leaf_reg':5,'loss_function':'RMSE','random_seed':42,'thread_count':2,
          'verbose':False,'allow_writing_files':False}},
        {'name':'NeighborhoodElasticNet','kind':'geometry_enet','weight':.22843472095279865,
         'clip':.975,'scale':'robust','params':{'alpha':.0005,'l1_ratio':.9}},
        {'name':'NeighborhoodSVR','kind':'geometry_svr','weight':.061807234933687547,
         'clip':.99,'scale':'standard','params':{'C':10,'gamma':.0001,'epsilon':.01}},
    ],
    'selection_protocol':'Candidate preprocessing, model settings and nonnegative weights selected on seed42 OOF; fixed recipe then checked on seed2026. Both remain development/stability results, not an untouched test.',
    'validation_rows_removed':0,
    'final_prediction':'Fit the frozen components on all labeled rows, then blend natural-log predictions and apply exp.',
}


def categorical_columns(frame):
    return frame.select_dtypes(exclude='number').columns.tolist()


def polynomial_features(raw):
    d=h.make_features(raw)
    for c in ['OverallQual','OverallCond','ExterQual','KitchenQual','BsmtQual']:
        d[c+'_squared']=d[c]**2
    d['OverallQual_cubed']=d.OverallQual**3
    d['ExterLiving']=d.ExterQual*np.log1p(d.GrLivArea)
    d['KitchenLiving']=d.KitchenQual*np.log1p(d.GrLivArea)
    d['BasementFinished']=d.BsmtQual*np.log1p(d.BsmtFinSF1+d.BsmtFinSF2)
    d['GarageQualityArea']=d.GarageQual*np.log1p(d.GarageArea)
    return d


def geometry_features(raw_a,raw_b):
    a,b=polynomial_features(raw_a),polynomial_features(raw_b)
    area=a.groupby('Neighborhood').GrLivArea.agg(['median','count'])
    area.loc[area['count']<5,'median']=np.nan
    quality=a.groupby('Neighborhood').OverallQual.agg(['median','count'])
    quality.loc[quality['count']<5,'median']=np.nan
    global_area=a.GrLivArea.median();global_quality=a.OverallQual.median()
    neighborhoods=sorted(a.Neighborhood.dropna().unique())
    for d in [a,b]:
        local_area=d.Neighborhood.map(area['median']).fillna(global_area)
        local_quality=d.Neighborhood.map(quality['median']).fillna(global_quality)
        delta=np.log1p(d.GrLivArea)-np.log1p(local_area)
        d['NeighborhoodMedianArea']=local_area
        d['NeighborhoodLivingRatio']=d.GrLivArea/local_area.clip(lower=1)
        d['NeighborhoodLogAreaDelta']=delta
        d['NeighborhoodQualityDelta']=d.OverallQual-local_quality
        d['RelativeAreaQualityInteraction']=delta*(d.OverallQual-local_quality)
        d['Neighborhood_SaleCondition']=d.Neighborhood.astype(str)+'__'+d.SaleCondition.astype(str)
        d['Neighborhood_OverallQual']=d.Neighborhood.astype(str)+'__'+d.OverallQual.astype(str)
        for n in neighborhoods:
            d['NeighborhoodAreaSlope_'+n]=delta*d.Neighborhood.eq(n).astype(float)
    return a,b


def context_keys(raw):
    n=raw.Neighborhood.astype(str)
    return pd.DataFrame({'Neighborhood':n,
        'Neighborhood_SaleCondition':n+'__'+raw.SaleCondition.astype(str),
        'Neighborhood_OverallQual':n+'__'+raw.OverallQual.astype(str)},index=raw.index)


def context_columns(raw_a,raw_b,y,smoothing,strength):
    """Only current training targets enter the statistics.

    Each training row's unscaled context value excludes its own target from
    both group sum and global prior. Validation uses training-only tables.
    """
    a=context_keys(raw_a).reset_index(drop=True)
    b=context_keys(raw_b).reset_index(drop=True)
    y=np.asarray(y,dtype=float)
    loo_prior=(y.sum()-y)/(len(y)-1);prior=float(y.mean())
    ta,tb=[],[]
    for c in a.columns:
        table=pd.DataFrame({'group':a[c],'target':y}).groupby('group').target.agg(['sum','count'])
        sums=a[c].map(table['sum']).to_numpy()-y
        counts=a[c].map(table['count']).to_numpy()-1
        av=(sums+smoothing*loo_prior)/(counts+smoothing)
        bv=(b[c].map(table['sum']).fillna(0).to_numpy()+smoothing*prior)/(
            b[c].map(table['count']).fillna(0).to_numpy()+smoothing)
        scaler=StandardScaler()
        ta.append(scaler.fit_transform(av[:,None]).ravel()*strength)
        tb.append(scaler.transform(bv[:,None]).ravel()*strength)
    return np.column_stack(ta),np.column_stack(tb)


def distance_weights(matrix,y,power):
    centered=matrix-matrix.mean(axis=0);yc=y-y.mean()
    denominator=np.sqrt((centered**2).sum(axis=0)*(yc**2).sum())
    corr=np.abs(np.divide(centered.T@yc,denominator,out=np.zeros(matrix.shape[1]),where=denominator>1e-12))
    return (np.maximum(corr,.01)/max(float(corr.mean()),.01))**power


def prepare_component(raw_a,raw_b,y,spec):
    kind=spec['kind']
    if kind.startswith('geometry'):
        a,b=geometry_features(raw_a,raw_b)
        _,_,sa,sb=h.prepare(a,b,scale=spec['scale'],clip=spec['clip'])
        return sa,sb,None
    a,b=h.make_features(raw_a),h.make_features(raw_b)
    if kind=='context_svr':
        _,_,sa,sb=h.prepare(a,b,scale='standard',clip=spec['clip'])
        weight=distance_weights(sa,np.asarray(y),spec['metric_power'])
        ca,cb=context_columns(raw_a,raw_b,y,spec['smoothing'],spec['context_strength'])
        return np.column_stack([sa*weight,ca]),np.column_stack([sb*weight,cb]),None
    a,b,cats=h.prepare_native(a,b)
    for col in a.select_dtypes(include=np.number):
        if a[col].nunique()>20 and col not in ['YearBuilt','YearRemodAdd','YrSold']:
            upper=a[col].quantile(spec['clip'])
            a[col]=a[col].clip(upper=upper);b[col]=b[col].clip(upper=upper)
    return a,b,cats


def model_for(spec):
    if spec['kind'] in ['context_svr','geometry_svr']:return SVR(**spec['params'])
    if spec['kind']=='geometry_enet':return ElasticNet(**spec['params'],max_iter=30000,tol=1e-5)
    return h.CatBoostRegressor(**spec['params'])


def predict_component(raw_a,raw_b,y,spec):
    a,b,cats=prepare_component(raw_a,raw_b,y,spec)
    model=model_for(spec)
    with threadpool_limits(limits=1):
        if cats is None:model.fit(a,y)
        else:model.fit(a,y,cat_features=cats)
        prediction=np.asarray(model.predict(b),dtype=float)
    assert np.isfinite(prediction).all()
    return prediction


def signature():
    dependencies=['numpy','pandas','scipy','scikit-learn','xgboost','catboost','six']
    record={'config':CONFIG,'versions':{n:importlib.metadata.version(n) for n in dependencies},
        'inputs':{n:hashlib.sha256((h.DATA/n).read_bytes()).hexdigest() for n in ['train.csv','test.csv']},
        'source':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                  for n in ['house_prices_v4_core.py','house_prices_v4_pipeline.py']}}
    record['signature']=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    return record


def run_experiment(force=False):
    OUT.mkdir(parents=True,exist_ok=True)
    stamp=signature();manifest_path=OUT/'manifest.json';summary_path=OUT/'summary.json'
    if not force and manifest_path.exists() and summary_path.exists():
        previous=json.loads(manifest_path.read_text())
        hashes=previous.get('artifact_sha256',{})
        if previous.get('signature')==stamp['signature'] and hashes and all(
            (OUT/n).is_file() and hashlib.sha256((OUT/n).read_bytes()).hexdigest()==digest
            for n,digest in hashes.items()):
            print('V4 cache verified: matching inputs, recipe, source, versions and output hashes.')
            return json.loads(summary_path.read_text())
    started=perf_counter();train,test,y=h.load_data()
    raw=train.drop(columns=['Id','SalePrice']);raw_test=test.drop(columns='Id')
    names=[s['name'] for s in CONFIG['components']]
    weights=np.array([s['weight'] for s in CONFIG['components']]);assert np.isclose(weights.sum(),1)
    comparison=[];fold_records=[];component_scores=[];seed_results={};saved=[]
    for seed in CONFIG['seeds']:
        baseline=h.run_v3_baseline(seed=seed,force=force)['v3_blend']
        oof={n:np.full(len(y),np.nan) for n in names};fold_id=np.full(len(y),-1,dtype=int)
        folds=list(h.KFold(CONFIG['folds'],shuffle=True,random_state=seed).split(raw))
        for spec in CONFIG['components']:
            start=perf_counter()
            for fold,(a,b) in enumerate(folds):
                fold_id[b]=fold
                oof[spec['name']][b]=predict_component(raw.iloc[a],raw.iloc[b],y[a],spec)
            score=h.rmse(y,oof[spec['name']])
            component_scores.append({'name':spec['name'],'seed':seed,'log_rmse':score})
            print(f"Seed {seed}: {spec['name']} OOF {score:.9f} ({perf_counter()-start:.1f}s)",flush=True)
        pred=np.column_stack([oof[n] for n in names])@weights
        assert np.isfinite(pred).all() and (fold_id>=0).all()
        seed_results[seed]={'v3':h.rmse(y,baseline),'v4':h.rmse(y,pred)}
        for name,value in [('V3',baseline),('V4',pred)]:
            comparison.append({'model':name,'seed':seed,'log_rmse':h.rmse(y,value)})
            for fold,(_,b) in enumerate(folds):
                fold_records.append({'model':name,'seed':seed,'fold':fold,'log_rmse':h.rmse(y[b],value[b]),'rows':len(b)})
        table=pd.DataFrame({'Id':train.Id,'fold':fold_id,'target_log':y,'V3':baseline,'V4':pred,**oof})
        path=f'oof_seed{seed}.csv';table.to_csv(OUT/path,index=False);saved.append(path)
        print('COMPARISON',seed,seed_results[seed],flush=True)
    improved=all(row['v4']<row['v3'] for row in seed_results.values())
    if not improved:
        (OUT/'failed_improvement_gate.json').write_text(json.dumps(seed_results,indent=2))
        raise RuntimeError('The frozen V4 did not improve V3 on both recorded splits; no V4 submission is exported.')
    test_pred={}
    for spec in CONFIG['components']:
        test_pred[spec['name']]=predict_component(raw,raw_test,y,spec)
    test_log=np.column_stack([test_pred[n] for n in names])@weights
    submission=pd.DataFrame({'Id':test.Id,'SalePrice':np.exp(test_log)})
    assert len(submission)==len(test) and submission.Id.tolist()==test.Id.tolist()
    assert np.isfinite(submission.SalePrice).all() and (submission.SalePrice>0).all()
    submission.to_csv(OUT/'submission_v4.csv',index=False)
    submission.to_csv(ROOT/'submission_v4.csv',index=False)
    pd.DataFrame(comparison).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(fold_records).to_csv(OUT/'fold_scores.csv',index=False)
    pd.DataFrame(component_scores).to_csv(OUT/'component_scores.csv',index=False)
    pd.DataFrame({'name':names,'weight':weights}).to_csv(OUT/'weights.csv',index=False)
    summary={'v3_seed42':seed_results[42]['v3'],'v4_seed42':seed_results[42]['v4'],
        'v3_seed2026':seed_results[2026]['v3'],'v4_seed2026':seed_results[2026]['v4'],
        'target_rmse':CONFIG['target_rmse'],
        'target_met':all(row['v4']<CONFIG['target_rmse'] for row in seed_results.values()),
        'improvement_verified':improved,
        'relative_improvement_seed42':1-seed_results[42]['v4']/seed_results[42]['v3'],
        'relative_improvement_seed2026':1-seed_results[2026]['v4']/seed_results[2026]['v3'],
        'selected_components':names,'config':CONFIG,'runtime_seconds':perf_counter()-started,
        'n_train':len(train),'n_test':len(test),'v3_notebook_saved_score':.117973,
        'selection_note':CONFIG['selection_protocol'],
        'cache_status':'Models were executed; future matching runs may reuse hashed results.',
        'version_note':'The V3 notebook saved 0.117973; local V3 reproduction differs slightly because package versions differ. Both versions above are evaluated locally under identical seeds and data.',
        'no_external_labels':True,'kaggle_submitted':False}
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    saved+=['submission_v4.csv','comparison.csv','fold_scores.csv','component_scores.csv','weights.csv','summary.json']
    stamp['artifact_sha256']={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in saved}
    manifest_path.write_text(json.dumps(stamp,indent=2)+'\n')
    print('V4 COMPLETE',json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    import sys
    run_experiment(force='--force' in sys.argv)
