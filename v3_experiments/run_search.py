from pathlib import Path
from time import perf_counter
import json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'v3_experiments'
train = pd.read_csv(ROOT/'Dataset/train.csv', keep_default_na=False, na_values=['','NA'])
test = pd.read_csv(ROOT/'Dataset/test.csv', keep_default_na=False, na_values=['','NA'])
y = np.log(train.SalePrice.to_numpy())
raw = train.drop(columns=['Id','SalePrice'])

# Reproduce V2's fixed, row-wise transformations exactly.
nb = json.loads((ROOT/'house_prices_v2.ipynb').read_text())
ns={'train':train,'test':test,'pd':pd,'np':np}
for cell in nb['cells']:
    source=cell['source']; source=''.join(source) if isinstance(source,list) else source
    if cell['cell_type']=='code' and (source.startswith('y = np.log') or source.startswith('none_columns =') or source.startswith('data["TotalSF"] =')):
        exec(source,ns)
v2_features=ns['data'].iloc[:len(train)].copy()


def make_features(frame):
    d=frame.copy()
    for c in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
        d.loc[(d[c]<1800)|(d[c]>d.YrSold+1),c]=np.nan
    no_basement=d.TotalBsmtSF.eq(0)
    no_garage=d.GarageCars.eq(0)&d.GarageArea.eq(0)
    absence={'Alley':pd.Series(True,index=d.index),'Fence':pd.Series(True,index=d.index),
             'MiscFeature':d.MiscVal.eq(0),'PoolQC':d.PoolArea.eq(0),
             'FireplaceQu':d.Fireplaces.eq(0),'MasVnrType':d.MasVnrArea.eq(0)}
    for c in ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']:
        absence[c]=no_basement
    for c in ['GarageType','GarageFinish','GarageQual','GarageCond']:
        absence[c]=no_garage
    for c,mask in absence.items():
        d.loc[mask&d[c].isna(),c]='None'
    for c in ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']:
        d.loc[no_basement&d[c].isna(),c]=0
    d.loc[d.MasVnrType.eq('None')&d.MasVnrArea.isna(),'MasVnrArea']=0
    for c in ['MSSubClass','MoSold']:
        d[c]=d[c].astype(str)
    d['HouseAge']=(d.YrSold-d.YearBuilt).clip(lower=0)
    d['RemodelAge']=(d.YrSold-d.YearRemodAdd).clip(lower=0)
    d['GarageAge']=(d.YrSold-d.GarageYrBlt).clip(lower=0)
    d.loc[no_garage,'GarageAge']=0
    d=d.drop(columns='GarageYrBlt')
    d['TotalSF']=d.TotalBsmtSF+d['1stFlrSF']+d['2ndFlrSF']
    d['FinishedSF']=d.GrLivArea+d.BsmtFinSF1+d.BsmtFinSF2
    d['TotalBath']=d.FullBath+.5*d.HalfBath+d.BsmtFullBath+.5*d.BsmtHalfBath
    d['TotalOutdoorSF']=d.WoodDeckSF+d.OpenPorchSF+d.EnclosedPorch+d['3SsnPorch']+d.ScreenPorch
    for source,name in [('GarageArea','HasGarage'),('TotalBsmtSF','HasBasement'),
                        ('Fireplaces','HasFireplace'),('PoolArea','HasPool'),('2ndFlrSF','HasSecondFloor')]:
        d[name]=np.where(d[source].isna(),np.nan,d[source].gt(0).astype(float))
    d['WasRemodeled']=(d.YearRemodAdd!=d.YearBuilt).astype(int)
    d['IsNew']=(d.YearBuilt>=d.YrSold).astype(int)
    quality={'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
    for c in ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']:
        d[c]=d[c].map(quality)
    d['BsmtExposure']=d.BsmtExposure.map({'None':0,'No':0,'Mn':1,'Av':2,'Gd':3})
    d['GarageFinish']=d.GarageFinish.map({'None':0,'Unf':1,'RFn':2,'Fin':3})
    for c in ['BsmtFinType1','BsmtFinType2']:
        d[c]=d[c].map({'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})
    # A few explicit interactions, without generating all feature pairs.
    d['QualityLivingArea']=d.OverallQual*np.log1p(d.GrLivArea)
    d['QualityTotalArea']=d.OverallQual*np.log1p(d.TotalSF)
    d['QualityHouseAge']=d.OverallQual*d.HouseAge
    d['TotalQuality']=d.OverallQual+d.OverallCond
    return d

advanced=make_features(raw)


def prepare(a,b,scale='standard',advanced_mode=True,clip=False):
    a=a.copy();b=b.copy()
    if advanced_mode:
        frontage=a.groupby('Neighborhood').LotFrontage.median()
        global_frontage=a.LotFrontage.median()
        for d in [a,b]:
            d['LotFrontage']=d.LotFrontage.fillna(d.Neighborhood.map(frontage)).fillna(global_frontage)
        for c in a.select_dtypes(include='object'):
            a[c]=a[c].fillna('Unknown');b[c]=b[c].fillna('Unknown')
            common=a[c].value_counts(); keep=common[common>=3].index
            a[c]=a[c].where(a[c].isin(keep),'Other')
            b[c]=b[c].where(b[c].isin(keep),'Other')
        excluded={'YearBuilt','YearRemodAdd','YrSold','OverallQual','OverallCond','QualityLivingArea','QualityTotalArea','QualityHouseAge','TotalQuality'}
        log_cols=[c for c in a.select_dtypes(include=np.number) if c not in excluded and a[c].nunique()>10 and a[c].min()>=0 and a[c].skew()>.75]
        for c in log_cols:
            a[c]=np.log1p(a[c]);b[c]=np.log1p(b[c])
    if clip:
        for c in a.select_dtypes(include=np.number):
            if a[c].nunique()>20 and c not in ['YearBuilt','YearRemodAdd','YrSold']:
                upper=a[c].quantile(.995 if clip is True else clip)
                a[c]=a[c].clip(upper=upper);b[c]=b[c].clip(upper=upper)
    medians=a.median(numeric_only=True)
    modes=a.select_dtypes(include='object').mode().iloc[0]
    a=a.fillna(medians).fillna(modes).fillna(0)
    b=b.fillna(medians).fillna(modes).fillna(0)
    a=pd.get_dummies(a,dtype=float)
    b=pd.get_dummies(b,dtype=float).reindex(columns=a.columns,fill_value=0)
    a=a.to_numpy(dtype=np.float64);b=b.to_numpy(dtype=np.float64)
    scaler=StandardScaler() if scale=='standard' else RobustScaler()
    sa=scaler.fit_transform(a);sb=scaler.transform(b)
    assert all(np.isfinite(x).all() for x in [a,b,sa,sb])
    return a,b,sa,sb


def build(spec):
    family=spec['family'];p=spec['params']
    if family=='cat':return CatBoostRegressor(**p)
    if family=='ridge':return Ridge(**p)
    if family=='elastic':return ElasticNet(**p,max_iter=30000,tol=1e-5)
    if family=='svr':return SVR(**p)
    if family=='gbr':return GradientBoostingRegressor(**p,random_state=42)
    if family=='xgb':return XGBRegressor(**p,tree_method='hist',objective='reg:squarederror',n_jobs=4,random_state=42)
    raise ValueError(family)

cache={}
def get_cache(prep,scale,seed):
    key=(prep,scale,seed)
    if key not in cache:
        x=v2_features if prep.startswith('v2') else advanced
        if prep.startswith('reduced'):
            x=x.drop(columns=['QualityLivingArea','QualityTotalArea','QualityHouseAge','TotalQuality'])
        splits=list(KFold(5,shuffle=True,random_state=seed).split(x))
        if prep=='advanced_native':
            native=[]
            for tr,va in splits:
                a=x.iloc[tr].copy();b=x.iloc[va].copy()
                frontage=a.groupby('Neighborhood').LotFrontage.median();global_front=a.LotFrontage.median()
                for frame in [a,b]:
                    frame['LotFrontage']=frame.LotFrontage.fillna(frame.Neighborhood.map(frontage)).fillna(global_front)
                    for c in frame.select_dtypes(include='object'):frame[c]=frame[c].fillna('Unknown').astype(str)
                native.append((tr,va,(a,b,None,None)))
            cache[key]=native
            return cache[key]
        cache[key]=[(tr,va,prepare(x.iloc[tr],x.iloc[va],scale,not prep.startswith('v2'),(.99 if 'clip99' in prep else .98 if 'clip98' in prep else .995 if prep.endswith('clip') else False))) for tr,va in splits]
    return cache[key]

rows=[];predictions={};specifications={};fold_rows=[]

def evaluate(spec,seed=42,save=True):
    start=perf_counter();oof=np.zeros(len(y));scores=[]
    for fold,(tr,va,(a,b,sa,sb)) in enumerate(get_cache(spec['prep'],spec['scale'],seed),1):
        m=build(spec)
        linear=spec['family'] in ['ridge','elastic','svr']
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            with threadpool_limits(limits=1):
                if spec['family']=='cat':m.fit(a,y[tr],cat_features=a.select_dtypes(include='object').columns.tolist())
                else:m.fit(sa if linear else a,y[tr])
                oof[va]=m.predict(sb if linear else b)
        assert np.isfinite(oof[va]).all()
        score=float(np.sqrt(np.mean((y[va]-oof[va])**2)));scores.append(score)
        if save:fold_rows.append({'model':spec['name'],'fold':fold,'log_rmse':score})
    score=float(np.sqrt(np.mean((y-oof)**2)))
    row={'model':spec['name'],'oof_log_rmse':score,'fold_mean':float(np.mean(scores)),
         'fold_std':float(np.std(scores,ddof=1)),'seconds':perf_counter()-start,'seed':seed}
    print(json.dumps(row),flush=True)
    if save:
        rows.append(row);predictions[spec['name']]=oof;specifications[spec['name']]=spec
        pd.DataFrame(rows).sort_values('oof_log_rmse').to_csv(OUT/'model_search.csv',index=False)
        pd.DataFrame(fold_rows).to_csv(OUT/'fold_scores.csv',index=False)
        (OUT/'model_specs.json').write_text(json.dumps(specifications,indent=2))
        np.savez_compressed(OUT/'oof_predictions.npz',**predictions)
    return row,oof


def spec(name,family,params,prep='advanced',scale='standard'):
    return dict(name=name,family=family,params=params,prep=prep,scale=scale)

if __name__ == '__main__':
    candidates = json.loads((OUT/'all_model_specs.json').read_text())
    for candidate in candidates.values():
        evaluate(candidate)
    print('Search complete. Blend settings remain locked in selected_config.json.')
