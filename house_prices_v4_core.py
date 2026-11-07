"""Shared V4 utilities; the starting feature and preprocessing code is the group's V3.
No external sale prices are used. Fold fitting must be performed by callers.
"""
from pathlib import Path
from time import perf_counter
import ctypes, hashlib, importlib.metadata, json, os, sys, warnings
ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = (
    ROOT / 'Dataset',
    ROOT / 'house-prices-advanced-regression-techniques',
    ROOT / 'Proj' / 'Dataset',
    Path('/kaggle/input/house-prices-advanced-regression-techniques'),
)
DATA = next((folder for folder in DATA_CANDIDATES
             if all((folder / name).is_file() for name in ('train.csv', 'test.csv'))),
            DATA_CANDIDATES[0])
OUT = ROOT / 'output' / 'project' / 'v4'
for folder in [ROOT/'tmp'/'v2_deps', ROOT/'tmp'/'public_solution_audit'/'deps']:
    if folder.is_dir() and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
runtime = ROOT/'tmp'/'public_solution_audit'/'openmp'/'libomp.dylib'
if sys.platform == 'darwin' and runtime.is_file():
    ctypes.CDLL(str(runtime), mode=ctypes.RTLD_GLOBAL)
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def load_data():
    train = pd.read_csv(DATA/'train.csv',keep_default_na=False,na_values=['','NA'])
    test = pd.read_csv(DATA/'test.csv',keep_default_na=False,na_values=['','NA'])
    assert train.Id.is_unique and test.Id.is_unique
    assert train.SalePrice.notna().all() and (train.SalePrice>0).all()
    return train,test,np.log(train.SalePrice.to_numpy())

def rmse(y,p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))

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
    for c in d.select_dtypes(exclude="number").columns:
        d[c]=d[c].astype(object).where(d[c].notna(),np.nan)
    return d


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


def prepare_native(a, b):
    a = a.copy()
    b = b.copy()
    frontage = a.groupby("Neighborhood").LotFrontage.median()
    global_frontage = a.LotFrontage.median()
    for frame in [a, b]:
        frame["LotFrontage"] = frame.LotFrontage.fillna(frame.Neighborhood.map(frontage)).fillna(global_frontage)
        for col in frame.select_dtypes(include="object"):
            frame[col] = frame[col].fillna("Unknown").astype(str)
    return a, b, a.select_dtypes(include="object").columns.tolist()


def build_model(spec):
    if spec["family"] == "svr":
        return SVR(**spec["params"])
    if spec["family"] == "xgb":
        return XGBRegressor(**spec["params"], tree_method="hist", objective="reg:squarederror", n_jobs=4, random_state=42)
    return CatBoostRegressor(**spec["params"])


def v3_specs():
    return [
        {'name':'v3_svr','family':'svr','params':{'C':10,'gamma':0.0001,'epsilon':0.01},'clip':0.99,'weight':0.5},
        {'name':'v3_xgb','family':'xgb','params':{'n_estimators':2000,'learning_rate':0.025,'max_depth':3,'min_child_weight':3,'subsample':0.8,'colsample_bytree':0.7,'reg_lambda':3,'reg_alpha':0.01},'clip':False,'weight':0.1},
        {'name':'v3_cat','family':'cat','params':{'iterations':1800,'depth':6,'learning_rate':0.03,'l2_leaf_reg':5,'loss_function':'RMSE','random_seed':42,'thread_count':2,'verbose':False,'allow_writing_files':False},'weight':0.4}
    ]

def run_v3_baseline(seed=42,force=False):
    folder=OUT/'v3_reproduction';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'oof_seed{seed}.npz'
    train,test,y=load_data()
    x=make_features(train.drop(columns=['Id','SalePrice']))
    predictions={}; records=[]
    if path.exists() and not force:
        cached=np.load(path)
        return {name:cached[name] for name in cached.files}
    for spec in v3_specs():
        pred=np.full(len(train),np.nan);start=perf_counter()
        for fold,(a,b) in enumerate(KFold(5,shuffle=True,random_state=seed).split(x)):
            model=build_model(spec)
            if spec['family']=='cat':
                xa,xb,cats=prepare_native(x.iloc[a],x.iloc[b])
                model.fit(xa,y[a],cat_features=cats)
                pred[b]=model.predict(xb)
            else:
                xa,xb,sa,sb=prepare(x.iloc[a],x.iloc[b],clip=spec['clip'])
                with threadpool_limits(limits=1):
                    model.fit(sa if spec['family']=='svr' else xa,y[a])
                    pred[b]=model.predict(sb if spec['family']=='svr' else xb)
            records.append({'model':spec['name'],'seed':seed,'fold':fold,'rmse':rmse(y[b],pred[b])})
        assert np.isfinite(pred).all()
        predictions[spec['name']]=pred
        print(f"V3 seed {seed} {spec['name']} {rmse(y,pred):.9f} in {perf_counter()-start:.1f}s",flush=True)
    predictions['v3_blend']=sum(s['weight']*predictions[s['name']] for s in v3_specs())
    np.savez_compressed(path,**predictions)
    pd.DataFrame(records).to_csv(folder/f'fold_scores_seed{seed}.csv',index=False)
    scores={name:rmse(y,pred) for name,pred in predictions.items()}
    (folder/f'scores_seed{seed}.json').write_text(json.dumps(scores,indent=2)+'\n')
    print('V3 baseline',seed,scores,flush=True)
    return predictions

if __name__=='__main__':
    run_v3_baseline()
