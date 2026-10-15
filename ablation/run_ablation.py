from methods import *
from sklearn.metrics import balanced_accuracy_score
OUT=Path(__file__).parent
MODES=('full','no_downweight','unit_synthetic','unit_all','ambiguity_only')
rows=[]
for rep in range(30):
    x,true=data(2000,1000+rep); xt,yt=data(10000,2000+rep)
    scaler=StandardScaler().fit(x); z=scaler.transform(x); zt=scaler.transform(xt)
    for scenario in ('clean','dispersed','localized'):
        y,bad=corrupt(x,true,scenario,3000+rep)
        anchor=None
        for mode in MODES:
            zz,yy,ww,info=proposed(z,y,PARAMS,4000+rep,mode)
            if mode=='full': anchor=(zz.copy(),info['endpoints'].copy())
            if mode in ('no_downweight','unit_synthetic','unit_all'):
                assert np.array_equal(zz,anchor[0]) and np.array_equal(info['endpoints'],anchor[1]), 'Weight ablation changed generated samples'
            clf=LogisticRegression(C=1,max_iter=2000).fit(zz,yy,sample_weight=ww)
            pred=clf.predict(zt); score=clf.decision_function(zt); tn,fp,fn,tp=confusion_matrix(yt,pred).ravel()
            recall=tp/(tp+fn); spec=tn/(tn+fp); ep=info['endpoints']; valid=(true==1)&~bad; dw=info['dw']
            contaminated=bad[ep].any(axis=1) if len(ep) else np.array([],bool)
            rows.append(dict(rep=rep,scenario=scenario,method=mode,gmean=np.sqrt(recall*spec),recall=recall,precision=safe(tp,tp+fp),f1=f1_score(yt,pred),ap=average_precision_score(yt,score),synthetic=len(zz)-len(z),fdw_min=safe((dw&valid).sum(),valid.sum()),dw_recall=safe((dw&bad).sum(),bad.sum()),endpoint_contamination=safe(contaminated.sum(),len(contaminated)),mean_valid_min_weight=info['original_weights'][valid].mean()))
    print('Completed',rep+1,flush=True)
x=pd.DataFrame(rows); x.to_csv(OUT/'ablation_raw.csv',index=False)
old=pd.read_csv(OUT.parent/'pilot/pilot_raw.csv')
check=x[x.method=='full'].merge(old[old.method=='proposed'],on=['rep','scenario'],suffixes=('_new','_old'))
for metric in ('gmean','recall','precision','f1','ap','endpoint_contamination'):
    assert np.allclose(check[metric+'_new'],check[metric+'_old'],atol=1e-12,equal_nan=True),metric
base=old[old.method.isin(['none','smote'])][x.columns]
combined=pd.concat([base,x]); combined.to_csv(OUT/'combined_raw.csv',index=False)
summary=combined.groupby(['scenario','method']).agg({c:['mean','std'] for c in ['gmean','recall','precision','f1','ap','fdw_min','endpoint_contamination']})
summary.to_csv(OUT/'summary.csv'); print(summary.to_string())
ci=[]; rng=np.random.default_rng(987)
for scenario in x.scenario.unique():
    pv=combined[combined.scenario==scenario].pivot(index='rep',columns='method',values='gmean')
    for variant in MODES[1:]:
        delta=(pv[variant]-pv['full']).to_numpy(); boot=delta[rng.integers(0,30,(5000,30))].mean(axis=1)
        ci.append(dict(scenario=scenario,comparison=variant+' minus full',delta=delta.mean(),lower=np.quantile(boot,.025),upper=np.quantile(boot,.975)))
pd.DataFrame(ci).to_csv(OUT/'paired_intervals.csv',index=False)
print('All full-method reproduction and unchanged-sample checks passed')
