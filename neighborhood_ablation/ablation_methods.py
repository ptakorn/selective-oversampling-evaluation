"""Fixed-parameter synthetic pilot; no tuning or test-driven selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import json, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, average_precision_score, roc_auc_score

PARAMS=dict(kmin=3,k0=10,kmax=30,alpha=1.,lam=.75,kappa=2.,xi=2.,low=.30,high=.50,dis=.60,amb=.30,wmin=.10,beta=1.)

def data(n,seed):
    rng=np.random.default_rng(seed); n1=n//11
    y=np.r_[np.zeros(n-n1,int),np.ones(n1,int)]
    x=rng.normal(size=(n,2)); x[y==1,0]+=2
    order=rng.permutation(n); return x[order],y[order]

def corrupt(x,true,kind,seed):
    rng=np.random.default_rng(seed); y=true.copy(); bad=np.zeros(len(y),bool)
    for c in (0,1):
        ids=np.flatnonzero(true==c); count=int(np.floor(.10*len(ids))) if kind!='clean' else 0
        if kind=='dispersed': chosen=rng.choice(ids,count,replace=False)
        else:
            center=np.array([2*c,0]); chosen=ids[np.argsort(np.sum((x[ids]-center)**2,axis=1),kind='stable')[:count]]
        bad[chosen]=True
    y[bad]=1-y[bad]; return y,bad

def proposed(z,y,p,seed,mode="full", neighborhood="adaptive"):
    n=len(y); km=p['kmax']; nn=NearestNeighbors(n_neighbors=km+1).fit(z)
    dd,ii=nn.kneighbors(z)
    # Explicitly remove focal index (important with duplicate coordinates).
    ids=np.array([row[row!=i][:km] for i,row in enumerate(ii)])
    dist=np.array([dd[i][ii[i]!=i][:km] for i in range(n)])
    s=dist[:,p['k0']-1]; radius=p['alpha']*np.maximum(np.median(s[ids[:,:p['k0']]],axis=1),1e-12)
    scales=[]
    for t in (.5,1.,2.):
        sizes=(np.full(n, int(round(p['k0']*t)), dtype=int) if neighborhood=='fixed' else np.clip((dist<=radius[:,None]*t).sum(axis=1),p['kmin'],km))
        assert np.all((sizes>=1)&(sizes<=km))
        mask=np.arange(km)[None,:]<sizes[:,None]
        h=np.maximum(dist[np.arange(n),sizes-1],1e-12)
        w=np.exp(-dist**2/(2*h[:,None]**2))*mask; w/=w.sum(axis=1)[:,None]
        q=(w*(y[ids]==y[:,None])).sum(axis=1)
        prior=np.bincount(y,minlength=2)/n
        qb=(q/prior[y])/(q/prior[y]+(1-q)/prior[1-y])
        eff=1/(w*w).sum(axis=1); expansion=dist[np.arange(n),sizes-1]/(radius*t)
        scales.append((sizes,w,qb,eff,expansion))
    sizes,w,_,_,_=scales[1]; sets=[set(ids[i,:sizes[i]]) for i in range(n)]
    mutual=np.array([[i in sets[j] for j in row] for i,row in enumerate(ids)])
    m=(w*(y[ids]==y[:,None])*mutual).sum(axis=1)
    qb=np.array([v[2] for v in scales]); r=p['lam']*qb.mean(axis=0)+(1-p['lam'])*m
    dis=1-qb.max(axis=0)
    g=(np.min([v[3] for v in scales],axis=0)>=p['kappa'])&(np.max([v[4] for v in scales],axis=0)<=p['xi'])
    frac=(w*(y[ids]==1)).sum(axis=1); a=np.zeros(n)
    for prob in (frac,1-frac):
        pos=prob>0; a[pos]-=prob[pos]*np.log2(prob[pos])
    dw=g&(r<=p['low'])&(dis>=p['dis'])
    cand=(y==1)&g&(r>=p['high'])&(a>=p['amb'])
    if mode=='ambiguity_only':
        cand=(y==1)&(a>=p['amb']); dw[:]=False
    partners={}
    for i in np.flatnonzero(cand):
        js=ids[i,:sizes[i]]
        if mode=='ambiguity_only': js=js[y[js]==1]
        else: js=js[(y[js]==1)&g[js]&(r[js]>=p['high'])&np.array([i in sets[j] for j in js])]
        if len(js): partners[i]=js
    sources=np.array(list(partners),int)
    if mode in ('no_downweight','unit_all'): dw[:]=False
    weights=np.where(dw,p['wmin']+(1-p['wmin'])*r,1.)
    budget=max(0,int(np.ceil(p['beta']*(y==0).sum()))-(y==1).sum())
    rng=np.random.default_rng(seed); synth=[]; sw=[]; endpoints=[]
    if len(sources) and budget:
        priority=a[sources] if mode=='ambiguity_only' else a[sources]*r[sources]; exact=budget*priority/priority.sum(); alloc=np.floor(exact).astype(int)
        alloc[np.argsort(-(exact-alloc),kind='stable')[:budget-alloc.sum()]]+=1
        for i,count in zip(sources,alloc):
            js=partners[i]; lookup={j:v for j,v in zip(ids[i],w[i])}; probs=np.array([lookup[j] for j in js]); probs/=probs.sum()
            chosen=rng.choice(js,size=count,p=probs); nu=rng.random(count)
            synth.extend(z[i]+nu[:,None]*(z[chosen]-z[i])); sw.extend(np.ones(count) if mode in ('unit_synthetic','unit_all','ambiguity_only') else np.minimum(r[i],r[chosen])); endpoints.extend(zip(np.full(count,i),chosen))
    zz=np.vstack([z,np.array(synth).reshape(-1,z.shape[1])]); yy=np.r_[y,np.ones(len(synth),int)]; ww=np.r_[weights,sw]
    assert np.isfinite(zz).all() and np.isfinite(ww).all() and (ww>0).all()
    assert not np.any(dw[sources]) and (len(synth)==budget or len(sources)==0)
    return zz,yy,ww,dict(dw=dw,sources=sources,endpoints=np.array(endpoints,int).reshape(-1,2),original_weights=weights,g=g,r=r,a=a,sizes=np.array([v[0] for v in scales]))

def smote(z,y,seed):
    rng=np.random.default_rng(seed); minor=np.flatnonzero(y==1); budget=(y==0).sum()-len(minor)
    nn=NearestNeighbors(n_neighbors=6).fit(z[minor]); ii=nn.kneighbors(z[minor],return_distance=False)
    partners=np.array([row[row!=i][:5] for i,row in enumerate(ii)])
    src=rng.integers(len(minor),size=budget); dst=partners[src,rng.integers(5,size=budget)]
    syn=z[minor[src]]+rng.random((budget,1))*(z[minor[dst]]-z[minor[src]])
    return np.vstack([z,syn]),np.r_[y,np.ones(budget,int)],np.ones(len(y)+budget),dict(endpoints=np.c_[minor[src],minor[dst]],sources=np.unique(minor[src]),dw=np.zeros(len(y),bool),original_weights=np.ones(len(y)))

def safe(num,den): return num/den if den else np.nan

