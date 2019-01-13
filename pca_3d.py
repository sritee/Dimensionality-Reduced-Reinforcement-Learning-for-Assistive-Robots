#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:40:12 2018

@author: sritee
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection

from joblib import dump, load

for dim in range(3,4):
    #pipeline = Pipeline([('pca', PCA(n_components=1))])
    #pipeline = Pipeline([('std',StandardScaler()),('pca', PCA(n_components=3))])
    pipeline = Pipeline([('std',StandardScaler()),('pca', PCA(n_components=dim+1)),('rscale',RobustScaler(quantile_range=(10,90)))])
    #pipeline = Pipeline([('std',StandardScaler()),('pca', PCA(n_components=dim)),('rscale',QuantileTransformer(n_quantiles=100))])
    #pipeline = Pipeline([('std',StandardScaler()),('pca', PCA(n_components=3)),('rscale',RobustScaler(quantile_range=(10,90)))])
    #pipeline = Pipeline([('std',StandardScaler()),('random',GaussianRandomProjection(n_components=3)),('rscale',RobustScaler(quantile_range=(10,90)))])
    #pipeline = Pipeline([('rscale',RobustScaler(quantile_range=(10,90)))])
    
    
    t= np.load('./3D_demos/good_demos.npy')
    #t=np.load('MCar4D/good_demos.npy')
    
    data=[]
    
    for g in range(t.shape[0]):
        
        for p in range(len(t[g])):
            
            #data.append(t[g][p][:-1])
            data.append(t[g][p])
    
    d=np.array(data)
    
    print(d[0])
    
    g=pipeline.fit_transform(d)
    
    pca=pipeline.named_steps['pca']
    
    #scaler=pipeline.named_steps['std']
    #r_s=pipeline.named_steps['rscale']
    
    print(g[0])
    
    #print(pca.explain_variance_ratio_)
    
    #dump(pipeline,'pca_pipeline_{}.joblib'.format(dim+1))
    #dump(pipeline,'pca_fulldim.joblib'.format(dim+1))
    
