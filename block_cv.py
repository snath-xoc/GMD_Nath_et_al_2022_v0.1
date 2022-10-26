'''
Functions for Blocked cross-validation
'''

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from load_data import load_meta_data
from scipy.stats import mode

def create_training_set(y,x,key,i_mon,n_steps=3):
    
    '''
    Pipelining of training data to GAM
    
    Inputs
    ------
    
    y: target variable
    x: predictor set
    key: which predictor (lon, lat, orog of treeFrac)
    i_mon: month index
    n_steps: number of simulations i.e. 3 for Aff, Def, Ref (so lon, lat and orog can be repeated)
    
    Outputs
    -------
    
    numpy array of individual predictor to be added to the predictor matrix
    
    '''
    idx_l, _, _, _, _ = load_meta_data()
    
    
    if key in ['lon','lat','orog','treeFrac_init']:
        x[np.isnan(x)]=0
        psuedo_x=np.repeat(x.reshape(-1,idx_l.sum()), n_steps, axis=0)
        return psuedo_x.reshape(-1,1)

    elif 'treeFrac' in key:

        
        psuedo_x=np.vstack((x[0].reshape(1,-1),x[1].reshape(1,-1),
                            np.zeros([1,idx_l.sum()])))
        psuedo_x[pd.isnull(psuedo_x)]=0
        return psuedo_x.reshape(-1,1)
    
def create_prediction_set(y,x,key,i_x,i_mon):
    
    
    '''
    Pipelining of training data to GAM
    
    Inputs
    ------
    
    y: target variable
    x: predictor set
    key: which predictor (lon, lat, orog of treeFrac)
    i_mon: month index
    i_x: number of simulation to be predicted for i.e. 0,1,2 for Aff, Def, Ref
    
    Outputs
    -------
    
    numpy array of individual predictor to be added to the predictor matrix
    
    '''
    idx_l, _, _, _, _ = load_meta_data()
    
    
    if key in ['lon','lat','orog','treeFrac_init']:
        psuedo_x=np.repeat(x.reshape(-1,idx_l.sum())[:,:], 1, axis=0)
        psuedo_x[np.isnan(psuedo_x)]=0
        return(psuedo_x.reshape(-1,1))
    elif 'treeFrac' in key:
        
        psuedo_x=np.vstack((x[0].reshape(1,-1),x[1].reshape(1,-1),
                            np.zeros([1,idx_l.sum()])))
        psuedo_x[pd.isnull(psuedo_x)]=0
        return psuedo_x[i_x,:].reshape(-1,1)
    
def calculate_rmse(X,Y):
    
    '''
    Calculate rmse assuming vector input
    '''
    
    return np.nanmean(np.sqrt((X-Y)**2))

def coeff_of_agreement(X,Y):
    
    '''
    Calculate coefficient of determination assuming vector input
    '''
    
    return 1-(np.nanmean((X-Y)**2)/(np.nanvar(X)+np.nanvar(Y)+((np.nanmean(X)-np.nanmean(Y))**2)))

def calculate_mae(X,Y):
    
    '''
    Calculate mae assuming vector input
    '''
    
    return np.nanmean(np.abs(X-Y))

def calculate_rmse_gp(X,Y):
    
    '''
    Calculate rmse assuming scalar input
    '''
    
    return np.sqrt((X-Y)**2)

def calculate_mae_gp(X,Y):
    
    '''
    Calculate mae assuming scalar input
    '''
    
    return np.abs(X-Y)

def spatial_cv(mod,X,y,nr_feat,groups,n_splits,weights):
    
    '''
    Perform Blocked CV
    
    Input
    -----
    mod: model
    X: predictor matrix
    y: target variable
    nr_feat: number of predictors in predictor matrix
    groups: composite climate-geographical blocks
    n_splits: number of blocks
    weights: weights applied during training GAM
    
    Calculates mean test score and chooses lambda and number of basis functions (n_splines) accordingly
    
    Output
    ------
    
    dictionary of train and test CV results grouped into block and extent of tree cover change
    
    '''
    cv=GroupKFold(n_splits=n_splits)
    X=X.reshape(3,-1,nr_feat)
    
    treeFrac_idx=np.argwhere(np.logical_or(X[0,:,-1]>0.01,X[1,:,-1]<-0.01))
    group_kfold=cv.split(X.reshape(3,-1,nr_feat)[0,:,:],y[0,:],groups)
    
    # Create a nested list of train and test indices for each fold
    train_indices, test_indices = [list(traintest) for traintest in zip(*group_kfold)]
    group_cv = [*zip(train_indices,test_indices)]
    
    scores_train={}
    scores_test={}
    
    scores_train['Aff']={0:{},1:{},2:{},3:{},4:{}}
    scores_train['Def']={0:{},1:{},2:{},3:{},4:{}}
    
    scores_test['Aff']={0:{},1:{},2:{},3:{},4:{}}
    scores_test['Def']={0:{},1:{},2:{},3:{},4:{}}
        
    for train_idx,test_idx in group_cv:
        
        i_group=int(mode(groups[test_idx]).mode[0])
        
        train_idx = np.intersect1d(train_idx,treeFrac_idx)
        test_idx = np.intersect1d(test_idx,treeFrac_idx)
        
        est=mod.fit(X.reshape(3,-1,nr_feat)[:,train_idx,:].reshape(-1,nr_feat),y[:,train_idx].flatten(),
                    weights=np.nan_to_num(weights.reshape(3,-1)[:,train_idx].flatten(),posinf=0,neginf=0))

            
        for i_bin,treeFrac_bin in enumerate([[0.01,0.15],[0.15,0.3],[0.3,0.5],[0.5,0.8],[0.8,1]]):
            
            idx_bin=np.intersect1d(train_idx,np.argwhere(np.logical_and(X[0,:,-1]>=treeFrac_bin[0],X[0,:,-1]<treeFrac_bin[1])))
            try:
                scores_train['Aff'][i_bin][i_group]=[coeff_of_agreement(y[0,idx_bin].flatten(),est.predict(X.reshape(3,-1,nr_feat)[0,idx_bin,:].reshape(-1,nr_feat))),
                                                   calculate_rmse(y[0,idx_bin].flatten(),est.predict(X.reshape(3,-1,nr_feat)[0,idx_bin,:].reshape(-1,nr_feat))),
                                                   calculate_mae(y[0,idx_bin].flatten(),est.predict(X.reshape(3,-1,nr_feat)[0,idx_bin,:].reshape(-1,nr_feat)))]
            except:
                scores_train['Aff'][i_bin][i_group]=[np.nan,np.nan,np.nan]
                
            idx_bin=np.intersect1d(train_idx,np.argwhere(np.logical_and(X[1,:,-1]<=-treeFrac_bin[0],X[1,:,-1]>-treeFrac_bin[1])))
            try:
                scores_train['Def'][i_bin][i_group]=[coeff_of_agreement(y[1,idx_bin].flatten(),est.predict(X.reshape(3,-1,nr_feat)[1,idx_bin,:].reshape(-1,nr_feat))),
                                                   calculate_rmse(y[1,idx_bin].flatten(),est.predict(X.reshape(3,-1,nr_feat)[1,idx_bin,:].reshape(-1,nr_feat))),
                                                   calculate_mae(y[1,idx_bin].flatten(),est.predict(X.reshape(3,-1,nr_feat)[1,idx_bin,:].reshape(-1,nr_feat)))]
            except:
                scores_train['Def'][i_bin][i_group]=[np.nan,np.nan,np.nan]
            
            idx_bin=np.intersect1d(test_idx,np.argwhere(np.logical_and(X[0,:,-1]>=treeFrac_bin[0],X[0,:,-1]<treeFrac_bin[1])))            
            try:
                scores_test['Aff'][i_bin][i_group]=[coeff_of_agreement(y[0,idx_bin].flatten(),est.predict(X.reshape(3,-1,nr_feat)[0,idx_bin,:].reshape(-1,nr_feat))),
                                                   calculate_rmse(y[0,idx_bin].flatten(),est.predict(X.reshape(3,-1,nr_feat)[0,idx_bin,:].reshape(-1,nr_feat))),
                                                   calculate_mae(y[0,idx_bin].flatten(),est.predict(X.reshape(3,-1,nr_feat)[0,idx_bin,:].reshape(-1,nr_feat)))]
            except:
                scores_test['Aff'][i_bin][i_group]=[np.nan,np.nan,np.nan]
                
            idx_bin=np.intersect1d(train_idx,np.argwhere(np.logical_and(X[1,:,-1]<=-treeFrac_bin[0],X[1,:,-1]>-treeFrac_bin[1])))            
            try:
                scores_test['Def'][i_bin][i_group]=[coeff_of_agreement(y[1,idx_bin].flatten(),est.predict(X.reshape(3,-1,nr_feat)[1,idx_bin,:].reshape(-1,nr_feat))),
                                                   calculate_rmse(y[1,idx_bin].flatten(),est.predict(X.reshape(3,-1,nr_feat)[1,idx_bin,:].reshape(-1,nr_feat))),
                                                   calculate_mae(y[1,idx_bin].flatten(),est.predict(X.reshape(3,-1,nr_feat)[1,idx_bin,:].reshape(-1,nr_feat)))]
            except:
                scores_test['Def'][i_bin][i_group]=[np.nan,np.nan,np.nan]
                                    
    
    return [scores_train,scores_test]


def GridSearch_sel(scores,groups,lambdas,n_splines):
    
    '''
    Perform Grid Search across Blocked CV
    
    Input
    -----
    Assumes scores has train and test as two components in list and within train/test there are dictionary of values for
    Aff and def and treeFrac bins
    
    Calculates mean test score and chooses lambda and number of basis functions (n_splines) accordingly
    
    Output
    ------
    
    dictrionary of month-specific selected parameters
    
    '''
    
    avg_scores_coa=np.zeros([len(lambdas),len(n_splines)])
    avg_scores_rmse=np.zeros([len(lambdas),len(n_splines)])
    avg_scores_mae=np.zeros([len(lambdas),len(n_splines)])
    
    groups=scores[lambdas[0]][n_splines[0]][1]['Aff'][0].keys()
    for i_lam,lam in enumerate(lambdas):
        
        for i_ns,ns in enumerate(n_splines):
            
            avg_scores_coa[i_lam,i_ns]=np.nanmean(np.vstack((np.vstack(([scores[lam][ns][1]['Aff'][i_bin][i_group][0] for i_group in groups for i_bin in scores[lam][ns][1]['Aff'].keys()])),
                                                         np.vstack(([scores[lam][ns][1]['Def'][i_bin][i_group][0] for i_group in groups for i_bin in scores[lam][ns][1]['Def'].keys()])))))
    
            avg_scores_rmse[i_lam,i_ns]=np.nanmean(np.vstack((np.vstack(([scores[lam][ns][1]['Aff'][i_bin][i_group][1] for i_group in groups for i_bin in scores[lam][ns][1]['Aff'].keys()])),
                                                         np.vstack(([scores[lam][ns][1]['Def'][i_bin][i_group][1] for i_group in groups for i_bin in scores[lam][ns][1]['Def'].keys()])))))
            
            avg_scores_mae[i_lam,i_ns]=np.nanmean(np.vstack((np.vstack(([scores[lam][ns][1]['Aff'][i_bin][i_group][2] for i_group in groups for i_bin in scores[lam][ns][1]['Aff'].keys()])),
                                                         np.vstack(([scores[lam][ns][1]['Def'][i_bin][i_group][2] for i_group in groups for i_bin in scores[lam][ns][1]['Def'].keys()])))))
            
          
    idx_opt_rmse=np.unravel_index(np.argmin(avg_scores_rmse),avg_scores_rmse.shape)
    idx_opt_mae=np.unravel_index(np.argmin(avg_scores_mae),avg_scores_mae.shape)
    
    return lambdas[mode(np.array([idx_opt_rmse[0],idx_opt_mae[0]])).mode[0]],n_splines[mode(np.array([idx_opt_rmse[1],idx_opt_mae[1]])).mode[0]]


def spatial_cv_obs(mod,X,y,nr_feat,groups,n_splits,weights,plot=False):
    
    '''
    Same as spatial cv but customised for Obs
    '''
    
    
    cv=GroupKFold(n_splits=n_splits)
    
    X=X.reshape(2,-1,nr_feat)
    treeFrac_idx=np.argwhere((X[0,:,-1]<-0.01)==True)
    
    treeFrac_idx=np.intersect1d(treeFrac_idx,np.argwhere(~np.isnan(y[0,:])))
    group_kfold=cv.split(X.reshape(2,-1,nr_feat)[0,:,:],y[0,:],groups)
    
    # Create a nested list of train and test indices for each fold
    train_indices, test_indices = [list(traintest) for traintest in zip(*group_kfold)]
    group_cv = [*zip(train_indices,test_indices)]
    
    scores_train={}
    scores_test={}
    
    scores_train['Def']={0:{},1:{},2:{},3:{},4:{}}
    
    scores_test['Def']={0:{},1:{},2:{},3:{},4:{}}
        
    for train_idx,test_idx in group_cv:
        
        i_group=int(mode(groups[test_idx]).mode[0])
        
        train_idx = np.intersect1d(train_idx,treeFrac_idx)
        test_idx = np.intersect1d(test_idx,treeFrac_idx)
        est=mod.fit(X.reshape(2,-1,nr_feat)[:,train_idx,:].reshape(-1,nr_feat),y[:,train_idx].flatten(),
                    weights=np.nan_to_num(weights.reshape(2,-1)[:,train_idx].flatten(),posinf=0,neginf=0))
        
        for i_bin,treeFrac_bin in enumerate([[0.01,0.15],[0.15,0.3],[0.3,0.5],[0.5,0.8],[0.8,1]]):
            
            idx_bin=np.intersect1d(train_idx,np.argwhere(np.logical_and(X[0,:,-1]<=-treeFrac_bin[0],X[0,:,-1]>-treeFrac_bin[1])))
            try:
                scores_train['Def'][i_bin][i_group]=[coeff_of_agreement(y[0,idx_bin].flatten(),est.predict(X.reshape(2,-1,nr_feat)[0,idx_bin,:].reshape(-1,nr_feat))),
                                                   calculate_rmse(y[0,idx_bin].flatten(),est.predict(X.reshape(2,-1,nr_feat)[0,idx_bin,:].reshape(-1,nr_feat))),
                                                   calculate_mae(y[0,idx_bin].flatten(),est.predict(X.reshape(2,-1,nr_feat)[0,idx_bin,:].reshape(-1,nr_feat)))]
            except:
                scores_train['Def'][i_bin][i_group]=[np.nan,np.nan,np.nan]
            
            idx_bin=np.intersect1d(train_idx,np.argwhere(np.logical_and(X[0,:,-1]<=-treeFrac_bin[0],X[0,:,-1]>-treeFrac_bin[1])))            
            try:
                scores_test['Def'][i_bin][i_group]=[coeff_of_agreement(y[0,idx_bin].flatten(),est.predict(X.reshape(2,-1,nr_feat)[0,idx_bin,:].reshape(-1,nr_feat))),
                                                   calculate_rmse(y[0,idx_bin].flatten(),est.predict(X.reshape(2,-1,nr_feat)[0,idx_bin,:].reshape(-1,nr_feat))),
                                                   calculate_mae(y[0,idx_bin].flatten(),est.predict(X.reshape(2,-1,nr_feat)[0,idx_bin,:].reshape(-1,nr_feat)))]
            except:
                scores_test['Def'][i_bin][i_group]=[np.nan,np.nan,np.nan]

                                    
    
    return [scores_train,scores_test]


def GridSearch_sel_obs(scores,groups,lambdas,n_splines):
    
    '''
    Same as GridSearch but customised for Obs
    '''
    
    avg_scores_coa=np.zeros([len(lambdas),len(n_splines)])
    avg_scores_rmse=np.zeros([len(lambdas),len(n_splines)])
    avg_scores_mae=np.zeros([len(lambdas),len(n_splines)])
    
    
    for i_lam,lam in enumerate(lambdas):
        
        for i_ns,ns in enumerate(n_splines):
            
            
            avg_scores_coa[i_lam,i_ns]=np.nanmean(np.vstack(([scores[lam][ns][1]['Def'][i_bin][i_group][0] for i_group in groups for i_bin in scores[lam][ns][1]['Def'].keys()])))
    
            avg_scores_rmse[i_lam,i_ns]=np.nanmean(np.vstack(([scores[lam][ns][1]['Def'][i_bin][i_group][1] for i_group in groups for i_bin in scores[lam][ns][1]['Def'].keys()])))
            
            avg_scores_mae[i_lam,i_ns]=np.nanmean(np.vstack(([scores[lam][ns][1]['Def'][i_bin][i_group][2] for i_group in groups for i_bin in scores[lam][ns][1]['Def'].keys()])))
            
          
    idx_opt_rmse=np.unravel_index(np.argmin(avg_scores_rmse),avg_scores_rmse.shape)
    idx_opt_mae=np.unravel_index(np.argmin(avg_scores_mae),avg_scores_mae.shape)
    
    return lambdas[mode(np.array([idx_opt_rmse[0],idx_opt_mae[0]])).mode[0]],n_splines[mode(np.array([idx_opt_rmse[1],idx_opt_mae[1]])).mode[0]]
    
