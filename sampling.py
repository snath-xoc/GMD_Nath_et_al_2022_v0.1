'''
Employing parametric bootstrap to explory unseen tree cover scenarios with the Land cover- climate emulator
'''



def get_uncertainty_space(est,df,df_lc,i_grid,treeFrac,var='ts_local'):
    
    '''
    
    Explore uncertainty space
    
    Input
    -----
    
    est: calibrated GAM
    df: ESM training data for temperatures
    df_ly: ESM land cover maps for training simulations
    i_grid: grid point to sample at
    treeFrac: tree cover change to sample at
    
    Output
    ------
    
    200 samples, sampled from 10 bootstraps
    
    '''
    idx_l, _, _, _, _ = load_meta_data()
    uncertainty_space_gp=np.zeros([200,12])
    
    for i_mon in [0,6]:## done only for Jan and Jul but can be changed
        
        y = np.nan_to_num(np.vstack((np.nanmean(df[var][0].reshape(-1,12,idx_l.sum())[:,i_mon,:],axis=0),
                           np.nanmean(df[var][1].reshape(-1,12,idx_l.sum())[:,i_mon,:],axis=0),
                           np.zeros([1,idx_l.sum()]))),posinf=0,neginf=0)

        X = np.hstack(([create_training_set(df[var],df_lc[key],key,i_mon)for key in df_lc.keys() if key in ['lon','lat','orog','treeFrac']]))

        weights=np.where(np.sqrt((y.flatten()-np.nanmean(y.flatten()))**2).reshape(-1,1)>1.5,0.6,1)
        weights[np.squeeze(np.argwhere(np.sqrt((y.flatten()-np.nanmean(y.flatten()))**2)>2))]=0.3
        
        #print(np.array([treeFrac])[:,None])
        X=np.nan_to_num(X,posinf=0,neginf=0) 
        y=np.nan_to_num(y,posinf=0,neginf=0) 
        
        uncertainty_space_gp[:,i_mon]=np.squeeze(est[var][i_mon].sample(X,y,sample_at_X=np.hstack((np.array([df_lc['lon'][i_grid]])[:,None],
                                                                               np.array([df_lc['lat'][i_grid]])[:,None],
                                                                               np.array([df_lc['orog'][i_grid]])[:,None],
                                                                                         np.array([treeFrac])[:,None])).reshape(-1,4),n_draws=200, n_bootstraps=10))

            
    return uncertainty_space_gp