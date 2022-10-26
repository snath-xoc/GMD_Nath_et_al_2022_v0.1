'''
K-means clustering of background climates
'''

from sklearn.cluster import KMeans
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from load_data import configs
from load_data import load_meta_data

def load_clim_data():
    
    '''
    
    Load background climate data for K-means clustering
    
    '''
    
    config = configs()
    idx_l, _, _, _, _ = load_meta_data()
    
    data_ts_ctl={}
    data_rh_ctl={}

    X_cluster={}

    for model in ['Obs','CESM2','MPI-ESM','EC-EARTH']:
        if model=='CESM2':
            data_ts_ctl[model]=xr.open_mfdataset(config['main_dir']+'crop-ctl/cesm/TG/TG_ctl_g025.nc').roll(lon=72).TG.values[:,idx_l]
            data_rh_ctl[model]=xr.open_mfdataset(config['main_dir']+'crop-ctl/cesm/RH2M/RH2M_ctl_cesm_g025.nc').roll(lon=72).RH2M.values[:,idx_l]
        elif model=='MPI-ESM':
            data_ts_ctl[model]=xr.open_mfdataset(config['main_dir']+'crop-ctl/Amon/ts/ts_ctl_g025.nc').roll(lon=72).ts.values[:,idx_l]
            data_rh_ctl[model]=xr.open_mfdataset(config['main_dir']+'crop-ctl/Amon/hurs/hurs_ctl_mpiesm_150-years_g025.nc').roll(lon=72).hurs.values[:,idx_l]

        elif model=='EC-EARTH':
            data_ts_ctl[model]=xr.open_mfdataset(config['main_dir']+'crop-ctl/Amon/ts/ts_ctl_ecearth_g025.nc').roll(lon=72).ts.values[:12,idx_l]
            data_rh_ctl[model]=xr.open_mfdataset(config['main_dir']+'crop-ctl/Amon/hurs/hurs_ctl_ecearth_g025.nc').roll(lon=72).hurs.values[:12,idx_l]

        elif model=='Obs':

            data_ts_ctl[model]=np.zeros([12,idx_l.sum()])
            data_rh_ctl[model]=np.zeros([12,idx_l.sum()])

            for i_mon in range(12):
                i_mon_lab=str(i_mon+1)
                if len(i_mon_lab)==1:
                    i_mon_lab='0'+i_mon_lab
                data_ts_ctl[model][i_mon,:]=xr.open_mfdataset('/net/so4/landclim/snath/data/WP1/ts_to_t2m/wc2.1_10m_tavg_%s_g025.nc'%i_mon_lab).roll(lon=72).Band1.values[idx_l]
                data_rh_ctl[model][i_mon,:]=xr.open_mfdataset('/net/so4/landclim/snath/data/WP1/ts_to_t2m/wc2.1_10m_vapr_%s_g025.nc'%i_mon_lab).roll(lon=72).Band1.values[idx_l]

    return data_ts_ctl, data_rh_ctl

def elbow_method(model):
    
    '''
    
    Elbow method selection of number of clusters
    
    '''
    
    data_ts_ctl, data_rh_ctl = load_clim_data()
    idx_l, _, _, _, _ = load_meta_data()
    
    X_cluster=np.hstack((np.nanmean(data_ts_ctl[model].reshape(-1,12,idx_l.sum()),axis=0).T,
                                np.nanmean(data_rh_ctl[model].reshape(-1,12,idx_l.sum()),axis=0).T,
                               ))
    
    X_cluster[np.isnan(X_cluster)]=0
    
    Sum_of_squared_distances = []
    K = range(1,12)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X_cluster)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def cluster_final(labels_final,model,k):
    
    '''
    
    Output final climate clusters
    
    Input
    -----
    
    labels: dictionary in which to store label outputted by the estimators
    
    '''
    data_ts_ctl, data_rh_ctl = load_clim_data()
    idx_l, _, _, _, _ = load_meta_data()
    
    X_cluster=np.hstack((np.nanmean(data_ts_ctl[model].reshape(-1,12,idx_l.sum()),axis=0).T,
                                np.nanmean(data_rh_ctl[model].reshape(-1,12,idx_l.sum()),axis=0).T,
                               ))
    
    X_cluster[np.isnan(X_cluster[model])]=0
    
    est=KMeans(n_clusters=k)
    est.fit(X_cluster)
    labels = est.labels_
    
    reg_dict={'NA':[1,3,4,5],'SA':[6,7,8,9,10],'AF':[14,15,16,17],
          'EURAS':[11,12,13,18,19,20,22,23],'AUS':[25,26],'CGI':[2],
         'TIB':[21],'SEA':[24]}
    
    ## Load meta SREX data to then extract geographical regions to be combined with climate regions
    
    srex_raw = xr.open_mfdataset('/net/so4/landclim/snath/data/srex-region-masks_20120709.srex_mask_SREX_masks_all.25deg.time-invariant.nc', combine='by_coords',decode_times=False)
    srex_vals=srex_raw.srex_mask.values
    
    labels_final[model]=np.zeros([idx_l.sum()])
    
    for reg, i in zip(reg_dict.keys(),(np.arange(1,len(reg_dict)+1)*10)):
        
        reg_mask = np.isin(srex_vals[idx_l],reg_dict[reg])
        
        labels_reg=np.unique(labels[reg_mask].astype(int))
        
        
        
        for label_reg in labels_reg:
            idx_reg_lab=\
            np.intersect1d(np.argwhere(reg_mask==True).flatten(),np.argwhere(labels.astype(int)==label_reg).flatten())
            
            labels_final[model][idx_reg_lab]=i
            
            
            
            i+=1
    
    return labels_final
    
    
