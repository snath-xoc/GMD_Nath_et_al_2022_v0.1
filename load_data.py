"""
Loading files to train the Land Cover-Climate emulator
"""

import os
import numpy as np
import xarray as xr
import mplotutils as mpu
import copy

def configs():
    
    '''
    Outputs config files, a bit messy with nested dictionaries as a start; but due to different naming conventions across ESMs, necessary
    '''
    
    config = {}
    
    config['main_dir'] = '/net/so4/landclim/snath/data/WP1/ctl_crop_frst/'
    config['ts_file_dirs'] = {
                            'CESM2':'frst-ctl/cesm/TG/TG_frst-ctl_cesm_signal-separated_g025.nc',
                            'MPI-ESM':'frst-ctl/Amon/ts/ts_frst-ctl_mpiesm_signal-separated_g025.nc',
                            'EC-EARTH':'frst-ctl/Amon/ts/ts_frst-ctl_ecearth_signal-separated_g025.nc',
                             'Obs':'crop-ctl/Duveillier/ts/ts_mean_def_g025.nc',
                          }
    
    config['lc_map_file_dirs'] = {
                                 'CESM2':'treeFrac/cesm_FRST_CTL_forestfractions_needle_broad_g025.nc',
                                  'MPI-ESM':'treeFrac/mpiesm_FRST_CTL_forestfractions_g025.nc',
                                  'EC-EARTH':'treeFrac/ecearth_FRST_CTL_forestfractions_g025.nc',
                                  'Obs':'crop-ctl/Duveillier/treeFrac/Obs_treeFrac_g025.nc',
                                 }
    
    config['ctl_lc_map_file_dirs'] = {
                                 'CESM2':'treeFrac/cesm_CTL_forestfractions_needle_broad_g025.nc',
                                  'MPI-ESM':'treeFrac/mpiesm_CTL_forestfractions_g025.nc',
                                  'EC-EARTH':'treeFrac/mpiesm_CTL_forestfractions_g025.nc',
                                 }
    
    config['avail_vars'] = {
                             'CESM2':['ts_mean','ts_mn','ts_mx'],
                              'MPI-ESM':['ts_mean'],
                              'EC-EARTH':['ts_mean'],
                              'Obs':['ts_mean','ts_mn','ts_mx'],

                             }

    config['var_names'] = {
                       'CESM2':{'ts_mean':'TG','ts_mn':'TSMN','ts_mx':'TSMX'},
                       'MPI-ESM':{'ts_mean':'ts'},
                       'EC-EARTH':{'ts_mean':'ts'},
                       'Obs':{'ts_mean':'ts_mean','ts_mn':'ts_night','ts_mx':'ts_day'},
                      }
    return config

def load_meta_data():
    
    '''
    Return useful metadata:
    
    idx_l: land-sea mask
    wgt_l: weights of grid-points for when taking global averages
    srex_rax: srex regions
    '''
    
    frac_l = xr.open_mfdataset('/net/so4/landclim/snath/data/interim_invariant_lsmask_regrid.nc', combine='by_coords',decode_times=False)
    #land-sea mask of ERA-interim bilinearily interpolated 
    frac_l_raw = np.squeeze(copy.deepcopy(frac_l.lsm.values))
    frac_l = frac_l.where(frac_l.lat>-60,0) # remove Antarctica from frac_l field (ie set frac l to 0)
    idx_l=np.squeeze(frac_l.lsm.values)>0.0 # idex land #-> everything >0 I consider land

    lon_pc, lat_pc = mpu.infer_interval_breaks(frac_l.lon, frac_l.lat)

    srex_raw = xr.open_mfdataset('/net/so4/landclim/snath/data/srex-region-masks_20120709.srex_mask_SREX_masks_all.25deg.time-invariant.nc', combine='by_coords',decode_times=False)
    lons, lats = np.meshgrid(srex_raw.lon.values,srex_raw.lat.values)

    wgt = np.cos(np.deg2rad(lats)) # area weights of each grid point
    wgt_l = (wgt*frac_l_raw)[idx_l] # area weights for land grid points (including taking fraction land into consideration)
    
    return idx_l, wgt_l, srex_raw, lon_pc, lat_pc
    

def load_data_tas(data,model):
    
    '''
    Returns the available climate variables under both Aff and Def scenarios (Only Def for Obs)
    
    Input
    -----
    
    data: dictionary in which to store the model data
    model: model for which to get data
    '''
    
    
    config = configs()
    idx_l, _, _, _, _ = load_meta_data()
    data[model] = {}
    
    for var in config['avail_vars'][model]:
        
        if var == 'ts_mean':
            
            file_path = config['ts_file_dirs'][model]
            
        else:
            
            file_path = config['ts_file_dirs'][model].replace(config['var_names'][model]['ts_mean'],config['var_names'][model][var])
            
       
        if model == 'Obs':
            
            var_name = 'ts'
            
        else:
            
            var_name = config['var_names'][model][var]
        
        data[model][var] = [xr.open_mfdataset(config['main_dir']+file_path).roll(lon=72)['%s_local'%var_name].values[:,idx_l]]
        
        if model!= 'Obs':
            
            data[model][var].append(xr.open_mfdataset(config['main_dir']+file_path.replace('frst-ctl','crop-ctl')).roll(lon=72)['%s_local'%var_name].values[:,idx_l])
            
    return data
        
def load_data_lc(data,model):
    
    '''
    Returns the land cover map under both Aff and Def scenarios (Only Def for Obs)
    
    Input
    -----
    
    data: dictionary in which to store the model data
    model: model for which to get data
    '''
    
    
    config = configs()
    idx_l, _, _, _, _ = load_meta_data()
    data[model] = {}
    
    df = xr.open_mfdataset(config['main_dir']+config['lc_map_file_dirs'][model]).roll(lon = 72)
    
    ## get geographical data first
    
    
    
    if model=='Obs':
        lon,lat = np.meshgrid(np.linspace(-180+2.5*0.5,180-2.5*0.5,int(360/2.5)),np.linspace(-90+2.5*0.5,90-2.5*0.5,int(180/2.5)))
        data[model]['lon'] = (lon[idx_l] - 180)##different grid used for Obs
        data[model]['lat']= lat[idx_l]
    else:
        lon,lat = np.meshgrid(df.lon.values,df.lat.values)
        data[model]['lon'] = (lon[idx_l] - 180)
        data[model]['lat'] = lat[idx_l]
    
    if model != 'Obs':
        data[model]['orog']=xr.open_mfdataset('/net/so4/landclim/snath/data/WP1/metadata/orog/orog_fx_'+model+'_historical_r1i1p1f1_g025.nc').roll(lon=72).orog.values[idx_l]
        
    else:
        data[model]['orog']=xr.open_mfdataset('/net/so4/landclim/snath/data/WP1/metadata/orog/orog_fx_CESM2_historical_r1i1p1f1_g025.nc').roll(lon=72).orog.values[idx_l]

    
    ## get land cover maps
    
    if model!= 'Obs':
        
        data[model]['treeFrac'] = [np.squeeze(df.all_forest.values)[idx_l]]
        data[model]['treeFrac'].append(np.squeeze(xr.open_mfdataset(config['main_dir']+config['lc_map_file_dirs'][model].replace('FRST','CROP')).roll(lon=72).all_forest.values)[idx_l])
        
    else:
        
        data[model]['treeFrac'] = -(np.squeeze(df['treeFrac']).values[idx_l])
        
    return data

def load_hooker_coeffs():
    
    '''
    Get Hooker et al. (2018) coeffs
    '''
    
    idx_l, _, _, _, _ = load_meta_data()
    
    coeffs={}

    for reg in ['GWR','CSWR']:

        coeffs[reg]={}

        for coeff in ['b0','b1','b2']:

            coeffs[reg][coeff]=xr.open_mfdataset('/net/so4/landclim/snath/data/WP1/Hooker_coeffs/Coeffs_'+reg+'_'+coeff+'_g025.nc').roll(lon=72)[coeff].values[:,idx_l]
            
    return coeffs



    
    

        
        
        
        
        
        
    
    
