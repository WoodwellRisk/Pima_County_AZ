import numpy as np
import rasterio
from haversine import haversine, haversine_vector, Unit
from scipy.optimize import minimize, show_options
from scipy.special import gamma
import xarray as xr
from scipy.stats import genextreme
import matplotlib.pyplot as plt
import glob2
from datetime import datetime
import seaborn as sns
import pandas as pd
import itertools
from multiprocessing import Pool


def calc_disagg_factors(na14_region):
    rps = [2, 5, 10, 25, 50, 100, 500]
    hrs = [1, 2, 3, 6, 12]

    na14_h_rp_disagg_array = []

    # loop through return periods
    for rp in rps:
        # read in daily NA14 raster for return period
        with rasterio.open(f'{na14_region}{rp}yr24ha_ams_remapbil.nc', 'r') as r:
            na14_d_rp = r.read(1)

        # loop through each subdaily amount
        for hr in hrs:
            # read in hourly NA14 raster for each subdaily amount and return period
            if hr == 1:
                with rasterio.open(f'{na14_region}{rp}yr60ma_ams_remapbil.nc', 'r') as h:
                    na14_h_rp = h.read(1)
            else:
                with rasterio.open(f'{na14_region}{rp}yr{hr:02d}ha_ams_remapbil.nc', 'r') as h:
                    na14_h_rp = h.read(1)

            # divide subdaily NA14 raster by the daily NA14 for the corresponding return period (e.g., 2hr_100yr / 24hr_100yr)
            na14_h_rp_disagg = na14_h_rp / na14_d_rp

            # stack disagg factors into 3D array
            na14_h_rp_disagg = np.expand_dims(na14_h_rp_disagg, axis=0)
            if len(na14_h_rp_disagg_array) == 0:
                na14_h_rp_disagg_array = na14_h_rp_disagg
            else:
                na14_h_rp_disagg_array = np.append(na14_h_rp_disagg_array, na14_h_rp_disagg, axis=0)

    # mask NaNs
    na14_h_rp_disagg_array = np.ma.masked_invalid(na14_h_rp_disagg_array)

    # return disaggregation grids for each return period and subdaily level
    return na14_h_rp_disagg_array


def annual_max(data_files):

    for f in data_files:
        print(f)
        out_file = f.split('/')[-1]
        out_file = './annual_max/'+out_file[:-3]+'_annual_max.nc'

        # open one of your files
        ds = xr.open_dataset(f)
        lat = ds.lat
        lon = ds.lon

        time = ds.time.dt.year
        years = np.unique(time)
        annual_max = []
        for y in years:
            # find maximum for a specific year (1990 in this example)
            ds_ymax = ds.sel(time=slice(str(y)+'-01-01', str(y)+'-12-31')).max('time')

            ds_ymax = np.array(ds_ymax.pr)

            annual_max.append(ds_ymax)

        annual_max = np.array(annual_max)

        annual_max[annual_max < 0] = 0

        dummy = ds.pr[0:len(years),:,:] # create dummy output variable
        dummy[:] = annual_max[:]

        ds['years'] = years

        ds['annual_max'] = xr.DataArray(data=dummy, dims=["years","lat", "lon"], coords=[years,lat,lon])
        vars = list(ds.keys())
        if 'time_bnds' in vars:
            data_out = ds.drop_vars(['time_bnds','pr','time'])
        else:
            data_out = ds.drop_vars(['pr','time'])
        data_out.to_netcdf(out_file)

    return

def calc_weights(point, other_points, radius):
    # create matrix of the Haversine distance in miles between all grid cells
    hav_dist = haversine_vector(point, other_points, unit=Unit.MILES, comb=True)
    # apply triweight kernel function to weights
    kern_dist = (np.power(radius,2) - np.power(hav_dist,2)) / np.power(radius,2)

    # set all negative kernel values to zero
    kern_dist[kern_dist < 0] = 0

    # normalize weights for each prediction point (grid cell) across rows; normalize rows of a matrix
    kern_dist_norm = kern_dist / kern_dist.sum()
    return kern_dist_norm


def gev_nll_array(data, x, cov):
    '''
    Log-likehood of GEV distribution for the region
    Log-likehood equation here: https://www.mas.ncl.ac.uk/~nlf8/teaching/mas8391/background/chapter2.pdf page 22
    '''
    mu0 = x[0] #location0
    mu1 = x[1] #location1
    mu = mu0 + (mu1 * cov) #location
    sigma0 = x[2] #scale0
    sigma1 = x[3] #scale1
    sigma = np.exp(sigma0 + (sigma1 * cov)) #scale
    xi = x[4] #shape

    n = len(data) # number of sample values
    m = 1 + (xi * ((data - mu) / sigma)) # define m
    if np.min(m) < 0.00001: # if minimum of m is close to zero or negative, return 1000000
        return np.full((len(cov),), 1000000.0)

    if (xi==0):
        loglik = -n*np.log(sigma) - np.sum((data-mu)/sigma) - np.sum(np.exp(-((data-mu)/sigma)))
    else:
        loglik = -n*np.log(sigma) - (1/xi+1) * np.sum(np.log(m)) - np.sum(m**(-1/xi))

    return -(loglik) # return log-likelihood


def weighted_gev_nll(x, data, weights, cov):
    '''
    beta distribution addition is taken from Martins and Stedinger (2000) https://repositorio.ufc.br/bitstream/riufc/59412/1/2000_art_esmartins3.pdf page 740
    natural log of beta is subtracted from NLL because we converted NLL to positive
    '''
    xi = x[4] #shape
    q = 5
    beta = (gamma(q+q) * (0.5 + xi)**(q-1) * (0.5 - xi)**(q-1)) / (gamma(q)*gamma(q)) # calculate beta weight for shape parameter

    nll_grid = np.apply_along_axis(gev_nll_array, 0, data, x, cov)
    weights = weights / len(cov) # each year has a weight
    return np.sum((nll_grid - np.log(beta))*weights) # multiply the log-likelihood by the weights


def regional_gmle(i, j, lat, lon, data, coords_df, max_dist, cov, qs_param):

    annual_max_data = data[:,i,j] # get annual max data for pixel
    if np.any(np.isnan(annual_max_data)):
        return [np.nan, np.nan, np.nan, np.nan, np.nan]

    else:
        # point coords
        point_coords = (lat, lon)
        # calculate weights
        weights = calc_weights(point_coords, coords_df, max_dist)

        weights_reshape = weights.reshape(data.shape[1], data.shape[2]) # reshape weights to 2D array

        loc_initial = qs_param.location.to_numpy()[i,j] #25np.mean(annual_max_data) # initialize location parameter, mu
        scale_initial = np.log(qs_param.scale.to_numpy()[i,j]) #7 # np.log(np.std(annual_max_data)) # initialize scale parameter, sigma
        shape_initial = qs_param.shape.to_numpy()[i,j] #0.05 # initialize shape parameter, xi
        x0 = [loc_initial, loc_initial, scale_initial, scale_initial, shape_initial] # initial vector of parameters
        # print(x0)

        # minimize weighted log likelihood to calculate the GEV parameters for each prediction point, same parameters are used for all log likelihood values
        mask = (weights_reshape != 0) & (~np.isnan((data[0,:,:])))# create mask of valid pixel data based on weight values and no nan pixels
        data_region = data[:,mask] # mask annual max data for valid pixels

        masked_weights = weights_reshape[mask] # select valid weights

        # minimize LL
        res = minimize(fun=weighted_gev_nll, x0=x0, args=(data_region,masked_weights,cov), method='Nelder-Mead', options={'adaptive':True,'fatol': 2e-8,'maxiter':len(x0)*300})
        loc0 = res.x[0] # get location0 parameter
        loc1 = res.x[1] # get location1 parameter
        scale0 = res.x[2] # get scale0 parameter
        scale1 = res.x[3] # get scale1 parameter
        shape = -res.x[4] # sign for shape parameter is switched in scipy.genextreme so need to apply negative

        return [loc0, loc1, scale0, scale1, shape]



def idf_curve(data, year_start, year_end, model, scenario, study_area, max_dist, cpus, erf, qs_param):
    now_time = datetime.now()

    # create pandas dataframe with index for each pixel and lat, lon
    coords_df = data.annual_max[0,:,:].to_dataframe().reset_index()[['lat', 'lon']]
    lat = coords_df.lat
    lon = coords_df.lon
    lat_len = len(data.lat)
    lon_len = len(data.lon)

    # extract annual max years
    sel_data = data.sel(years=slice(str(year_start), str(year_end)))
    # convert data to numpy array
    annual_max = sel_data.annual_max.to_numpy()

    lats_index = np.repeat(np.arange(lat_len), lon_len)
    lons_index = []
    for t in range(lat_len):
        lons_index.extend(np.arange(lon_len))

    # years array
    years = np.arange(year_start, year_end+1)
    # effective radiative forcing array
    erf = erf[erf['Year'].isin(years)]['Effective Radiative Forcing (W/m^2)'].values
    cov = erf

    input_tuple = list(zip(lats_index, lons_index, lat, lon, itertools.repeat(annual_max),itertools.repeat(coords_df),itertools.repeat(max_dist),itertools.repeat(cov),itertools.repeat(qs_param)))

    with Pool(cpus) as p:
        gev_params = p.starmap(regional_gmle, input_tuple) # get GEV parameters with regional GMLE

    gev_params = np.reshape(np.array(gev_params), (lat_len, lon_len, 5))

    lat = data.lat
    lon = data.lon
    data['location0'] = xr.DataArray(data=gev_params[:,:,0], dims=["lat", "lon"], coords=[lat,lon])
    data['location1'] = xr.DataArray(data=gev_params[:,:,1], dims=["lat", "lon"], coords=[lat,lon])
    data['scale0'] = xr.DataArray(data=gev_params[:,:,2], dims=["lat", "lon"], coords=[lat,lon])
    data['scale1'] = xr.DataArray(data=gev_params[:,:,3], dims=["lat", "lon"], coords=[lat,lon])
    data['shape'] = xr.DataArray(data=gev_params[:,:,4], dims=["lat", "lon"], coords=[lat,lon])

    data_out = data.drop_vars(['annual_max','years'])
    data_out.to_netcdf(f'./IDF_NS_parameters/{study_area}_{model}_{scenario}_IDF_parameters_{year_start}-{year_end}_{max_dist}miles.nc')

    print(datetime.now() - now_time)

    return


def delta_method(gev_params, baseline_year, future_year, na14, percentile, erf):

    idf_base = np.empty((gev_params.location0.shape[0],gev_params.location0.shape[1])) # create output for IDF baseline values
    idf_future = np.empty((gev_params.location0.shape[0],gev_params.location0.shape[1])) # create output for IDF future values
    cov_baseline = erf[erf['Year'] == int(baseline_year)]['Effective Radiative Forcing (W/m^2)'].values[0]
    cov_future = erf[erf['Year'] == int(future_year)]['Effective Radiative Forcing (W/m^2)'].values[0]

    # loop through GEV parameters
    for i in range(gev_params.location0.shape[0]):
        for j in range(gev_params.location0.shape[1]):
            if np.isnan(gev_params.shape[i,j]):
                 idf_base[i,j] = np.nan
            # calculate IDF values for all percentiles
            else:
                location_base = gev_params.location0[i,j] + (gev_params.location1[i,j] * cov_baseline)
                scale_base = np.exp(gev_params.scale0[i,j] + (gev_params.scale1[i,j] * cov_baseline))
                idf_base[i,j] = genextreme.ppf(percentile, c=gev_params.shape[i,j], loc=location_base, scale=scale_base)

                location_future = gev_params.location0[i,j] + (gev_params.location1[i,j] * cov_future)
                scale_future = np.exp(gev_params.scale0[i,j] + (gev_params.scale1[i,j] * cov_future))
                idf_future[i,j] = genextreme.ppf(percentile, c=gev_params.shape[i,j], loc=location_future, scale=scale_future)

    bias = (na14.pfe.squeeze() * 25.4) / idf_base # calculate bias between NA14 and baseline climate model period
    idf_future_corrected = idf_future * bias # correct future climate model amount with bias

    return idf_future_corrected / 25.4


# integrate disagg factors into delta function

##############################

if __name__ == "__main__":

    na14_region = 'sw'
    scenario = 'ssp585'
    year_start = 1971
    year_end = 2100
    radius = 40 # set radius (miles) for each prediction point
    baseline_year = 1989.5
    future_year = 2039.5
    cpus = 31
    study_area = 'EPC'

    daily_pr_files = glob2.glob(f'./pr/{study_area}*_{scenario}*.nc')
    # annual_max(daily_pr_files)

    erf = pd.read_csv("./rcmip-radiative-forcing-annual-means-v5-1-0.csv")
    erf = erf[(erf['Scenario'] == scenario) & (erf['Variable'] == 'Effective Radiative Forcing|Anthropogenic')]
    erf.drop(axis=1, labels=['Model','Scenario','Region','Mip_Era','Unit','Activity_Id','Variable'], inplace=True)
    erf = erf.T
    erf.reset_index(inplace=True)
    erf.rename(columns={'index':'Year',404:'Effective Radiative Forcing (W/m^2)'},inplace=True)
    erf['Year'] = erf['Year'].astype(int)

    annual_max_files = glob2.glob(f'./annual_max/{study_area}*{scenario}*.nc')
    # annual_max_files = glob2.glob(f'./annual_max/{study_area}_pr_M*_{scenario}*.nc')
    count = 1
    for f in annual_max_files:
        print(f"Progress: {count} of {len(annual_max_files)} files")
        data = xr.open_dataset(f, engine='netcdf4')
        model = f.split('/')[-1].split('_')[2]
        # if model == "MPI-ESM1-2-HR":
        qs_param_file = f'./IDF_QS_parameters/EPC_{model}_{scenario}_IDF_QS_parameters_1975-2004_40miles.nc'
        qs_param = xr.open_dataset(qs_param_file, engine='netcdf4')
        idf_curve(data, year_start, year_end, model, scenario, study_area, radius, cpus, erf, qs_param)
        count = count + 1


    ### create loop for subdaily and daily
    rps = [2, 5, 10, 25, 50, 100, 500] # return periods
    percentiles = [(1 - (1/i)) for i in rps] # percentiles for return periods
    hrs = [1, 2, 3, 6, 12]

    idf_param_files = sorted(glob2.glob(f'./IDF_NS_parameters/{study_area}*{scenario}*{year_start}-{year_end}_{radius}miles.nc'))
    out_pr = []
    models = []
    for f in idf_param_files:
        print(f)
        m = f.split('_')[3]
        print(m)
        params = xr.open_dataset(f, engine='netcdf4')
        na14 = xr.open_dataset(f'./na14/{study_area}_sw100yr24ha_ams.nc', engine='netcdf4')
        idf_corr_future = delta_method(params, baseline_year, future_year, na14, 0.99, erf)
        out_pr.append(idf_corr_future)
        models.append(m)

    na14 = xr.open_dataset(f'./na14/{study_area}_sw100yr24ha_ams.nc', engine='netcdf4')
    na14['pr_mm'] = xr.DataArray(data=out_pr, dims=["models","lat", "lon"], coords=[np.arange(1,len(models)+1),na14.lat,na14.lon])
    na14['models'] = models

    data_out = na14.drop_vars(['pfe','time'])
    data_out.to_netcdf(f'{study_area}_NS_{scenario}_100yr_24hr_future_{radius}miles.nc')
