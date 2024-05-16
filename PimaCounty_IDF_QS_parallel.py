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
from multiprocessing import Pool
import itertools



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


def gev_nll(x, data):
    '''
    Log-likehood of GEV distribution
    '''
    mu = x[0] #location
    sigma = np.exp(x[1]) #scale
    xi = x[2] #shape
    # Log-likehood equation here: https://www.mas.ncl.ac.uk/~nlf8/teaching/mas8391/background/chapter2.pdf page 22
    n = len(data) # number of sample values
    m = 1 + (xi * (data - mu) / sigma) # define m
    if np.min(m) < 0.00001: # if minimum of m is close to zero or negative, return 1000000
        return 1000000

    if (xi==0):
        loglik = -n*np.log(sigma) - np.sum((data-mu)/sigma) - np.sum(np.exp(-((data-mu)/sigma)))
    else:
        loglik = -n*np.log(sigma) - (1/xi+1) * np.sum(np.log(m)) - np.sum(m**(-1/xi))

    return -(loglik) # return log-likelihood

def gev_nll_array(data, x):
    '''
    Log-likehood of GEV distribution for the region
    '''
    mu = x[0] #location
    sigma = np.exp(x[1]) #scale
    xi = x[2] #shape
    # Log-likehood equation here: https://www.mas.ncl.ac.uk/~nlf8/teaching/mas8391/background/chapter2.pdf page 22
    n = len(data) # number of sample values
    m = 1 + (xi * ((data - mu) / sigma)) # define m
    if np.min(m) < 0.00001: # if minimum of m is close to zero or negative, return 1000000
        return 1000000.0

    if (xi==0):
        loglik = -n*np.log(sigma) - np.sum((data-mu)/sigma) - np.sum(np.exp(-((data-mu)/sigma)))
    else:
        loglik = -n*np.log(sigma) - (1/xi+1) * np.sum(np.log(m)) - np.sum(m**(-1/xi))

    return -(loglik) # return log-likelihood


def weighted_gev_nll(x, data, weights):
    '''
    beta distribution addition is taken from Martins and Stedinger (2000) https://repositorio.ufc.br/bitstream/riufc/59412/1/2000_art_esmartins3.pdf page 740
    natural log of beta is subtracted from NLL because we converted NLL to positive
    '''
    xi = x[2] #shape
    q = 5
    beta = (gamma(q+q) * (0.5 + xi)**(q-1) * (0.5 - xi)**(q-1)) / (gamma(q)*gamma(q)) # calculate beta weight for shape parameter

    nll_grid = np.apply_along_axis(gev_nll_array, 0, data, x) # multiply the log-likelihood by the weights
    return np.sum((nll_grid - np.log(beta) )*weights)


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


def regional_gmle(i, j, lat, lon, data, coords_df, max_dist):

            annual_max_data = data[:,i,j] # get annual max data for pixel
            if np.any(np.isnan(annual_max_data)):
                return [np.nan, np.nan, np.nan]

            else:
                # point coords
                point_coords = (lat, lon)
                # calculate weights
                weights = calc_weights(point_coords, coords_df, max_dist)

                weights_reshape = weights.reshape(data.shape[1], data.shape[2]) # reshape weights to 2D array

                loc_initial = np.mean(annual_max_data) # initialize location parameter, mu
                scale_initial = np.log(np.std(annual_max_data)) # initialize scale parameter, sigma
                shape_initial = 0.1 # initialize shape parameter, xi
                x0 = [loc_initial, scale_initial, shape_initial] # initial vector of parameters

                # minimize weighted log likelihood to calculate the GEV parameters for each prediction point, same parameters are used for all log likelihood values

                mask = (weights_reshape != 0) & (~np.isnan((data[0,:,:])))# create mask of valid pixel data based on weight values and no nan pixels
                data_region = data[:,mask] # mask annual max data for valid pixels

                masked_weights = weights_reshape[mask] # select valid weights
                # minimize LL
                res = minimize(fun=weighted_gev_nll, x0=x0, args=(data_region,masked_weights), method='Nelder-Mead')
                loc = res.x[0] # get location parameter
                scale = np.exp(res.x[1]) # get scale parameter
                shape = -res.x[2] # sign for shape parameter is switched in scipy.genextreme so need to apply negative
                # print(i, j, lat, lon, loc)
                # save parameters to output raster
                return [loc, scale, shape]


def idf_curve(data, baseline_start, baseline_end, future_start, future_end, model, scenario, study_area, max_dist, cpus):
    now_time = datetime.now()
    # create pandas dataframe with index for each pixel and lat, lon
    coords_df = data.annual_max[0,:,:].to_dataframe().reset_index()[['lat', 'lon']]
    lat = coords_df.lat
    lon = coords_df.lon
    lat_len = len(data.lat)
    lon_len = len(data.lon)

    # extract annual max years
    baseline_data = data.sel(years=slice(str(baseline_start), str(baseline_end)))
    # convert data to numpy array
    annual_max = baseline_data.annual_max.to_numpy()

    lats_index = np.repeat(np.arange(lat_len), lon_len)
    lons_index = []
    for t in range(lat_len):
        lons_index.extend(np.arange(lon_len))

    input_tuple = list(zip(lats_index, lons_index, lat, lon, itertools.repeat(annual_max),itertools.repeat(coords_df),itertools.repeat(max_dist)))

    with Pool(cpus) as p:
        gev_params_baseline = p.starmap(regional_gmle, input_tuple) # get GEV parameters with regional GMLE

    gev_params_baseline = np.reshape(np.array(gev_params_baseline), (lat_len, lon_len, 3))

    data['location'] = xr.DataArray(data=gev_params_baseline[:,:,0], dims=["lat", "lon"], coords=[data.lat,data.lon])
    data['scale'] = xr.DataArray(data=gev_params_baseline[:,:,1], dims=["lat", "lon"], coords=[data.lat,data.lon])
    data['shape'] = xr.DataArray(data=gev_params_baseline[:,:,2], dims=["lat", "lon"], coords=[data.lat,data.lon])

    data_out = data.drop_vars(['annual_max','years'])
    data_out.to_netcdf(f'./IDF_QS_parameters/{study_area}_{model}_{scenario}_IDF_QS_parameters_{baseline_start}-{baseline_end}_{max_dist}miles.nc')

    # extract annual max years
    future_data = data.sel(years=slice(str(future_start), str(future_end)))
    # convert data to numpy array
    annual_max = future_data.annual_max.to_numpy()
    input_tuple = list(zip(lats_index, lons_index, lat, lon, itertools.repeat(annual_max),itertools.repeat(coords_df),itertools.repeat(max_dist)))
    with Pool(cpus) as p:
        gev_params_future = p.starmap(regional_gmle, input_tuple) # get GEV parameters with regional GMLE

    gev_params_future = np.reshape(gev_params_future, (lat_len, lon_len, 3))
    data_out = data.drop_vars(['location','scale','shape'])
    data_out['location'] = xr.DataArray(data=gev_params_future[:,:,0], dims=["lat", "lon"], coords=[data.lat,data.lon])
    data_out['scale'] = xr.DataArray(data=gev_params_future[:,:,1], dims=["lat", "lon"], coords=[data.lat,data.lon])
    data_out['shape'] = xr.DataArray(data=gev_params_future[:,:,2], dims=["lat", "lon"], coords=[data.lat,data.lon])

    data_out = data_out.drop_vars(['annual_max','years'])
    data_out.to_netcdf(f'./IDF_QS_parameters/{study_area}_{model}_{scenario}_IDF_QS_parameters_{future_start}-{future_end}_{max_dist}miles.nc')

    print(datetime.now() - now_time)

    return


def delta_method(gev_params_baseline, gev_params_future, na14, percentile):

    idf_baseline = np.empty((gev_params_baseline.location.shape[0],gev_params_baseline.location.shape[1])) # create output for IDF baseline values
    idf_future = np.empty((gev_params_future.location.shape[0],gev_params_future.location.shape[1])) # create output for IDF future values
    # loop through GEV parameters

    for i in range(gev_params_baseline.location.shape[0]):
        for j in range(gev_params_baseline.location.shape[1]):
            if np.isnan(gev_params_baseline.shape[i,j]):
                 idf_baseline[i,j] = np.nan
                 idf_future[i,j] = np.nan
            # calculate IDF values for all percentiles
            idf_baseline[i,j] = genextreme.ppf(percentile, c=gev_params_baseline.shape[i,j], loc=gev_params_baseline.location[i,j], scale=gev_params_baseline.scale[i,j])
            idf_future[i,j] = genextreme.ppf(percentile, c=gev_params_future.shape[i,j], loc=gev_params_future.location[i,j], scale=gev_params_future.scale[i,j])

    bias = (na14.pfe.squeeze() * 25.4) / idf_baseline # calculate bias between NA14 and baseline climate model period
    return (idf_future * bias) / 25.4 # correct future climate model amount with bias

def annual_max(data_files):

    for f in data_files:
        print(f)
        out_file = f.split('/')[-1]
        out_file = './annual_max/'+out_file[:-3]+'_annual_max.nc'

        # open one of your files
        ds = xr.open_dataset(f)
        lat = ds.lat
        lon = ds.lon

        years = np.unique(ds.time.dt.year)
        annual_max = []
        for y in years:
            # find maximum for a specific year (1990 in this example)
            ds_ymax = ds.sel(time=slice(str(y)+'-01-01', str(y)+'-12-31')).max('time')
            ds_ymax = np.array(ds_ymax.pr)
            annual_max.append(ds_ymax)

        annual_max = np.array(annual_max)
        annual_max[annual_max < 0] = 0

        ds['years'] = years
        ds['annual_max'] = xr.DataArray(data=annual_max, dims=["years","lat", "lon"], coords=[years,lat,lon])
        vars = list(ds.keys())
        if 'time_bnds' in vars:
            data_out = ds.drop(labels=['time_bnds','pr','time'])
        else:
            data_out = ds.drop(labels=['pr','time'])
        data_out.to_netcdf(out_file)

    return


# quantile delta method (proportion method) vs delta method vs percentile method
# quasistationary or nonstationary method?

# integrate disagg factors into delta function

##############################

if __name__ == "__main__":

    daily_pr_files = glob2.glob('./daily_pr/*.nc')
    # annual_max(daily_pr_files)

    na14_region = 'sw'
    scenario = 'ssp245'
    baseline_start = 1975
    baseline_end = 2004
    future_start = 2025
    future_end = 2054
    radius = 40 # set radius (miles) for each prediction point
    cpus = 31
    study_area = 'EPC'

    # annual_max_files = glob2.glob('./annual_max/*.nc')
    annual_max_files = glob2.glob(f'./annual_max/{study_area}_pr_M*_{scenario}*.nc')
    count = 1
    for f in annual_max_files:
        print(f"Progress: {count} of {len(annual_max_files)} files")
        data = xr.open_dataset(f, engine='netcdf4')
        study_area = f.split('/')[-1].split('_')[0]
        model = f.split('/')[-1].split('_')[2]
        idf_curve(data, baseline_start, baseline_end, future_start, future_end, model, scenario, study_area, radius, cpus)
        count = count + 1

    print(stop)
    ### create loop for subdaily and daily
    rps = [2, 5, 10, 25, 50, 100, 500] # return periods
    percentiles = [(1 - (1/i)) for i in rps] # percentiles for return periods
    hrs = [1, 2, 3, 6, 12]

    idf_param_baseline_files = sorted(glob2.glob(f'./IDF_QS_parameters/*{baseline_start}-{baseline_end}_{radius}miles.nc'))
    idf_param_future_files = sorted(glob2.glob(f'./IDF_QS_parameters/*{future_start}-{future_end}_{radius}miles.nc'))
    out_pr = []
    models = []
    for b, f in zip(idf_param_baseline_files, idf_param_future_files):
        print(b, f)
        m = b.split('_')[3]
        baseline_params = xr.open_dataset(b, engine='netcdf4')
        future_params = xr.open_dataset(f, engine='netcdf4')
        na14 = xr.open_dataset('./na14/EPC/EPC_sw100yr24ha_ams.nc', engine='netcdf4')
        idf_corr_future = delta_method(baseline_params, future_params, na14, 0.99)
        out_pr.append(idf_corr_future)
        models.append(m)

    na14 = xr.open_dataset('./na14/EPC/EPC_sw100yr24ha_ams.nc', engine='netcdf4')
    na14['pr_in'] = xr.DataArray(data=out_pr, dims=["models","lat", "lon"], coords=[np.arange(1,len(models)+1),na14.lat,na14.lon])
    na14['models'] = models

    data_out = na14.drop_vars(['pfe','time'])
    data_out.to_netcdf(f'{study_area}_QS_{scenario}_100yr_24hr_future_{radius}miles.nc')
