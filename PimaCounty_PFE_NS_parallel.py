
import numpy as np
from haversine import haversine_vector, Unit #, haversine
from scipy.optimize import minimize, basinhopping, shgo, direct, differential_evolution, dual_annealing, fsolve #, show_options
import xarray as xr
from scipy.stats import genextreme
import glob2
from datetime import datetime
import pandas as pd
import itertools
from multiprocessing import Pool
from os.path import exists
import matplotlib.pyplot as plt
from numba import jit
from sklearn.linear_model import LinearRegression, QuantileRegressor
import lmoments3 as lm
from lmoments3 import distr
from math import gamma
import scipy


def apply_disagg_factors(idf_future_daily, na14_daily, na14_subdaily):
    '''
    Takes in the daily and subdaily NA14 rasters and computes the future subdaily by calculating the disaggregation
    factors between the NA14 daily and subdaily.
    '''

    # divide subdaily NA14 raster by the daily NA14 for the corresponding return period (e.g., 2hr_100yr / 24hr_100yr)
    na14_subdaily_disagg = na14_subdaily.pfe / na14_daily.pfe

    # mask NaNs
    na14_subdaily_disagg = np.ma.masked_invalid(na14_subdaily_disagg)

    # apply disaggregation factors to adjusted future IDF values to get subdaily values
    idf_corr_future_subdaily = idf_future_daily * na14_subdaily_disagg

    # return subdaily adjusted future IDF values
    return idf_corr_future_subdaily


def annual_max(data_files):
    '''
    Calculate the annual maxima from daily rainfall climate data.
    '''

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
    '''
    Function to calculate the triweight kernel distance function between points or grid cells. The distances are converted 
    to weights for the weighted log-likelihood minimization.
    '''
    
    # create matrix of the Haversine distance in miles between all grid cells
    hav_dist = haversine_vector(point, other_points, unit=Unit.MILES, comb=True)
    # apply triweight kernel function to weights
    kern_dist = (np.power(radius,2) - np.power(hav_dist,2)) / np.power(radius,2)

    # set all negative kernel values to zero
    kern_dist[kern_dist < 0] = 0

    # normalize weights for each prediction point (grid cell) across rows; normalize rows of a matrix
    kern_dist_norm = kern_dist / kern_dist.sum()
    return kern_dist_norm

@jit(nopython=True)
def gev_nll_array(data, x, cov):
    '''
    Log-likehood of GEV distribution for the region. Data is the annual max, x is the parameter vector, and cov
    is the temporal covariate.
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
        return 1000000.0

    if (xi==0):
        loglik = -n*np.log(sigma) - np.sum((data-mu)/sigma) - np.sum(np.exp(-((data-mu)/sigma)))
    else:
        loglik = -n*np.log(sigma) - (1/xi+1) * np.sum(np.log(m)) - np.sum(m**(-1/xi))

    return -np.sum(loglik) # return log-likelihood

@jit(nopython=True)
def weighted_gev_nll(x, data, weights, cov):
    '''
    Function to calculate the weighted Negative Log-Likelihood for the GEV distribution. A beta distribution weight is applied to constrict the shape parameter.
    The beta distribution addition is taken from Martins and Stedinger (2000) https://repositorio.ufc.br/bitstream/riufc/59412/1/2000_art_esmartins3.pdf page 740.
    The natural log of beta is subtracted from NLL because we converted NLL to positive.
    '''
    xi = x[4] #shape
    q = 5
    beta = (gamma(q+q) * (0.5 + xi)**(q-1) * (0.5 - xi)**(q-1)) / (gamma(q)*gamma(q)) # calculate beta weight for shape parameter

    nll_grid = np.zeros((data.shape[1]))
    for i in range(data.shape[1]):
        nll_grid[i] = gev_nll_array(data[:,i], x, cov)

    # print(np.sum(nll_grid * weights) - np.log(beta), x[0],x[1],x[2],x[3],x[4])
    return np.sum(nll_grid * weights) - np.log(beta)  # multiply the log-likelihood by the weights


def regional_gmle(i, j, lat, lon, data, coords_df, max_dist, cov, qs_param, round):
    '''
    Estimates the parameters for the GEV distribution of a pixel using a weighted negative log-likelihood approach. 
    Two rounds of fitting are completed to ensure the global minima is found.
    '''

    annual_max_data = data[:,i,j] # get annual max data for pixel
    if np.any(np.isnan(annual_max_data)):
        return [np.nan, np.nan, np.nan, np.nan, np.nan]

    else:
        now_time = datetime.now()
        # point coords
        point_coords = (lat, lon)
        # calculate weights
        weights = calc_weights(point_coords, coords_df, max_dist)

        weights_reshape = weights.reshape(data.shape[1], data.shape[2]) # reshape weights to 2D array
        mask = (weights_reshape != 0) & (~np.isnan((data[0,:,:])))# create mask of valid pixel data based on weight values and no nan pixels
        data_region = data[:,mask] # mask annual max data for valid pixels
        masked_weights = weights_reshape[mask] # select valid weights

        if round == 2:
            loc0_initial = qs_param[0] # initialize location parameter, mu
            loc1_initial = qs_param[1]
            scale0_initial = qs_param[2] # initialize scale parameter, sigma
            scale1_initial = qs_param[3]
            shape_initial = qs_param[4] # initialize shape parameter, xi
            x0 = [loc0_initial, loc1_initial, scale0_initial, scale1_initial, -shape_initial]

        else:
            loc0_initial = qs_param.location.to_numpy()[i,j] # initialize location parameter, mu
            scale0_initial = np.log(qs_param.scale.to_numpy()[i,j]) # initialize scale parameter, sigma
            shape_initial = qs_param.shape.to_numpy()[i,j] # initialize shape parameter, xi

            x0 = [loc0_initial, 0, scale0_initial, 0, -shape_initial]

        res = minimize(fun=weighted_gev_nll, x0=x0, args=(data_region,masked_weights,cov), method='Nelder-Mead', options={'adaptive':True, 'maxiter':1000, 'fatol':1e-7}) #
        # print(res)
        loc0 = res.x[0] # get location0 parameter
        loc1 = res.x[1] # get location1 parameter
        scale0 = res.x[2] # get scale0 parameter
        scale1 = res.x[3] # get scale1 parameter
        shape = -res.x[4] # sign for shape parameter is switched in scipy.genextreme so need to apply negative

        return [loc0, loc1, scale0, scale1, shape]


def anomoly(grid, radius):
    '''
    Function to figure out which pixels need to be re-optimized in the minimization.
    '''
    mean = scipy.ndimage.generic_filter(grid, np.mean, size=radius) #calculate mean within a radius
    std = scipy.ndimage.generic_filter(grid, np.std, size=radius) #calculate standard deviation within a radius
    dif = np.abs((grid - mean)) > (std * 0.5) ## determine which pixels fall outside half the standard deviation plus the mean

    return dif


def idf_curve(data, year_start, year_end, model, scenario, study_area, max_dist, cpus, cov, qs_param):
    '''
    Calculates the GEV parameters of the baseline and future time periods of a particular climate model and scenario.
    '''
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

    input_tuple = list(zip(lats_index, lons_index, lat, lon, itertools.repeat(annual_max),itertools.repeat(coords_df),itertools.repeat(max_dist),itertools.repeat(cov),itertools.repeat(qs_param),itertools.repeat(1)))

    with Pool(cpus) as p:
        gev_params = p.starmap(regional_gmle, input_tuple) # get GEV parameters with regional GMLE

    gev_params = np.reshape(np.array(gev_params), (lat_len, lon_len, 5))

    print(f"First round done {datetime.now() - now_time}")

    location0 = gev_params[:,:,0]
    location1 = gev_params[:,:,1]
    scale0 = gev_params[:,:,2]
    scale1 = gev_params[:,:,3]
    shape = gev_params[:,:,4]

    radius = 3
    dif_loc0 = anomoly(location0, radius)
    dif_loc1 = anomoly(location1, radius)
    dif_scale0 = anomoly(scale0, radius)
    dif_scale1 = anomoly(scale1, radius)
    dif_shape = anomoly(shape, radius)

    lats_index, lons_index = np.where((dif_loc0 == 1) | (dif_loc1 == 1) | (dif_scale0 == 1) | (dif_scale1 == 1) | (dif_shape == 1))

    if len(lats_index) > 0:

        lat = np.unique(lat)[lats_index]
        lon = np.unique(lon)[lons_index]

        average_loc0 = scipy.ndimage.generic_filter(location0, np.mean, size=radius)[lats_index, lons_index]
        average_loc1 = scipy.ndimage.generic_filter(location1, np.mean, size=radius)[lats_index, lons_index]
        average_scale0 = scipy.ndimage.generic_filter(scale0, np.mean, size=radius)[lats_index, lons_index]
        average_scale1 = scipy.ndimage.generic_filter(scale1, np.mean, size=radius)[lats_index, lons_index]
        average_shape = scipy.ndimage.generic_filter(shape, np.mean, size=radius)[lats_index, lons_index]
        average_params = list(zip(average_loc0, average_loc1, average_scale0, average_scale1, average_shape))

        input_tuple = list(zip(lats_index, lons_index, lat, lon, itertools.repeat(annual_max),itertools.repeat(coords_df),itertools.repeat(max_dist),itertools.repeat(cov),average_params,itertools.repeat(2)))
        print(f"Number of pixels to redo: {len(input_tuple)}")

        with Pool(cpus) as p:
            gev_params_new = p.starmap(regional_gmle, input_tuple) # get GEV parameters with regional GMLE

        for k in range(len(gev_params_new)):
            gev_params[lats_index[k], lons_index[k], :] = gev_params_new[k]

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


def delta_method(gev_params, baseline_year, future_year, rp, na14_daily, erf):
    '''
    Bias-adjust the projected future daily extreme precipitation using the NA14 data as the reference dataset and the quantile delta method.
    '''

    idf_base = np.empty((gev_params.location0.shape[0],gev_params.location0.shape[1])) # create output for IDF baseline values
    idf_future = np.empty((gev_params.location0.shape[0],gev_params.location0.shape[1])) # create output for IDF future values
    cov_baseline = erf[erf['Year'] == int(baseline_year)]['Effective Radiative Forcing (W/m^2)'].values[0]
    cov_future = erf[erf['Year'] == int(future_year)]['Effective Radiative Forcing (W/m^2)'].values[0]
    percentile = (1 - (1 / rp)) # percentile for return period

    # loop through GEV parameters
    for i in range(gev_params.location0.shape[0]):
        for j in range(gev_params.location0.shape[1]):
            if np.isnan(gev_params.shape[i,j]):
                 idf_base[i,j] = np.nan
            else:
                # calculate PFE values for all percentiles
                location_base = gev_params.location0[i,j] + (gev_params.location1[i,j] * cov_baseline)
                scale_base = np.exp(gev_params.scale0[i,j] + (gev_params.scale1[i,j] * cov_baseline))
                idf_base[i,j] = genextreme.ppf(percentile, c=gev_params.shape[i,j], loc=location_base, scale=scale_base)

                location_future = gev_params.location0[i,j] + (gev_params.location1[i,j] * cov_future)
                scale_future = np.exp(gev_params.scale0[i,j] + (gev_params.scale1[i,j] * cov_future))
                idf_future[i,j] = genextreme.ppf(percentile, c=gev_params.shape[i,j], loc=location_future, scale=scale_future)

    # calculate bias between daily NA14 and baseline climate model period, correct future climate model amount w/bias
    bias = (na14_daily.pfe.squeeze() * 25.4) / idf_base
    bias_adjusted_idf_future = (idf_future * bias) / 25.4

    return bias_adjusted_idf_future


##############################

if __name__ == "__main__":

    na14_region = 'sw'
    scenario = 'ssp245'
    year_start = 1971
    year_end = 2100
    radius = 40 # set radius (miles) for each prediction point
    baseline_year = 1989.5
    future_year = 2039.5
    cpus = 31
    study_area = 'EPC'

#    daily_pr_files = glob2.glob(f'./pr/{study_area}*_{scenario}*.nc')
#    annual_max(daily_pr_files)

    erf = pd.read_csv("./rcmip-radiative-forcing-annual-means-v5-1-0.csv")
    erf = erf[(erf['Scenario'] == scenario) & (erf['Variable'] == 'Effective Radiative Forcing|Anthropogenic')]
    erf.drop(axis=1, labels=['Model','Scenario','Region','Mip_Era','Unit','Activity_Id','Variable'], inplace=True)
    erf = erf.T
    erf.reset_index(inplace=True)
    if scenario == 'ssp245':
        erf.rename(columns={'index':'Year',308:'Effective Radiative Forcing (W/m^2)'},inplace=True)
    if scenario == 'ssp585':
        erf.rename(columns={'index':'Year',404:'Effective Radiative Forcing (W/m^2)'},inplace=True)
    erf['Year'] = erf['Year'].astype(int)
    # years array
    years = np.arange(year_start, year_end+1)
    # effective radiative forcing array
    cov = erf[erf['Year'].isin(years)]['Effective Radiative Forcing (W/m^2)'].values

    annual_max_files = glob2.glob(f'./annual_max/{study_area}*{scenario}*.nc')

    count = 1
    for f in annual_max_files:
        print(f"Progress: {count} of {len(annual_max_files)} files")
        data = xr.open_dataset(f, engine='netcdf4')
        model = f.split('/')[-1].split('_')[2]
        if model == 'CNRM-CM6-1-HR':
            count = count + 1
            continue
        qs_param_file = f'./IDF_QS_parameters/{study_area}_{model}_{scenario}_IDF_QS_parameters_1975-2004_{radius}miles.nc'
        qs_param = xr.open_dataset(qs_param_file, engine='netcdf4')
        idf_curve(data, year_start, year_end, model, scenario, study_area, radius, cpus, cov, qs_param)
        count = count + 1


    # establish return periods and time durations of interest
    rps = [2, 5, 10, 25, 50, 100, 500] # return periods
    # rps = [100]
    hrs = [1, 2, 3, 6, 12] # sub-daily time durations (daily is automatically included)

    # loop thru return periods and time durations, calculate PFEs
    for rp in rps:
        print(rp)
        # open daily NA14 file for specified return period
        na14_daily = xr.open_dataset(f'./na14/{study_area}_{na14_region}{rp}yr24ha_ams.nc', engine='netcdf4')

        # open existing (or else calculate) daily PFE values for return period
        if exists(f'{study_area}_NS_{scenario}_{rp}yr_24hr_future_{radius}miles.nc'):
            print(f'Daily file for {rp}-yr return period already exists. Loading file...')
            data_out = xr.open_dataset(f'{study_area}_NS_{scenario}_{rp}yr_24hr_future_{radius}miles.nc', engine='netcdf4')
            out_pr = data_out.pr_in
            models = data_out.models

            data_out.close()

        # loop thru NS parameter files (models) for specific rp
        else:
            print(f'Daily file for {rp}-yr return period does not exist. Calculating daily IDF values now...')
            # get list of NS parameter files
            idf_param_files = sorted(glob2.glob(f'./IDF_NS_parameters/{study_area}_*_{scenario}_*_{year_start}-{year_end}_{radius}miles.nc'))
            out_pr = []
            models = []

            for f in idf_param_files:
                print(f)
                m = f.split('_')[3]
                params = xr.open_dataset(f, engine='netcdf4')

                # calculate daily PFEs and correct for NA14 bias for return period
                idf_corr_future = delta_method(params, baseline_year, future_year, rp, na14_daily, erf)
                out_pr.append(idf_corr_future)
                models.append(m)

                params.close()

            # save daily output for return period
            na14_daily['pr_in'] = xr.DataArray(data=out_pr, dims=["models","lat", "lon"], coords=[np.arange(1,len(models)+1), na14_daily.lat, na14_daily.lon])
            na14_daily['models'] = models
            data_out = na14_daily.drop_vars(['pfe','time'])
            data_out.to_netcdf(f'./IDF_NS_results/{study_area}_NS_{scenario}_{rp}yr_24hr_future_{radius}miles.nc')

            data_out.close()

        for hr in hrs:
            # calculate subdaily disaggregation factors using NA14
            # apply disagg factors to daily bias-adjusted idf values to get sub-daily values for specific rp and hr
            if hr != 1:
                na14_subdaily = xr.open_dataset(f'./na14/{study_area}_{na14_region}{rp}yr{hr:02d}ha_ams.nc', engine='netcdf4')
            else:
                na14_subdaily = xr.open_dataset(f'./na14/{study_area}_{na14_region}{rp}yr60ma_ams.nc', engine='netcdf4')

            out_pr_subdaily = apply_disagg_factors(out_pr, na14_daily, na14_subdaily)

            # save subdaily output for return period
            na14_subdaily['pr_in'] = xr.DataArray(data=out_pr_subdaily, dims=["models","lat", "lon"], coords=[np.arange(1,len(models)+1), na14_subdaily.lat, na14_subdaily.lon])
            na14_subdaily['models'] = models
            data_out = na14_subdaily.drop_vars(['pfe','time'])
            data_out.to_netcdf(f'./IDF_NS_results/{study_area}_NS_{scenario}_{rp}yr_{hr}hr_future_{radius}miles.nc')

            na14_subdaily.close()
            data_out.close()

        na14_daily.close()

    print('done')
