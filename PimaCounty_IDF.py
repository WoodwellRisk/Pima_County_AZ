import numpy as np
import rasterio
from haversine import haversine, haversine_vector, Unit
from scipy.optimize import minimize, show_options
from scipy.special import gamma
import xarray as xr
from scipy.stats import genextreme
import matplotlib.pyplot as plt
import glob2


def calc_disagg_factors():
    rps = [2, 5, 10, 25, 50, 100, 500]
    hrs = [1, 2, 3, 6, 12]

    na14_h_rp_disagg_array = []

    # loop through return periods
    for rp in rps:
        # read in daily NA14 raster for return period
        with rasterio.open(f'sw{rp}yr24ha_ams_remapbil.nc', 'r') as r:
            na14_d_rp = r.read(1)

        # loop through each subdaily amount
        for hr in hrs:
            # read in hourly NA14 raster for each subdaily amount and return period
            if hr == 1:
                with rasterio.open(f'sw{rp}yr60ma_ams_remapbil.nc', 'r') as h:
                    na14_h_rp = h.read(1)
            else:
                with rasterio.open(f'sw{rp}yr{hr:02d}ha_ams_remapbil.nc', 'r') as h:
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
    sigma = x[1] #scale
    xi = x[2] #shape
    # Log-likehood equation here: https://www.mas.ncl.ac.uk/~nlf8/teaching/mas8391/background/chapter2.pdf page 22
    n = len(data) # number of sample values
    m = 1 + (xi * (data - mu) / sigma) # define m
    if np.min(m) < 0.00001: # if minimum of m is close to zero or negative, return 1000000
        return 1000000

    if sigma < 0.00001: # if scale is close to zero or negative, return 1000000
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
    sigma = x[1] #scale
    xi = x[2] #shape
    # Log-likehood equation here: https://www.mas.ncl.ac.uk/~nlf8/teaching/mas8391/background/chapter2.pdf page 22
    n = len(data) # number of sample values
    m = 1 + (xi * (data - mu) / sigma) # define m
    if np.min(m) < 0.00001: # if minimum of m is close to zero or negative, return 1000000
        return 1000000

    if sigma < 0.00001: # if scale is close to zero or negative, return 1000000
       return 1000000

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

def calc_weights(data, radius):
    # take in grid cell coordinates and convert to list of tuples
    coords = list(zip(data['lat'], data['lon']))

    # create matrix of the Haversine distance in miles between all grid cells
    hav_dist = haversine_vector(coords, coords, unit=Unit.MILES, comb=True)

    # apply triweight kernel function to weights
    kern_dist = (np.power(radius,2) - np.power(hav_dist,2)) / np.power(radius,2)

    # set all negative kernel values to zero
    kern_dist[kern_dist < 0] = 0

    # normalize weights for each prediction point (grid cell) across rows; normalize rows of a matrix
    row_sums = kern_dist.sum(axis=1)
    kern_dist_norm = kern_dist / row_sums[:, np.newaxis]

    return kern_dist_norm



def regional_gmle(data, start_year, end_year):
    # set radius (miles) for each prediction point
    max_dist = 10

    # create pandas dataframe with index for each pixel and lat, lon
    coords_df = data.annual_max[0,:,:].to_dataframe().reset_index()[['lat', 'lon']]
    lat_len = len(data.lat)
    lon_len = len(data.lon)

    # calculate weights distance between each grid cell in miles based on radius
    pixel_weights = calc_weights(coords_df, max_dist)
    # reshape pixel weights into 3D array
    weights_array = pixel_weights.reshape((len(pixel_weights),lat_len,lon_len))

    # extract annual max years
    data = data.sel(years=slice(str(start_year), str(end_year)))
    # convert data to numpy array
    data = data.annual_max.to_numpy()

    gev_params = np.empty((3,lat_len,lon_len))
    for i in range(lat_len): # go through lats
        for j in range(lon_len): # go through lons
            weights_pixel = pixel_weights[i+j,:] # get weights for pixel
            annual_max_data = data[:,i,j] # get annual max data for pixel

            weights_array = weights_pixel.reshape(data.shape[1], data.shape[2]) # reshape weights to 2D array
            loc_initial = np.mean(annual_max_data) # initialize location parameter, mu
            scale_initial = np.std(annual_max_data) # initialize scale parameter, sigma
            shape_initial = 0.1 # initialize shape parameter, xi
            x0 = [loc_initial, scale_initial, shape_initial] # initial vector of parameters
            # minimize weighted log likelihood to calculate the GEV parameters for each prediction point, same parameters are used for all log likelihood values

            mask = (weights_array != 0) # create mask of valid pixel data based on weight values
            data_region = data[:,mask] # mask annual max data for valid pixels
            weights = weights_array[mask] # select valid weights

            # minimize LL
            res = minimize(fun=weighted_gev_nll, x0=x0, args=(data_region,weights), method='Nelder-Mead')
            loc = res.x[0] # get location parameter
            scale = res.x[1] # get scale parameter
            shape = -res.x[2] # sign for shape parameter is switched in scipy.genextreme so need to apply negative

            # save parameters to output raster
            gev_params[:,i,j] = (loc, scale, shape)
            # print(res)
            # print(genextreme.ppf(0.99, c=shape, loc=loc, scale=scale))

            # res = minimize(fun=gev_nll, x0=x0, args=annual_max_data, method='Nelder-Mead')
            # print(res)
            # print(genextreme.ppf(0.99, c=-res.x[2], loc=res.x[0], scale=res.x[1]))

            # shape, loc, scale = genextreme.fit(annual_max_data, loc=loc_initial, scale=scale_initial)
            # print(loc, scale, shape)
            # print(genextreme.ppf(0.99, c=shape, loc=loc, scale=scale))

    return gev_params


def annual_max():
    data_files = glob2.glob('./*clipped.nc')
    #data_files = glob2.glob('*ba.nc')

    for f in data_files:
        print(f)
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
            data_out = ds.drop(labels=['time_bnds','pr','time'])
        else:
            data_out = ds.drop(labels=['pr','time'])
        data_out.to_netcdf(f[:-3]+'_annual_max.nc')

    return


def idf_curve():
    # annual_max()

    rps = [2, 5, 10, 25, 50, 100, 500] # return periods
    percentiles = [(1 - (1/i)) for i in rps] # percentiles for return periods

    data = xr.open_dataset('clipped_annual_max.nc', engine='netcdf4')

    baseline_start = 1975
    baseline_end = 2004
    future_start = 2025
    future_end = 2054
    gev_params = regional_gmle(data, baseline_start, baseline_end) # get GEV parameters with regional GMLE

    idf = np.empty((len(percentiles),gev_params.shape[1],gev_params.shape[2])) # create output for IDF values
    # loop through GEV parameters
    for i in range(gev_params.shape[1]):
        for j in range(gev_params.shape[2]):
            # calculate IDF values for all percentiles
            idf[:,i,j] = genextreme.ppf(percentiles, c=gev_params[2,i,j], loc=gev_params[0,i,j], scale=gev_params[1,i,j])

    print(idf)


    return


def delta_method():


    return


# use quantile delta method (how does this compare to delta method)
# quasistationary or nonstationary method?

##############################


idf_curve()
