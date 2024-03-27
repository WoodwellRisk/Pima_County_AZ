import numpy as np
import rasterio
from haversine import haversine, haversine_vector, Unit
from scipy.optimize import minimize, show_options
from scipy.special import gamma
import xarray as xr
from scipy.stats import genextreme
import matplotlib.pyplot as plt
import glob2


def EPC_grid():

    # create grid for Eastern Pima County (EPC) based on downscaled Daymet grid

    # reprojet NA14 rasters to EPC grid

    return

def calc_disagg_factors():
    rps = [2, 5, 10, 25, 50, 100, 500]
    hrs = [1, 2, 3, 6, 12]

    # loop through return periods
    for i in rps:
        # read in daily NA14 raster for return period
        na14_d_rp = rasterio.open()

        # loop through each subdaily amount
        for j in hrs:
            # read in hourly NA14 raster for each subdaily amount and return period
            na14_h_rp = rasterio.open()
            # divide subdaily NA14 raster by the daily NA14 for the corresponding return period (e.g., 2hr_100yr / 24hr_100yr)
            na14_h_rp_disagg = na14_h_rp / na14_d_rp

    # return disaggregation grids for each return period and subdaily level
    return na14_h_rp_disagg





def gev_nll(data, x):
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
        loglik = -n*np.log(sigma) - (1/xi+1) * np.sum(np.log(m)) - np.sum(m**(-1/xi)) # third part of equation

    return -(loglik) # return log-likelihood

def weighted_gev_nll(x, data, weights):
    xi = x[2] #shape
    q = 5
    beta = (gamma(q+q) * (0.5 + xi)**(q-1) * (0.5 - xi)**(q-1)) / (gamma(q)*gamma(q)) # calculate beta weight for shape parameter
    print(np.log(beta))
    nll_grid = np.apply_along_axis(gev_nll, 0, data, x) # multiply the log-likelihood by the weights
    return np.sum(nll_grid*weights) - beta

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

def regional_gmle(data):
    # set radius (miles) for each prediction point
    max_dist = 10

    # create pandas dataframe with index for each pixel and lat, lon
    coords_df = data.annual_max[0,:,:].to_dataframe().reset_index()[['lat', 'lon']]

    # calculate weights distance between each grid cell in miles based on radius
    pixel_weights = calc_weights(coords_df, max_dist)

    # go across rows to match up with dataframe of coordinates
    data = data.annual_max.to_numpy()

    for i in range(data.shape[1]): # go through lats
        for j in range(data.shape[2]): # go through lons
            weights = pixel_weights[i+j,:] # get weights for pixel
            weights = weights.reshape(data.shape[1], data.shape[2]) # reshape weights to 2D array
            loc_initial = np.mean(data[:,i,j]) # initialize location parameter, mu
            scale_initial = np.std(data[:,i,j]) # initialize scale parameter, sigma
            shape_initial = 0.1 # initialize shape parameter, xi
            x0 = [loc_initial, scale_initial, shape_initial] # initial vector of parameters
            # minimize weighted log likelihood to calculate the GEV parameters for each prediction point, same parameters are used for all log likelihood values
            res = minimize(fun=weighted_gev_nll, x0=x0, args=(data,weights), method='Nelder-Mead')
            print(res)
            print(genextreme.ppf(0.99, c=res.x[2], loc=res.x[0], scale=res.x[1]))

            shape, loc, scale = genextreme.fit(data[:,i,j])
            print(loc, scale, shape)
            print(genextreme.ppf(0.99, c=shape, loc=loc, scale=scale))
            print(poop)



    # x = np.linspace(genextreme.ppf(0.01, c=2.420e-01, loc=1.843e-18, scale=2.456e-18), genextreme.ppf(0.99, c=2.420e-01, loc=1.843e-18, scale=2.456e-18), 100)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(x, genextreme.pdf(x, c=2.420e-01, loc=1.843e-18, scale=2.456e-18), 'r-', lw=5, alpha=0.6, label='genextreme pdf')

    # plt.show()
    return


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


# use quantile delta method (how does this compare to delta method)
# quasistationary or nonstationary method?

##############################

# annual_max()

data = xr.open_dataset('clipped_annual_max.nc', engine='netcdf4')
# print(data)

regional_gmle(data)
