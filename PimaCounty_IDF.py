import numpy as np
import rasterio
from haversine import haversine, haversine_vector, Unit
from scipy.optimize import minimize, show_options
import xarray as xr
from scipy.stats import genextreme
import matplotlib.pyplot as plt



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
        loglik = -n*np.log(sigma) - (1/xi+1) * np.sum(np.log(m)) - np.sum(m**(-1/xi)) # third part of equation

    return -(loglik) # return log-likelihood


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
    coords_df = data.pr[0,:,:].to_dataframe().reset_index()[['lat', 'lon']]

    # calculate weights distance between each grid cell in miles based on radius
    pixel_weights = calc_weights(coords_df, max_dist)

    # go across rows to match up with dataframe of coordinates

    test = data.pr[:,0,0].to_numpy()
    test = np.sort(test)[-50:]

    # minimize weighted log likelihood to calculate the GEV parameters for each prediction point, same parameters are used for all log likelihood values
    loc_initial = np.mean(test) # initialize location parameter, mu
    scale_initial = np.std(test) # initialize scale parameter, sigma
    shape_initial = 0.1 # initialize shape parameter, xi
    x0 = [loc_initial, scale_initial, shape_initial] # initial vector of parameters
    res = minimize(fun=gev_nll, x0=x0, args=test, method='Nelder-Mead')
    print(res)

    shape, loc, scale = genextreme.fit(test)
    print(loc, scale, shape)

    # x = np.linspace(genextreme.ppf(0.01, c=2.420e-01, loc=1.843e-18, scale=2.456e-18), genextreme.ppf(0.99, c=2.420e-01, loc=1.843e-18, scale=2.456e-18), 100)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(x, genextreme.pdf(x, c=2.420e-01, loc=1.843e-18, scale=2.456e-18), 'r-', lw=5, alpha=0.6, label='genextreme pdf')

    # plt.show()
    return


# constrain shape parameter with beta distribution
# use quantile delta method (how does this compare to delta method)
# quasistationary or nonstationary method?

##############################

data = xr.open_dataset('clipped.nc', engine='netcdf4')
# print(data)

regional_gmle(data)
