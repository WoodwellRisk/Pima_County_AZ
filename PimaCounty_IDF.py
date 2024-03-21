import numpy as np
import rasterio
from haversine import haversine, Unit


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





def gev_nll(data):
    # https://www.mas.ncl.ac.uk/~nlf8/teaching/mas8391/background/chapter2.pdf page 22
    loc = 0 # initialize location parameter, mu
    scale = 0 # initialize scale parameter, sigma
    shape = 0 # initialize shape parameter, xi
    n = len(data) # number of sample values
    mean = np.mean(data) # mean of sample
    std = np.std(data) # standard deviation of sample
    par1 = n*np.log(std) # first part of equation
    par2 = (1 + (1/shape)) * np.sum(np.log(1 + shape * ((data - loc) / scale))) # second part of equation
    par3 = np.sum(np.power(1 + shape * ((data - loc) / scale),(-1/shape))) # third part of equation

    return - par1 - par2 - par3 # return negative log-likelihood


def calc_weights(radius):
    # take in station coordinates
    # create matrix of the Haversine distance in miles between all stations
    haversine(pp, other_stations, unit=Unit.MILES)

    # apply triweight kernel function to weights
    dist_matrix = (np.power(radius,2) - np.power(dist_matrix,2)) / np.power(radius,2)

    # set all negative kernel values to zero
    dist_matrix[dist_matrix < 0] = 0

    # normalize weights for each prediction point
    x_norm = x/np.sum(x)

    return

def regional_gmle():
    # set radius (miles) for each prediction point
    max_dist = 10

    # calculate weights distance between each station in miles based on radius
    calc_weights(max_dist)

    # maximize weighted negative log likelihood to calculate the GEV parameters for each prediction point, same parameters are used for all log likelihood values



    return



##############################
