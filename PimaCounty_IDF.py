


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











##############################
