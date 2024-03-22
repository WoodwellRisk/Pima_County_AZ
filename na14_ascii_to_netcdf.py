from datetime import datetime
from netCDF4 import date2num
import numpy as np
import codecs
from wwrtools import netcdf_tools

var = 'pfe'
var_long = 'precipitation frequency estimates'
units = 'in'

rps = [2, 5, 10, 25, 50, 100, 500]
hrs = [1, 2, 3, 6, 12, 24]

header_row_count = 6

for rp in rps:
    for hr in hrs:
        if hr == 1:
            na14_file = f'sw{rp}yr60ma_ams'            
        else:
            na14_file = f'sw{rp}yr{hr:02d}ha_ams'
            
        with codecs.open(f'C:/Users/kgassert/Downloads/{na14_file}/{na14_file}.asc', encoding='utf-8-sig') as f:
            header = np.genfromtxt(f, max_rows=header_row_count)
            ncols, nrows = int(header[0, 1]), int(header[1, 1])
            xllcorner, yllcorner = header[2, 1], header[3, 1]
            cellsize = header[4, 1]
            missing_value = int(header[5, 1])
            data = np.genfromtxt(f, missing_values=missing_value, usemask=True)
            data = data / 1000  # na14 original units are "inches * 1000"
            
            # generate lats and lons
            latmin = yllcorner + (cellsize / 2)
            lonmin = xllcorner + (cellsize / 2)
            lats = np.flip(np.linspace(latmin, latmin + (cellsize * nrows), nrows, endpoint=False))  # invert lats
            lons = np.linspace(lonmin, lonmin + (cellsize * ncols), ncols, endpoint=False)
            
            # make up a timestamp for saving to netcdf
            time = [datetime(2010, 12, 31)]
            time_units = 'days since 1850-01-01'
            calendar = 'proleptic_gregorian'
            time = date2num(time, units=time_units, calendar=calendar)
        
            # save netcdf
            netcdf_tools.saveLatLonDataToNetCDF(data, var, var_long, units, f'{na14_file}.nc', var_long, \
                                               lats, lons, time, time_units, calendar, np.nan, compress_bool=True)
                
                