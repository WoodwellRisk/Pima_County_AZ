import glob
from cdo import Cdo
cdo = Cdo()
cdo.debug = True

location = 'EPC'
na14_region = 'sw'

# specify name of file with desired grid, to be used to remap na14 data
grid_file = f'{location}_pr_CanESM5_ssp585_1971-2100_ds_bil_daymet_then_ba.nc'

# remap na14 netcdfs to resolution of grid_file
for s in glob.glob(f'{na14_region}*.nc'):
	print(f'Remapping {s}')
	cdo.remapbil(grid_file, input=s, output=f'{location}_{s}', options='-z zip')

print('done')
i
