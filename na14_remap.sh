# specify name of file with desired grid, to be used to remap na14 data
grid_file=PimaCo_pr_CanESM5_ssp585_1971-2100_ds_bil_daymet_then_ba.nc

# remap na14 netcdfs to resolution of grid_file
for s in sw*.nc; do cdo -z zip -remapbil,"$grid_file" "$s" "$s"_remapbil.nc; done &

# rename remapped files
mmv "sw*.nc_remapbil.nc" "sw#1_remapbil.nc"

