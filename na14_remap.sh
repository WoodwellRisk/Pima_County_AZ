# remap na14 netcdfs to resolution of community precipitation data
for s in sw*.nc; do cdo -z zip -remapbil,PimaCo_pr_CanESM5_ssp245_1971-2100_ds_bil_daymet_then_ba.nc "$s" "$s"_remapbil.nc; done &

# rename remapped files
mmv "sw*.nc_remapbil.nc" "PimaCo_sw#1_remapbil.nc"

