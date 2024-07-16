# Pima_County_AZ

We present code for calculating updates to future precipitation using two approaches, referred to as quasistationary (QS) and nonstationary (NS) methodologies. Our analysis is based on the methodology laid out in [NOAA’s Analysis of Impact of Nonstationary Climate on NOAA Atlas 14 Estimates](https://hdsc.nws.noaa.gov/pfds/files25/NA14_Assessment_report_202201v1.pdf), which updates current (stationary) NOAA Atlas 14 Precipitation Frequency Estimates (PFEs) for nonstationary climate conditions. In our QS approach, precipitation estimates are calculated for two different time periods, the baseline (1975–2004) and future (2025–2054), in which each period is treated as stationary. In our NS approach, precipitation estimates are calculated for the entire time period (i.e., 1971–2100) using a temporal parameter (in this case, radiative forcing) to represent changes in extreme precipitation through time.<br>

PimaCounty_PFE_QS_parallel.py: Calculates future PFEs using the quasistationary approach.<br>
PimaCounty_PFE_NS_parallel.py: Calculates future PFEs using the nonstationary approach.
