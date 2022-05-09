
# bickley_jet 
This example simulates the evolution of an unstable, geostrophically balanced, Bickley jet. The initial conditions superpose the Bickley jet with small-amplitude perturbations.See "The nonlinear evolution of barotropically unstable jets," J. Phys. Oceanogr. (2003) for more details on this problem. This example also uses the WENO5 advection scheme with smoothness coefficients that are optimized with respect to an arbitrary loss function $G$. 

## Data Dependencies
- NOAA-CIRES 20th Century Reanalysis V2c (https://www.esrl.noaa.gov/psd/data/gridded/data.20thC_ReanV2c.html)
	- uwnd
	- vwnd
	- gph 
- NOAA ERSST (https://www.esrl.noaa.gov/psd/data/gridded/data.noaa.ersst.v5.html)
    - sst 
- USGS Gage Data for Hermann, Louisville, and Vicksburg 
	- top ten dates at each location 

## Packages
- env_methods: package with methods for handling data 
- vis_methods: package with methods for visualizing data 

## Scripts

## Notebooks
- gph_wind_anomaly.ipynb: Jupyter Notebook that generates anomaly and mean maps for GPH and wind vectors 
- sst_anomaly.ipynb: Jupyter Notebook that generates anomaly maps for SST around heavy flooding events 
- ind_event.ipynb: Jupyter Notebook that generates anomaly and composite maps for single flood events 
- geo_corr.ipynb: Jupyter Notebook that computes Pearson correlation in env. variables to single specified point 

## Python Dependencies
- matplotlib 
- netCDF4
- cartopy 
- datetime 
- numpy 
- pandas 
- copy
