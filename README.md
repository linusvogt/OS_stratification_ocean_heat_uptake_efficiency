# OS_stratification_ocean_heat_uptake_efficiency
Processed data and code to reproduce the main figures in the paper "Stratification and overturning circulation are intertwined controls on ocean heat uptake efficiency in climate models" (Vogt et al. 2025, Ocean Science, https://doi.org/10.5194/egusphere-2024-3442)

The file `plot_figures.py` contains the main functions to plot Figures 1-5 of the manuscript, which are saved in the `figures/` directory.
The `util*.py` files contain utility functions used to load data, perform processing, and plot.
The `data/` and `region_masks/` directories contain processed CMIP model output, ECCO state estimate data, and region masks as netCDF files.
