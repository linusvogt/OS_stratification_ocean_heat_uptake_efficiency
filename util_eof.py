import numpy as np
import xarray as xr
from eofs.xarray import Eof

from util_data import load_var_dict


def load_eof_input(var, season, regress_centers=False):
    if var == 'mld':
        data = load_var_dict(var='mld', kwargs=dict())
    elif var == 'strat':
        data = load_var_dict(
            var='strat', kwargs=dict(var='total', maxlev=1500, exp='piControl'))

    models = sorted(data.keys(), key=lambda m: m.model)
    return data, models


def eof_without_outliers(data, models, n_eofs, n_outliers):
    """Exlude outliers from EOF analysis by iteratively removing the model with the most extreme (positive/negative) PC"""
    eof, pcs, varfrac = apply_eof(data, models, n_eofs=n_eofs)
    outlier_models = []
    for i in range(n_outliers):
        # print(f'{i=}, {eof=}')
        outlier = find_outlier_model(pcs, models).model
        outlier_models.append(outlier)
        # print(outlier)
        models = [m for m in models if m.model != outlier]
        eof, pcs, varfrac = apply_eof(data=data, models=models, n_eofs=n_eofs)
    return eof, pcs, varfrac, models, outlier_models


def apply_eof(data, models, n_eofs, regress_centers=False):
    if regress_centers:
        centers = list(set([model.center for model in models]))
        da = xr.concat([xr.concat([data[m] for m in models if m.center == center], 'center').mean(
            'center') for center in centers], 'time').compute()
    else:
        da = xr.concat([data[m] for m in models], 'time').compute()

    da = da.transpose('time', 'lat', 'lon')

    coslat = np.cos(np.deg2rad(da.lat.values)).clip(0., 1.)
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(da, weights=wgts)

    eof = solver.eofsAsCorrelation(neofs=n_eofs).squeeze()
    #eof = solver.eofs(neofs=n_eofs).squeeze()
    varfrac = solver.varianceFraction()
    pcs = solver.pcs(pcscaling=1)

    return eof, pcs, varfrac


def find_outlier_model(pcs, models):
    max_pc = pcs.isel(mode=0).max('time')
    min_pc = pcs.isel(mode=0).min('time')

    if abs(max_pc) > abs(min_pc):
        outlier = models[int(pcs.isel(mode=0).argmax('time'))]
        #print(f'max outlier: {outlier}')
        return outlier
    outlier = models[int(pcs.isel(mode=0).argmin('time'))]
    #print(f'min outlier: {outlier}')
    return outlier
