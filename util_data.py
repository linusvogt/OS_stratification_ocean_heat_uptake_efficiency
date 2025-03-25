import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from util import ModelCMIP6, iter_models


def load_var_dict(var, kwargs):
    funcs = {
        'strat': load_strat_dict,
        'strat_scalar': load_strat_regional_dict,
        'ohue': load_ohue_dict,
        'moc': load_moc_dict,
        'mld': load_mld_dict,
        'mld_scalar': load_mld_scalar_dict,
    }
    assert var in funcs.keys(), f'{var=} not recogized'

    out = funcs[var](**kwargs)

    return out


def load_strat_dict(exp, var, maxlev):
    df = pd.read_csv('data/model_members.csv')
    data_dict = {}
    for _, row in df.iterrows():
        logging.info(f'Loading N²: {row.model}')
        model_obj = ModelCMIP6(row.center, row.model, row.piControl)
        da = load_n2_regrid(model=model_obj, exp=exp, maxlev=maxlev, var=var)
        data_dict[model_obj] = da
    return data_dict


def load_strat_regional_dict(region, var, maxlev=1500):
    if 'natl' in region:
        reg_str = region.replace("-", "_")
    else:
        reg_str = region
    mask = xr.open_dataset(
        f'region_masks/mask_{reg_str}_1deg_reg_grid.nc').mask
    weights = xr.open_dataset(
        'region_masks/reg_1deg_grid_area_ocean.nc').cell_area.fillna(0.)

    out = load_strat_dict(exp='piControl', maxlev=maxlev, var=var)

    averaged = {}
    for model, da in out.items():
        da = (da * mask).weighted(weights).mean(['lon', 'lat'])
        da = xr.DataArray(float(da))
        averaged[model] = da
    out = averaged

    out = {model: da.compute() for model, da in out.items()}
    return out


def load_mld_dict():
    df = pd.read_csv('data/model_members.csv')
    data_dict = {}
    for _, row in df.iterrows():
        logging.info(f'Loading MLD: {row.model}')

        path = Path('data/mld_piControl/')
        path /= (f'mld_piControl_{row.model}_{row.piControl}_annual.nc')
        da = xr.open_dataset(path).mld.mean('year')

        model_obj = ModelCMIP6(row.center, row.model, row.piControl)
        data_dict[model_obj] = da
    return data_dict


def load_mld_scalar_dict(region):
    mask = xr.open_dataset(
        f'region_masks/mask_{region.replace("-", "_")}_1deg_reg_grid.nc').mask
    weights = xr.open_dataset(
        'region_masks/reg_1deg_grid_area_ocean.nc').cell_area.fillna(0.)

    def load_exp(exp, mean):
        out = {}
        for mm, model in iter_models(exp):
            logging.info(f'Loading MLD {exp}: {model.model}')
            try:
                da = load_mld_regrid(model=model)
                out[model] = da
            except Exception as e:
                logging.info(f'Skipping {model.model} due to: {e}')
        return out

    out = load_exp('piControl', mean=True)

    averaged = {}
    for model, da in out.items():
        da = (da * mask).weighted(weights).mean(['lon', 'lat'])
        da = xr.DataArray(float(da))
        averaged[model] = da
    out = averaged

    return out


def load_n2_regrid(model, exp, maxlev, var):
    path = Path('data/stratification_piControl/')
    path /= f'min0m_max{maxlev}m/{var}'
    var_name = 'n2' if var == 'total' else f'n2-{var}'
    path /= (f'{var_name}_min0m_max{maxlev}m_{exp}_'
             + f'{model.model}_{model.member}_annual.nc')
    da = xr.open_dataset(path).n2
    return da


def load_mld_regrid(model):
    path = Path('data/mld_piControl/')
    path /= ('mld_piControl_'
             + f'{model.model}_{model.member}_annual.nc')
    da = xr.open_dataset(path).mld
    return da


def n2_ecco_outpath(n2_var, regridded, lev_int, year=None):
    p = Path(f'data/stratification_ecco/{n2_var}')
    if lev_int:
        p /= 'lev_int'
    else:
        p /= 'full'
    reg_str = '_reg' if regridded else ''
    year_str = f'_year{str(year).zfill(2)}' if year is not None else ''
    levint_str = '_levint' if lev_int else ''
    p /= f'n2_{n2_var}_ECCO{reg_str}{levint_str}{year_str}.nc'
    return p


def load_n2_ecco(n2_var, lev_int, regridded=True, return_ds=False):
    p = n2_ecco_outpath(n2_var, regridded=regridded, lev_int=lev_int)

    if lev_int:
        ds = xr.open_dataset(p, use_cftime=True).squeeze()
    else:
        files = sorted(list(p.parent.glob('*.nc')))
        assert len(files) == 26, f'{len(files)=}'
        ds = xr.open_mfdataset(
            files, parallel=True, use_cftime=True).squeeze()

    assert len(ds.time) % 12 == 0, f'{len(ds.time)=}'

    if return_ds:
        return ds
    return ds.n2


def load_ohue_dict():
    out = {}
    for mm, model in iter_models('piControl'):
        logging.info(f'Loading OHUE: {model.model}')
        try:
            ohue = get_ohue(model.model)
            out[model] = ohue
        except Exception as e:
            logging.info(f'Skipping {model.model} due to: {e}')
    return out


def get_ohue(modelname):
    ohue_file = 'data/ohue_1pctCO2.csv'
    df = pd.read_csv(ohue_file)
    if modelname in df.model.values:
        ohue = float(df[df.model == modelname].ohue.values[0])
    else:
        raise ValueError(f'No OHUE for {modelname}')
    return ohue


def load_moc_dict(name):
    df = pd.read_csv('data/data_moc.csv')
    key = {'amoc': 'amoc', 'upper': 'm_so', 'wmt': 'm_wmt'}[name]
    data = {
        ModelCMIP6(row.institution_id, row.source_id, row.member_id):
        row[key]
        for _, row in df.iterrows()
    }
    return data


def reg_grid_mask_outpath(region):
    p = Path(f'region_masks/mask_{region}_1deg_reg_grid.nc')
    return p
