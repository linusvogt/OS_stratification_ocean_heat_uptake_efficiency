import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import proplot as pplt
import xarray as xr


def iter_models(exp, reverse=False):
    """Iterate over models used for analysis, with member_id for `exp`.

    Yields index and model, like enumerate.

    If `exp` is an iterable of experiments, yield (index and) model for each
    experiment in `exp`.
    """
    df = pd.read_csv('data/model_members.csv')

    rows_iterator = list(df.iterrows())
    if reverse:
        rows_iterator = rows_iterator[::-1]

    for rr, row in rows_iterator:
        if exp is None:
            member = None
            model = ModelCMIP6(row.center, row.model, member)
            yield rr, model
        elif type(exp) == str:
            exp_key = 'ssp' if 'ssp' in exp else exp
            assert exp_key in df.columns, f'{exp=} member not in csv file'
            member = row[exp_key]
            model = ModelCMIP6(row.center, row.model, member)
            yield rr, model
        else:  # multiple experiments
            models = []
            for experiment in exp:
                exp_key = 'ssp' if 'ssp' in experiment else experiment
                assert exp_key in df.columns, \
                    f'{experiment=} member not in csv file'
                member = row[exp_key]
                model = ModelCMIP6(row.center, row.model, member)
                models.append(model)
            yield rr, models


def load_regrid_area(ocean_only=False, rename_xy=False, r360x180=False):
    """Load area file for regular 1-degree grid."""
    if r360x180:
        assert not ocean_only
        ds = xr.load_dataset('/home/lvogt/gridarea_360x180.nc')
    else:
        oc_str = '_ocean' if ocean_only else ''
        ds = xr.load_dataset(f'/home/lvogt/reg_1deg_grid_area{oc_str}.nc')
    if rename_xy:
        try:
            ds = ds.rename_dims(lon='x', lat='y')
        except KeyError:
            pass
    area = ds.cell_area.squeeze()
    return area


@dataclass(frozen=True)
class ModelCMIP6:
    """Represents a CMIP6 model by its institution, name, and member ID."""

    center: str
    model: str
    member: str

    def __eq__(self, other):
        if isinstance(other, ModelCMIP6):
            return self.__key() == other.__key()
        return NotImplemented

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __gt__(self, other):
        return (other.center, other.model, other.member) > (
            other.center, other.model, other.member)

    def __key(self):
        return (self.center, self.model, self.member)

    def __hash__(self):
        return hash(self.__key())

    def __repr__(self):
        r = f'<ModelCMIP6: {self.center} | {self.model} | {self.member}>'
        return r

    def fname_repr(self) -> str:
        return f'{self.center}_{self.model}_{self.member}'


def nice_title(msg, for_logfile=True):
    """Transform a string into a nice title to be used for a logging file.

    TODO: wrap to 80 characters
    """
    import time
    from pathlib import Path

    import __main__
    try:
        script_name_str = (f'Logfile for {Path(__main__.__file__).name}\n'
                           if for_logfile else '')
    except Exception:
        script_name_str = ''
    ct = str(time.ctime()) + '\n'
    underline = '-' * max([len(msg), len(ct), len(script_name_str)]) + '\n'
    msg = (underline
           + script_name_str
           + ct
           + underline
           + msg + '\n'
           + underline)
    if for_logfile:
        msg = '\n' + msg
    return msg


def log_setup(name: str, level='info', reverse=False):
    p = Path('.')
    if '/' in name:
        spl = name.split('/')
        p /= '/'.join(spl[:-1])
        name = spl[-1]
    rev_str = '_r' if reverse else ''
    filename = p / f'LOG_{name}{rev_str}.log'
    level_obj = {'info': logging.INFO, 'debug': logging.DEBUG}[level]
    logging.basicConfig(
        filename=filename, filemode='w',
        level=level_obj,
        format='> %(message)s',
        force=True)
    logging.info(nice_title('Start.'))
    print(f'logfile created: {str(filename)}', flush=True)


def fix_xylims(ax, xvals, yvals, xfactor=0.2, yfactor=0.2):
    """Increase the x/y limit spacing by a factor of xfactor/yfactor"""

    xmin = np.nanmin(xvals)
    xmax = np.nanmax(xvals)
    xdelta = xmax - xmin
    ax.format(xlim=(xmin-(xfactor/2)*xdelta, xmax+(xfactor/2)*xdelta))

    ymin = np.nanmin(yvals)
    ymax = np.nanmax(yvals)
    ydelta = ymax - ymin
    ax.format(ylim=(ymin-(yfactor/2)*ydelta, ymax+(yfactor/2)*ydelta))


def plot_linear_regression(
        ax, xvals, yvals, text=True, textloc='upper left', color='k',
        invert_xy=False, mark_outliers=True,
        label=None, pval_lim=0.05):
    """Calculate and plot linear regression onto scatter plot.

    Returns:
        reg (scipy regression object)
        ln (line plot object)
    """
    from scipy.stats import linregress

    xvals, yvals = np.array(xvals), np.array(yvals)

    # remove nans
    x_bad, y_bad = np.isnan(xvals), np.isnan(yvals)
    xvals = xvals[~x_bad & ~y_bad]
    yvals = yvals[~x_bad & ~y_bad]

    assert len(xvals) == len(yvals), f'{len(xvals)=}, {len(yvals)=}'

    if invert_xy:
        xvals, yvals = yvals, xvals
    sort = np.argsort(xvals)
    xvals, yvals = xvals[sort], yvals[sort]

    reg = linregress(xvals, yvals)

    pval = reg.pvalue
    if invert_xy:
        linear_xvals = np.linspace(np.min(yvals), np.max(yvals), 100)
        linear_yvals = (linear_xvals - reg.intercept) / reg.slope
    else:
        linear_xvals = np.linspace(np.min(xvals), np.max(xvals), 100)
        linear_yvals = reg.intercept + reg.slope * linear_xvals

    linestyle = '--' if pval >= pval_lim else '-'
    ln = ax.plot(
        linear_xvals, linear_yvals, linestyle=linestyle, color=color,
        label=label)

    if text:
        if type(textloc) == str:
            x, ha = (0.05, 'left') if 'left' in textloc else (0.95, 'right')
            y, va = (0.95, 'top') if 'upper' in textloc else (0.05, 'bottom')
        else:
            x, y = textloc
            ha, va = 'left', 'top'
        r_str = f'r={reg.rvalue:.2f}'
        if pval_lim == 0.05:
            pval_str = f'p<{pval_lim:.2f}' if pval < pval_lim else f'p={pval:.2f}'
        else:
            pval_str = f'p<{pval_lim:.3f}' if pval < pval_lim else f'p={pval:.3f}'
        ax.text(s=f'{r_str}\n{pval_str}',
                x=x, y=y, transform=ax.transAxes,
                ha=ha, va=va, color=color)
    return reg, ln


def corrmap(dict_x, dict_y, regress_centers=False,
            so=False, boundinglat=-30, land=True, proj=None, refwidth=None,
            fig=None, ax=None, regrid=True, add_cyclic=True, cbar=True,
            pval_limit=0.05, values=None, plot_slope=False):
    """Plot inter-model correlation map between two quantities.

    At least one of the quantities must be 2D (defined at each grid cell,
    not a scalar) so that the resulting correlation can be plotted as a map.

    Params:
        dict_x, dict_y (dict): dictionaries where the keys are models
            (ModelCMIP6) and the values are scalar or 2D quantities (as
            xr.DataArray)
        so (bool, default=False): whether to do polar Southern Ocean plot

    Returns:
        fig (Figure)
        ax (Axes)
        models: list with elements of type ModelCMIP6
    """
    if proj is None:
        proj = 'splaea' if so else 'robin'
    lon_0 = 0 if so else 202
    if fig is None and ax is None:
        fig, ax = pplt.subplots(proj=proj, proj_kw=dict(lon_0=lon_0),
                                refwidth=refwidth)
    else:
        assert fig is not None and ax is not None

    dict_x = {ModelCMIP6(mod.center, mod.model, None): da
              for mod, da in dict_x.items()}
    dict_y = {ModelCMIP6(mod.center, mod.model, None): da
              for mod, da in dict_y.items()}
    models = list(set.intersection(set(dict_x.keys()), set(dict_y.keys())))
    models = sorted(models, key=lambda m: m.center)
    assert len(models) > 1, 'Not enough models in common between dict a and b'
    logging.info(f'Using {len(models)} models')

    # make values DataArrays for concatenation
    dict_x = {model: xr.DataArray(val) for model, val in dict_x.items()}
    dict_y = {model: xr.DataArray(val) for model, val in dict_y.items()}

    if regress_centers:
        dict_x = {
            ModelCMIP6(center, center, None):
            xr.concat([dict_x[mod] for mod in dict_x.keys()
                       if mod.center == center], 'model').mean('model')
            for center in [model.center for model in dict_x.keys()]
        }
        dict_y = {
            ModelCMIP6(center, center, None):
            xr.concat([dict_y[mod] for mod in dict_y.keys()
                       if mod.center == center], 'model').mean('model')
            for center in [model.center for model in dict_y.keys()]
        }
        models = sorted(
            set.intersection(set(dict_x.keys()), set(dict_y.keys())),
            key=lambda m: m.center)

    # concatenate
    arr_x = xr.concat([dict_x[model] for model in models], 'model')
    arr_y = xr.concat([dict_y[model] for model in models], 'model')

    # calculate correlation
    try:
        if plot_slope:
            rval, pval, slope = correlation_rpvals(
                arr_x, arr_y, dim='model', return_slope=True)
        else:
            rval, pval = correlation_rpvals(
                arr_x, arr_y, dim='model')
    except ValueError as e:
        logging.info(f'Caught error, regridding before r/pvalue comp: {e}')
        arr_x = _regrid(arr_x)
        arr_y = _regrid(arr_y)
        if plot_slope:
            rval, pval, slope = correlation_rpvals(
                arr_x, arr_y, dim='model', return_slope=True)
        else:
            rval, pval = correlation_rpvals(
                arr_x, arr_y, dim='model')

    stip = xr.where((pval < pval_limit) | np.isnan(rval), 0, 1)

    if plot_slope:
        da_plot = slope
        contourf_kw = dict(symmetric=True, robust=True, extend='both')
        cbar_kw = dict(label='Regression slope', values=values)
    else:
        da_plot = rval
        contourf_kw = dict(vmin=-1, vmax=1, values=values)
        cbar_kw = dict(label=r'$r$', values=values)
    format_kw = dict(land=land)

    # cyc_data, cyc_lon = add_cyclic_point(rval, coord=rval.lon)
    # rval = xr.DataArray(np.array(cyc_data).T,
    #                     coords={'lon': cyc_lon, 'lat': rval.lat})
    im = contourf_plot(
        da=da_plot, ax=ax,
        contourf_kw=contourf_kw, cbar_kw=cbar_kw, cbar=cbar,
        format_kw=format_kw,
        stipling=stip, stipling_style='.......',
        regrid=regrid, fix_grid=False, add_cyclic=add_cyclic,
        southern_ocean=so, boundinglat=boundinglat)

    logging.info('Done')
    return im, fig, ax, models


def correlation_rpvals(
        da_a: xr.DataArray, da_b: xr.DataArray, dim='model',
        return_slope=False):
    """
    Multi-model linear regression of e.g. OHU depth onto stratif. base/trend.

    The r and p-values are from a least squares linear
    regression model (p-value is for the hypothesis that the slope is nonzero).

    Params:
        da_a, da_b (xr.DataArray): arrays consisting of (in-)dependent variable
            concatenated along `dim`, e.g. multimodel data where `dim`='model'.
            dims (`dim`, 'x', 'x')
        dim (str, default='model'): dimension along which to regress

    Returns:
        rval (xr.DataArray): Pearson's correlation coefficient
        pval (xr.DataArray): p-value of test that slope is different from
            zero (two-sided test)

    """
    from scipy.stats import linregress

    def _pval(a, b):
        try:
            good = ~np.isnan(a) & ~np.isnan(b)
            a, b = a[good], b[good]
            if not good.any():
                return np.nan
            pvalue = linregress(a, b).pvalue
            return pvalue
        except Exception:   # all-nan slice, e.g. on continent
            return np.nan

    def _rval(a, b):
        try:
            good = ~np.isnan(a) & ~np.isnan(b)
            if not good.any():
                return np.nan
            a, b = a[good], b[good]
            rvalue = linregress(a, b).rvalue
            return rvalue
        except Exception:   # all-nan slice, e.g. on continent
            return np.nan

    def _slope(a, b):
        try:
            good = ~np.isnan(a) & ~np.isnan(b)
            a, b = a[good], b[good]
            if not good.any():
                return np.nan
            slope = linregress(a, b).slope
            return slope
        except Exception:   # all-nan slice, e.g. on continent
            return np.nan

    pval = xr.apply_ufunc(
        _pval,
        da_a, da_b,
        input_core_dims=[[dim], [dim]],
        output_dtypes=(float, ),
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs=dict(allow_rechunk=True)
    ).compute()

    rval = xr.apply_ufunc(
        _rval,
        da_a, da_b,
        input_core_dims=[[dim], [dim]],
        output_dtypes=(float, ),
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs=dict(allow_rechunk=True)
    ).compute()

    slope = xr.apply_ufunc(
        _slope,
        da_a, da_b,
        input_core_dims=[[dim], [dim]],
        output_dtypes=(float, ),
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs=dict(allow_rechunk=True)
    ).compute()

    if 'lon' in da_a.coords:
        bp = da_a
    elif 'lon' in da_b.coords:
        bp = da_b

    try:
        rval['lon'] = bp.lon
        rval['lat'] = bp.lat
        pval['lon'] = bp.lon
        pval['lat'] = bp.lat
    except Exception:
        pass

    assert not np.isnan(rval).all()
    assert not np.isnan(pval).all()

    if return_slope:
        return rval, pval, slope

    return rval, pval


def sel_season(da, season, reso_init='monthly'):
    """Take monthly array, select season, return yearly array."""
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
              'oct', 'nov', 'dec']

    month_initials = 'jfmamjjasond'

    if season == 'daily':
        assert reso_init == 'daily'
        return da

    if season == 'monthly':
        if reso_init == 'monthly':
            return da
        elif reso_init == 'daily':
            return da.resample(time='1M').mean()

    # yearly mean
    elif season in ['yearly', 'annual']:
        return da.groupby('time.year').mean()

    # single month
    elif season in months:
        month_idx = months.index(season) + 1
        da = da.where(da['time.month'] == month_idx).groupby(
            'time.year').mean()
        return da

    elif season.lower() in ['djf', 'mam', 'jja', 'son']:
        da = da.where(da['time.season'] == season.upper())
        if reso_init == 'daily':
            da = da.resample(time='1M').mean()
        da = da.rolling(min_periods=3, center=True, time=3).mean()
        da = da.groupby('time.year').mean()
        return da

    # multiple months
    elif season in 2*month_initials:
        season_idx = (2*month_initials).index(season)
        if season_idx <= len(month_initials) - len(season):  # inside one year
            da = da.where(
                da['time.month'].isin(
                    list(range(season_idx+1, season_idx+len(season)+1))
                )
            ).groupby('time.year').mean()
            return da
        else:  # wraps into following year (e.g. DJF)
            assert reso_init == 'monthly', 'Cannot do this with daily data yet'
            season_indices = list(range(season_idx, season_idx+len(season)))
            season_indices = [(idx + 1) % 12 for idx in season_indices]
            season_indices = [12 if idx ==
                              0 else idx for idx in season_indices]

            da = da.where(da['time.month'].isin(season_indices))
            da = da.isel(time=slice(season_idx, -(12-season_idx)))
            da = da.coarsen(time=12).mean()
            try:
                da['time'] = np.array([tt.year for tt in da.time.values])
            except AttributeError:
                da['time'] = np.array(
                    [pd.DatetimeIndex([tt]).year[0] for tt in da.time.values])
            da = da.rename(time='year')
            return da

    # yearly statistics
    elif season in ['yearmax', 'yearmin', 'yearmean', 'yearmedian']:
        if season == 'yearmax':
            da = da.groupby('time.year').max()
        elif season == 'yearmin':
            da = da.groupby('time.year').min()
        elif season == 'yearmean':
            da = da.groupby('time.year').mean()
        elif season == 'yearmedian':
            da = da.groupby('time.year').median()
        return da

    elif season == 'summer':
        return xr.where(da.lat >= 0, sel_season(da, 'jja'), sel_season(da, 'djf'))
    elif season == 'winter':
        return xr.where(da.lat >= 0, sel_season(da, 'djf'), sel_season(da, 'jja'))
    elif season == 'diff':
        da_summer = sel_season(da, 'summer')
        da_winter = sel_season(da, 'winter')
        return da_summer - da_winter

    else:
        raise ValueError(f'{season=}')


def contourf_plot(
    ax, da,
    format_kw=None, contourf_kw=None,
    regrid=True, add_cyclic=False,
    lon=None, lat=None, fix_grid=True,
    southern_ocean=False, boundinglat=-30,
    cbar=True, cbar_label=None,
    cbar_kw=None,
    stipling=None, stipling_style=None, stipling_color='k',
    fill=True
):
    """Plot a contour+contourf map of a 2D array.
    Includes gridlines, coastlines, and black land mask.

    Parameters:
        ax: proplot axis with (e.g.) proj='robin'
        da (xr.DataArray): 2D array to plot
        contourf_kw:  dict with keys 'cmap', 'levels', and 'extend'
        stipling (xr.DataArray): 2D array with 1=stipling, else no stipling.
            same shape as `da`

    Returns:
        c -- (QuadContourSet) return value of ax.contourf() call, to be used
             for e.g. colorbar
    """
    try:
        da = da.compute()
    except Exception:
        pass

    if lat is None:
        lat = da.lat
    if lon is None:
        lon = da.lon

    orig_lon = lon
    if add_cyclic:
        da, lon = _add_cyclic(da, lon)

    if regrid:
        da = _regrid(da)
        lon = da.lon
        lat = da.lat

    if southern_ocean:
        da = da.where(da.lat <= boundinglat)
        ax.format(boundinglat=boundinglat)

    if fix_grid:
        lon, lat, da = fix_orca(lon, lat, da)

    contour_func = ax.contourf if fill else ax.contour

    if contourf_kw is None:
        im = contour_func(lon, lat, da)
    else:
        im = contour_func(lon, lat, da, **contourf_kw)

    if cbar:
        if cbar_label is not None and cbar_kw is not None:
            try:
                cbar = ax.colorbar(im, label=cbar_label, **cbar_kw)
            except TypeError:
                msg = f'got {cbar_label=} and {cbar_kw["label"]=}, '
                msg += 'disregarding cbar_label'
                warnings.warn(msg)
                cbar = ax.colorbar(im, **cbar_kw)
        elif cbar_kw is not None:
            cbar = ax.colorbar(im, **cbar_kw)
        elif cbar_label is not None:
            cbar = ax.colorbar(im, label=cbar_label)
        else:
            cbar = ax.colorbar(im)

    format_kw_default = dict(land=True, gridlinewidth=0)
    if format_kw is not None:
        format_kw_default.update(format_kw)
        format_kw = format_kw_default
    else:
        format_kw = format_kw_default
    if not format_kw['land']:
        format_kw['coast'] = True
    ax.format(**format_kw)

    # TODO don't use recursion
    if stipling is not None:
        if stipling_style is None:
            stipling_style = '....'
        contourf_plot(ax=ax, da=stipling,
                      contourf_kw=dict(levels=[0.8, 1.2], hatches=[
                                       stipling_style], hatchcolor=stipling_color),
                      cbar=False, southern_ocean=southern_ocean,
                      format_kw=format_kw,
                      regrid=regrid,
                      lon=orig_lon, lat=lat,
                      fix_grid=fix_grid, cbar_label=cbar_label,
                      cbar_kw=cbar_kw, add_cyclic=add_cyclic,
                      boundinglat=boundinglat)

    return im


def _regrid(da, xres=1, yres=0.5, ignore_degenerate=True, periodic=True):
    """Regrid a dataset to a globally uniform lat-lon grid."""
    import xesmf as xe

    da_name = 'var' if da.name is None else da.name
    ds = da.to_dataset(name=da_name)

    target_grid = xe.util.grid_global(d_lon=xres, d_lat=yres)

    regridder = xe.Regridder(
        ds, target_grid, 'bilinear', periodic=periodic,
        ignore_degenerate=ignore_degenerate)

    ds_out = regridder(ds)
    da_out = ds_out[da_name]
    return da_out


def _add_cyclic(da, lon):
    from cartopy.util import add_cyclic_point

    try:
        da = da.transpose('lat', 'lon')
    except Exception:
        try:
            da = da.transpose('latitude', 'longitude')
        except Exception:
            da = da.transpose('y', 'x')

    cyc_da, cyc_lon = add_cyclic_point(da, coord=lon)
    try:
        da = xr.DataArray(cyc_da, coords=dict(lon=cyc_lon, lat=da.lat))
    except Exception:
        da = xr.DataArray(cyc_da.T, coords=dict(lon=cyc_lon, lat=da.lat)).T
    return da, da.lon


def fix_orca(lon: np.ndarray, lat: np.ndarray, data: np.ndarray):
    """Fix ORCA grid data by projecting onto regular lon-lat grid.

    Linearly interpolate longitude, latitude and data values onto a global
    regular 1-degree grid.
    Note: currently works only for 2d data, no time-dependence.

    From: ajheaps.github.io/cf-plot/irregular.html

    Args:
        lon (ndarray): 2D longitudes on original grid (e.g. ORCA)
        lat (ndarray): 2D latitudes on original grid (e.g. ORCA)
        data (ndarray): 2D data values on original grid (e.g. ORCA)

    Returns:
        lon_new (ndarray): 2D longitudes on new regular grid
        lat_new (ndarray): 2D latitudes on new regular grid
        data_new (ndarray): 2D data values on new regular grid

    TODO: add support for arrays with time dimension (broadcast over time)
    """
    from scipy.interpolate import griddata

    assert data.ndim == 2, f'data must be 2D, got {data.ndim=}'

    # convert to numpy array to allow flattening
    lon, lat, data = np.array(lon), np.array(lat), np.array(data)

    lon, lat, data = lon.flatten(), lat.flatten(), data.flatten()
    pts = np.squeeze(np.where(lon < -150))
    lon = np.append(lon, lon[pts]+360)
    lat = np.append(lat, lat[pts])
    data = np.append(data, data[pts])

    pts = np.squeeze(np.where(lon > 150))
    lon = np.append(lon, lon[pts]-360)
    lat = np.append(lat, lat[pts])
    data = np.append(data, data[pts])

    # target grid: global 1 degree
    xpts = np.arange(-180, 180.25, 1)
    ypts = np.arange(-90, 90.25, 1)
    lon_new, lat_new = np.meshgrid(xpts, ypts)

    data_new = griddata((lon, lat), data, (lon_new, lat_new),
                        method='linear')

    return lon_new, lat_new, data_new
