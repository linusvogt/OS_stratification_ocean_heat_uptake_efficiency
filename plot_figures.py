import logging

import numpy as np
import proplot as pplt
import util_eof
import xarray as xr

import util
import util_data


def figure_01():
    """Scatter plots of OHUE vs. global mean N² & MLD + 3 MOCs."""
    # OHUE
    dict_y = util_data.load_var_dict(var='ohue', kwargs=dict())

    # set up figure
    layout = [
        [0, 1, 1, 2, 2, 0],
        [0, 1, 1, 2, 2, 0],
        [3, 3, 4, 4, 5, 5],
        [3, 3, 4, 4, 5, 5],
    ]
    fig, axs = pplt.subplots(
        layout, share=0, hratios=[1, 1, 1, 1], wratios=[1, 1, 1, 1, 1, 1])
    axs.format(abc='(a)')

    # set up model numbers
    models = list(dict_y.keys())
    models = sorted(models, key=lambda m: (m.center, m.model))
    model_numbers = {model.model: mm for mm, model in enumerate(models)}

    # load x data
    variables_x = ['strat', 'mld', 'amoc', 'm_so', 'm_wmt']
    handles = {}
    for var_x, ax in zip(variables_x, axs):

        if var_x == 'strat':
            name_x = 'Global mean N²'
            unit_x = 'm/s²'
            dict_x = util_data.load_var_dict(
                var='strat',
                kwargs=dict(exp='piControl', var='total', maxlev=1500))
        elif var_x == 'mld':
            name_x = 'Global mean MLD'
            unit_x = 'm'
            dict_x = util_data.load_var_dict(
                var='mld', kwargs=dict())
        elif var_x == 'amoc':
            name_x = 'AMOC'
            unit_x = 'Sv'
            dict_x = util_data.load_var_dict(
                var='moc', kwargs=dict(name='amoc'))
        elif var_x == 'm_so':
            name_x = r'$M_\mathrm{SO}$'
            unit_x = 'Sv'
            dict_x = util_data.load_var_dict(
                var='moc', kwargs=dict(name='upper'))
        elif var_x == 'm_wmt':
            name_x = r'$M_\mathrm{WMT}$'
            unit_x = 'Sv'
            dict_x = util_data.load_var_dict(
                var='moc', kwargs=dict(name='wmt'))

        # global mean
        if var_x in ['strat', 'mld']:
            area = util.load_regrid_area()
            dict_x = {model:
                      da.weighted(area).mean(['lat', 'lon'])
                      for model, da in dict_x.items()}

        # set up data dicts and model list
        dict_x = {util.ModelCMIP6(mod.center, mod.model, None): da
                  for mod, da in dict_x.items()}
        dict_y = {util.ModelCMIP6(mod.center, mod.model, None): da
                  for mod, da in dict_y.items()}
        models = list(set.intersection(set(dict_x.keys()), set(dict_y.keys())))
        models = sorted(models, key=lambda m: (m.center, m.model))
        assert len(models) > 1, \
            'Not enough models in common between dict a and b'

        xvals, yvals = {}, {}
        models_used = []

        # scatter
        for model in models:
            x, y = float(dict_x[model]), float(dict_y[model])
            if np.isnan(x) or np.isnan(y):
                continue

            # numbers as scatter markers
            mm = model_numbers[model.model]
            num_colon = f'{mm+1}:'
            label = (f' {num_colon: <6}{model.model}' if mm+1 < 10
                     else f'{num_colon: <6}{model.model}')
            h = ax.scatter(
                x, y,
                alpha=0,
                label=label
            )
            ax.text(
                x=x, y=y, s=mm+1,
                color='k',
            )

            if var_x == 'strat':
                handles[model] = h
            xvals[model] = x
            yvals[model] = y
            models_used.append(model)

        xvals = np.array([xvals[model] for model in models_used])
        yvals = np.array([yvals[model] for model in models_used])

        textloc = 'upper right' if var_x == 'strat' else 'upper left'
        util.plot_linear_regression(
            ax=ax, xvals=xvals, yvals=yvals,
            textloc=textloc)
        util.fix_xylims(ax=ax, xvals=xvals, yvals=yvals)

        # set title
        title = f'{name_x} vs. OHUE'
        ax.set_title(title, fontweight='bold')
        ax.format(
            xlabel=f'{name_x} ({unit_x})', ylabel='OHUE (W/m² per °C)')

    fig.legend(list(handles.values()), loc='b', label='CMIP6 models')

    fig.save('figures/fig_01.png', dpi=250)


def figure_02():
    """Corrmaps of N² and MLD vs. OHUE."""
    pplt.rc.update(fontsize=9)
    dict_y = util_data.load_var_dict(
        var='ohue', kwargs=dict())

    variables_x = ['strat', 'mld']

    fig, axs = pplt.subplots(
        nrows=2, proj='pcarree', proj_kw=dict(lon_0=202), refwidth='120mm')
    axs.format(abc='(a)')

    var_names = {
        'strat': 'Local preindustrial stratification',
        'mld': 'Local preindustrial MLD'}

    for var_x, ax in zip(variables_x, axs):
        if var_x == 'strat':
            dict_x = util_data.load_var_dict(
                var='strat',
                kwargs=dict(exp='piControl', var='total', maxlev=1500))
        elif var_x == 'mld':
            dict_x = util_data.load_var_dict(var='mld', kwargs=dict())

        # plot
        try:
            im, fig, ax, models = util.corrmap(
                dict_x, dict_y, so=False, proj='pcarree',
                refwidth=5, regress_centers=False, regrid=False,
                add_cyclic=True, plot_slope=False, fig=fig, ax=ax, cbar=False)
        except Exception:
            im, fig, ax, models = util.corrmap(
                dict_x, dict_y, so=False, proj='pcarree',
                refwidth=5, regress_centers=False, regrid=True,
                add_cyclic=False, plot_slope=False, fig=fig, ax=ax, cbar=False)

        if len(models) < 28:
            raise ValueError('Not (yet) all models available')

        # set title
        title = (f'{var_names[var_x]} vs. OHUE, inter-model correlation')
        ax.set_title(title, fontweight='bold')
        ax.format(facecolor='grey')

    fig.colorbar(im, loc='r', length=0.7, label=r'$r$')

    fig.format(
        latlines=30, lonlines=60, lonlabels='b', latlabels='l'
    )

    fig.save('figures/fig_02.png', dpi=250)


def figure_03():
    """Corrmaps of N² and MLD vs. the three MOCs."""

    variables_x = ['strat', 'mld']
    variables_y = ['amoc', 'm_so', 'm_wmt']

    fig, axs = pplt.subplots(
        ncols=2, nrows=3,
        proj='pcarree', proj_kw=dict(lon_0=202), refwidth='120mm')
    axs.format(abc='(a)')

    var_names = {
        'strat': 'Local preindustrial stratification',
        'mld': 'Local preindustrial MLD',
        'amoc': 'AMOC',
        'm_so': r'$M_\mathrm{SO}$',
        'm_wmt': r'$M_\mathrm{WMT}$',
    }

    for xx, var_x in enumerate(variables_x):

        # load x variable
        if var_x == 'strat':
            dict_x = util_data.load_var_dict(
                var='strat', kwargs=dict(
                    exp='piControl', var='total', maxlev=1500))
        elif var_x == 'mld':
            dict_x = util_data.load_var_dict(
                var='mld', kwargs=dict())

        for yy, var_y in enumerate(variables_y):

            # load y variable
            if var_y == 'amoc':
                dict_y = util_data.load_var_dict(
                    var='moc', kwargs=dict(name='amoc'))
            elif var_y == 'm_so':
                dict_y = util_data.load_var_dict(
                    var='moc', kwargs=dict(name='upper'))
            elif var_y == 'm_wmt':
                dict_y = util_data.load_var_dict(
                    var='moc', kwargs=dict(name='wmt'))

            ax = axs[yy, xx]

            # plot
            try:
                im, fig, ax, models = util.corrmap(
                    dict_x, dict_y, so=False, proj='pcarree',
                    refwidth=5, regress_centers=False, regrid=False,
                    add_cyclic=True, plot_slope=False, fig=fig, ax=ax,
                    cbar=False)
            except Exception:
                im, fig, ax, models = util.corrmap(
                    dict_x, dict_y, so=False, proj='pcarree',
                    refwidth=5, regress_centers=False, regrid=True,
                    add_cyclic=False, plot_slope=False, fig=fig, ax=ax,
                    cbar=False)

            # set title
            title = (f'{var_names[var_x]} vs. {var_names[var_y]}, '
                     + 'inter-model correlation')
            ax.format(title=title, facecolor='grey')

    long_var_names = {
        'amoc': 'AMOC',
        'm_so': f'Southern Ocean overturning ({var_names["m_so"]})',
        'm_wmt': f'Southern Ocean overturning ({var_names["m_wmt"]})',
    }

    axs.format(
        toplabels=[var_names[var] for var in variables_x],
        leftlabels=[long_var_names[var] for var in variables_y],
    )

    fig.colorbar(im, loc='b', length=0.7, label=r'$r$')
    fig.format(
        latlines=30, lonlines=60, lonlabels='b', latlabels='l'
    )

    fig.save('figures/fig_03.png', dpi=250)


def figure_04():
    """Stratification ensemble mean, spread, and bias wrt. ECCO."""

    fig, axs = pplt.subplots(
        nrows=3, ncols=3, proj='pcarree', proj_kw=dict(lon_0=202))
    axs.format(abc='(a)')

    strat_vars = ['total', 'T', 'S']
    plot_modes = ['mean', 'cv', 'bias']

    var_names = {
        'total': r'$N^2$',
        'T': r'$N^2_T$',
        'S': r'$N^2_S$',
    }
    mode_names = {
        'mean': 'ensemble mean',
        'cv': 'relative ensemble spread',
        'bias': 'ensemble mean bias',
    }

    for vv, var in enumerate(strat_vars):
        logging.info(f'{var=}')
        for mm, mode in enumerate(plot_modes):
            logging.info(f'\t{mode=}')
            ax = axs[vv, mm]

            if mode == 'mean':
                plot_n2(
                    var=var, mode='mean',
                    ecco_hist_period=True, fig=fig, ax=ax)
            elif mode == 'cv':
                plot_n2_variance_fraction(
                    which='cv_total', var=var,
                    ecco_hist_period=True, fig=fig, ax=ax)
            elif mode == 'bias':
                plot_n2(
                    var=var, mode='bias',
                    ecco_hist_period=True, fig=fig, ax=ax)

            ax.format(title=f'{var_names[var]}, {mode_names[mode]}')

    axs.format(
        toplabels=('Ensemble mean', 'Relative ensemble spread',
                   'Ensemble mean bias'),
        leftlabels=(
            'Total ' + r'$N^2$',
            'Temperature ' + r'$N^2$',
            'Salinity ' + r'$N^2$')
    )

    fig.format(
        latlines=30, lonlines=60, lonlabels='b', latlabels='l'
    )

    fig.save('figures/fig_04.png', dpi=250)


def figure_05():
    """Inter-model stratification EOFs and regional vs. local N² corrmaps."""

    fig, axs = pplt.subplots(
        nrows=2, ncols=2, proj='pcarree', proj_kw=dict(lon_0=202),
        refwidth='110mm')
    axs.format(abc='(a)')

    panels = [('eof', 0), ('corrmap', 'global'),
              ('eof', 1), ('corrmap', 'natl_40N')]

    for panel, ax in zip(panels, axs):

        if panel[0] == 'eof':
            plot_eof(
                var='strat', season='annual', eof_mode=panel[1],
                fig=fig, ax=ax, cbar=False)
        elif panel[0] == 'corrmap':
            im = corrmap_strat_strat_scalar(
                region=panel[1], var='total', season='annual',
                fig=fig, ax=ax, cbar=False)
            region_name = {'global': 'Global mean',
                           'natl_40N': 'Subpolar N. Atlantic'}
            title = (f'{region_name[panel[1]]} vs. local preindustrial '
                     + 'stratification\ninter-model correlation')
            ax.format(title=title)

    fig.format(
        latlines=30, lonlines=60, lonlabels='b', latlabels='l'
    )

    fig.colorbar(im, loc='r', length=0.7, label=r'$r$')

    fig.save('figures/fig_05.png', dpi=250)


def plot_n2(var, mode, ecco_hist_period=False, fig=None, ax=None):
    if ecco_hist_period and mode == 'std':
        raise ValueError('Cannot compare ens std with ECCO')

    data_dict = {}
    exp = 'historical' if ecco_hist_period else 'piControl'
    for mm, model in util.iter_models(exp):
        logging.info(
            f'{mm+1}/{len(list(util.iter_models(None)))}: {model.model}')

        try:
            da = util_data.load_n2_regrid(
                model=model, exp=exp, maxlev=1500, var=var)
            if ecco_hist_period:
                da = da.sel(year=slice('1992', '2017')).mean('year')
            data_dict[model] = da
        except Exception as e:
            logging.info(f'Skipping {model.model} due to: {e}')

    logging.info(f'{len(data_dict)} models')

    if mode in ['mean', 'bias']:
        da = xr.concat(list(data_dict.values()), 'model').mean('model')
    elif mode == 'std':
        da = xr.concat(list(data_dict.values()), 'model').std('model')

    if ecco_hist_period and mode == 'bias':
        da_ecco = util_data.load_n2_ecco(n2_var=var, lev_int=True).mean('time')
        da = da - da_ecco
        arctic = xr.open_dataset(
            'region_masks/mask_arctic_1deg_reg_grid.nc').mask
        da = da.where(arctic != 1)

    if fig is None and ax is None:
        fig, ax = pplt.subplots(proj='pcarree', proj_kw=dict(lon_0=202),
                                refwidth=5)
    else:
        assert fig is not None and ax is not None, \
            'fig and ax must be either both None or both supplied'

    vmin = (0 if (var in ['total', 'T'] or mode == 'std')
            and not ecco_hist_period
            else None)
    extend = 'max' if mode == 'std' else 'both'
    if mode == 'mean':
        contourf_kw = dict(vmin=-0.05, vmax=0.05,
                           cmap='Div', extend='both')
    elif mode == 'bias':
        contourf_kw = dict(vmin=-0.005, vmax=0.005,
                           cmap='Div', extend='both')
    elif mode == 'std':
        contourf_kw = dict(vmin=vmin, robust=True, cmap='Reds3', extend=extend)

    util.contourf_plot(
        ax=ax, da=da,
        contourf_kw=contourf_kw,
        cbar_kw=dict(label='N² [m/s²]'),
        regrid=True, fix_grid=False, add_cyclic=False
    )

    title_str = f'Annual {var} stratification'
    if mode == 'mean':
        title = f'{title_str}, multi-model mean'
    elif mode == 'bias':
        title = f'{title_str}, multi-model mean minus ECCO'
    elif mode == 'std':
        title = f'{title_str}, inter-model std. dev.'
    ax.format(title=title, facecolor='grey')

    return fig, ax


def plot_n2_variance_fraction(
        which, var, ecco_hist_period=False, fig=None, ax=None):
    # assert var in ['T', 'S']
    data_dict = {n2_var: {} for n2_var in ['total', var]}
    for n2_var in ['total', var]:
        exp = 'historical' if ecco_hist_period else 'piControl'
        for mm, model in util.iter_models(exp):
            logging.info(
                f'{mm+1}/{len(list(util.iter_models(None)))}: {model.model}')

            try:
                da = util_data.load_n2_regrid(
                    model=model, exp=exp, maxlev=1500, var=n2_var)
                if ecco_hist_period:
                    da = da.sel(year=slice('1992', '2017')).mean('year')
                data_dict[n2_var][model] = da
            except Exception as e:
                logging.info(f'Skipping {model.model} due to: {e}')

    std_dict = {
        n2_var:
        xr.concat(list(data_dict[n2_var].values()), 'model').std('model')
        for n2_var in ['total', var]
    }
    mean_dict = {
        n2_var:
        xr.concat(list(data_dict[n2_var].values()), 'model').mean('model')
        for n2_var in ['total', var]
    }
    if which == 'cv_total':
        da = 100 * (std_dict[var] / np.abs(mean_dict['total']))
    elif which == 'cv':
        da = 100 * (std_dict[var] / np.abs(mean_dict[var]))
    elif which == 'varfrac':
        da = 100 * (std_dict[var]**2 / std_dict['total']**2)

    if fig is None and ax is None:
        fig, ax = pplt.subplots(proj='pcarree', proj_kw=dict(lon_0=202),
                                refwidth=5)
    else:
        assert fig is not None and ax is not None

    vmax = {'varfrac': 150, 'cv_total': 90, 'cv': 100}[which]
    contourf_kw = dict(vmin=0, vmax=vmax, cmap='Speed', extend='max')

    util.contourf_plot(
        ax=ax, da=da,
        contourf_kw=contourf_kw,
        cbar_kw=dict(label='%'),
        regrid=False, fix_grid=False, add_cyclic=True
    )

    title = {
        'varfrac': f'N^2_{var} variance relative to total N² variance',
        'cv': f'N^2_{var} std. dev. relative to N²_{var} average',
        'cv_total': f'N^2_{var} std. dev. relative to total N² average'
    }[which]
    ax.format(title=title)

    return fig, ax


def plot_eof(var, season, eof_mode, n_outliers=5, fig=None, ax=None,
             hatch_density=True, cbar=True):
    data, models = util_eof.load_eof_input(var, season)
    eof, pcs, varfrac, models, outlier_models = util_eof.eof_without_outliers(
        data, models, n_eofs=5, n_outliers=n_outliers)

    if fig is None and ax is None:
        fig, ax = pplt.subplots(
            proj='pcarree', proj_kw=dict(lon_0=202), refwidth='83mm')
    else:
        assert fig is not None and ax is not None

    if var == 'strat':
        eof = -eof
        pcs = -pcs

        util.contourf_plot(
            ax=ax, da=eof.sel(mode=eof_mode), regrid=True, add_cyclic=False,
            fix_grid=False, contourf_kw=dict(vmin=-1, vmax=1),
            stipling=None, stipling_style='oo', stipling_color='red',
            cbar=cbar)
        var_expl = float(varfrac.sel(mode=eof_mode))
        ax.format(
            title=f'EOF #{eof_mode+1} explains {100*var_expl:.1f}% of inter-model variance')

        px = ax.panel('l', share=False)
        pc_vals = pcs.sel(mode=eof_mode).values
        px.violin(pc_vals.reshape(-1, 1))
        #px.format(ylim=(-1, 1))
        for val in pc_vals:
            px.hlines(y=val, x1=-0.02, x2=0.02, color='k', zorder=20)
        px.format(title='Loadings')

    ax.format(facecolor='grey')

    return fig, ax


def corrmap_strat_strat_scalar(
        region, var, season, fig=None, ax=None, cbar=True):
    # load data
    dict_x = util_data.load_var_dict(
        var='strat', kwargs=dict(var=var, maxlev=1500, exp='piControl'))
    dict_y = util_data.load_var_dict(
        var='strat_scalar', kwargs=dict(region=region, var=var))

    mask = xr.open_dataset(util_data.reg_grid_mask_outpath(region)).mask
    weights = xr.open_dataset(
        'region_masks/reg_1deg_grid_area_ocean.nc').cell_area.fillna(0.)

    dict_y = {
        model:
        (da * mask).weighted(weights).mean(['lon', 'lat']).squeeze().compute()
        for model, da in dict_x.items()
    }

    # plot
    im, fig, ax, models = util.corrmap(
        dict_x, dict_y, so=False, proj='pcarree', refwidth=5,
        regress_centers=False, regrid=False, add_cyclic=True,
        fig=fig, ax=ax, cbar=cbar)

    # contour around region
    if region != 'global':
        ax.contour(mask.lon, mask.lat, mask.fillna(0.), color='blue3')

    # set title
    region_name = 'N. Atlantic' if 'natl' in region else region.title()
    title = f'Preindustrial {var} stratification, {region_name} vs. local'
    title += '\ninter-model correlation'
    ax.format(title=title)
    ax.format(facecolor='grey')

    return im


if __name__ == "__main__":
    util.log_setup('figures')
    figure_05()
