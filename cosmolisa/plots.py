import numpy as np
import corner
import os
import sys
import matplotlib.pyplot as plt

from cosmolisa import cosmology as cs
from cosmolisa import likelihood as lk

truth_color = "#4682b4"

# Mathematical labels used by the different models.
labels_plot = {
    'h': r'$h$',
    'om': r'$\Omega_m$',
    'ol': r'$\Omega_\Lambda$',
    'w0': r'$w_0$',
    'w1': r'$w_1$',
    'Xi0': r'$\Xi_0$',
    'n1': r'$n1$',
    'b': r'$b$',
    'n2': r'$n2$',
    'RatePW': [r'$h$', r'$\Omega_m$', r'$\log_{10} r_0$', r'$p_1$'],
    'Rate': [r'$h$', r'$\Omega_m$', r'$\log_{10} r_0$', r'$\log_{10} p_1$',
             r'$p_2$', r'$p_3$'],
    'Luminosity': [r'$\phi^{*}/Mpc^{3}$', r'$a$', r'$M^{*}$', r'$b$',
                   r'$\alpha$', r'$c$'],
    }


def hist_settings(par, samples, outdir, name, bins=20, truths=None):
    """Histogram for single-parameter inference."""
    fmt = "{{0:{0}}}".format('.3f').format
    fig = plt.figure()
    plt.hist(samples, density=True, bins=bins, alpha = 1.0, 
             histtype='step', edgecolor="black", lw=1.2)
    quantiles = np.quantile(samples, [0.05, 0.5, 0.95])
    plt.axvline(quantiles[0], linestyle='dashed', color='k', lw=1.5)
    plt.axvline(quantiles[1], linestyle='dashed', color='k', lw=1.5)
    plt.axvline(quantiles[2], linestyle='dashed', color='k', lw=1.5)
    plt.axvline(truths, linestyle='dashed', color=truth_color, lw=1.5)
    med, low, up = (quantiles[1], quantiles[0]-quantiles[1], 
                    quantiles[2]-quantiles[1])
    plt.title(r"${par} = {{{med}}}_{{{low}}}^{{+{up}}}$".format(
        par=labels_plot[par].strip('r$$'), med=fmt(med),
        low=fmt(low), up=fmt(up)), size=16)
    plt.xlabel(labels_plot[par], fontsize=16)
    fig.savefig(os.path.join(outdir,'Plots', name+'.pdf'),
        bbox_inches='tight')
    fig.savefig(os.path.join(outdir,'Plots', name+'.png'),
        bbox_inches='tight')

def histogram(x, **kwargs):
    """Function to call hist_settings with the appropriate model."""
    hist_settings(par=kwargs['par'],
                  samples=x[kwargs['par']],
                  truths=kwargs['truths'][kwargs['par']],
                  outdir=kwargs['outdir'],
                  name=f"histogram_{kwargs['par']}_90CI")


def corner_config(labels, samps_tuple, quantiles_plot, 
                  outdir, name, truths=None, **kwargs):
    """Instructions used to make corner plots.
    'title_quantiles' is not specified, hence plotted quantiles
    coincide with 'quantiles'. This holds for the version of corner.py
    indicated in the README file. 
    """
    samps = np.column_stack(samps_tuple)
    fig = corner.corner(samps,
                        labels=labels,
                        quantiles=quantiles_plot,
                        show_titles=True, 
                        title_fmt='.3f',
                        title_kwargs={'fontsize': 16},
                        label_kwargs={'fontsize': 16},
                        use_math_text=True,
                        truths=truths)
    fig.savefig(os.path.join(outdir,"Plots", name+".pdf"),
                bbox_inches='tight')
    fig.savefig(os.path.join(outdir,"Plots", name+".png"),
                bbox_inches='tight')

def corner_plot(x, **kwargs):
    """Function to call corner_config."""
    samps_tuple = tuple([x[p] for p in kwargs['pars']])
    truths = [kwargs['truths'][p] for p in kwargs['pars']]
    labels = [labels_plot[p] for p in kwargs['pars']]
    corner_config(labels,
                  samps_tuple=samps_tuple,
                  quantiles_plot=[0.16, 0.5, 0.84],
                  truths=truths,
                  outdir=kwargs['outdir'],
                  name="corner_plot_68CI")
    corner_config(labels,
                  samps_tuple=samps_tuple,
                  quantiles_plot=[0.05, 0.5, 0.95],
                  truths=truths,
                  outdir=kwargs['outdir'],
                  name="corner_plot_90CI")
