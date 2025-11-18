#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import configparser
import subprocess
import numpy as np
import json
import logging
import argparse, shutil
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from pprint import pformat

# Import internal and external modules.
from cosmolisa import readdata
from cosmolisa import plots
from cosmolisa import cosmology as cs
from cosmolisa import likelihood as lk
from nessai.model import Model
from nessai.flowsampler import FlowSampler


class CosmologicalModel(Model):
    """CosmologicalModel class:
    Data, likelihood, prior, and settings of the analysis
    are specified here. The abstract modules 'log_prior' and
    'log_likelihood', as well as the attributes 'names' and
    'bounds', are inherited from nessai.model.Model and
    have to be explicitly defined inside this class.
    """

    def __init__(self, model, data, **kwargs):

        # Define class attributes
        self.data = data
        self.N = len(self.data)
        self.event_class = kwargs['event_class']
        self.model_str = model
        self.model = model.split("+")
        self.truths = kwargs['truths']
        self.z_threshold = kwargs['z_threshold']
        self.snr_threshold = kwargs['snr_threshold']
        self.com_vol_mode = kwargs['com_vol_mode']
        self.dl_gw_true = kwargs['dl_gw_true']
        self.O = None
        self.gal_interp = kwargs['gal_interp']
        self.alpha = kwargs['alpha']
        self.use_alpha = kwargs.get('use_alpha', False)
        self.logger = kwargs['logger']

        # Define priors and bounds.
        self.names_list = []
        self.bounds_dict = dict()

        if ('h' in self.model):
            self.names_list.append('h')
            self.bounds_dict['h'] = kwargs['prior_bounds']['h']

        if ('om' in self.model):
            self.names_list.append('om')
            self.bounds_dict['om'] = kwargs['prior_bounds']['om']

        if ('ol' in self.model):
            self.names_list.append('ol')
            self.bounds_dict['ol'] = kwargs['prior_bounds']['ol']

        if ('w0' in self.model):
            self.names_list.append('w0')
            self.bounds_dict['w0'] = kwargs['prior_bounds']['w0']

        if ('w1' in self.model):
            self.names_list.append('w1')
            self.bounds_dict['w1'] = kwargs['prior_bounds']['w1']

        # Check truths and prior bounds compatibility.
        for par in self.names_list:
            assert kwargs['prior_bounds'][par][0] <= self.truths[par], (
             f"{par}: your lower prior bound excludes the true value!")
            assert kwargs['prior_bounds'][par][1] >= self.truths[par], (
             f"{par}: your upper prior bound excludes the true value!")

        self.names = self.names_list
        self.bounds = self.bounds_dict
        
        assert len(self.names) != 0, ("Undefined parameter space!"
        "Please check that the model exists.")

        
        self._initialise_galaxy_hosts()
            

        self.logger.info("\n"+5*"===================="+"\n")
        self.logger.info("CosmologicalModel model initialised with:")
        self.logger.info(f"Event class: {self.event_class}")
        self.logger.info(f"Analysis model: {self.model}")
        self.logger.info(f"Number of events: {len(self.data)}")
        self.logger.info(f"Free parameters: {self.names}")
        self.logger.info("\n"+5*"===================="+"\n")
        self.logger.info("Prior bounds:")
        for name in self.names:
            self.logger.info(f"{str(name).ljust(17)}: {self.bounds[name]}")
        self.logger.info("\n"+5*"===================="+"\n")

    def _initialise_galaxy_hosts(self):
        self.hosts = {
            e.ID: np.array([(g.redshift, g.dredshift, g.weight)
            for g in e.potential_galaxy_hosts]) for e in self.data
            }
        self.galaxy_redshifts = np.hstack([self.hosts[e.ID][:,0] 
            for e in self.data]).copy(order='C')

    def _alpha_eval(self, h, om):
        a = float(self.alpha(np.array([h, om])))
        if not np.isfinite(a) or a <= 0.0:
            return 1e-12
        return a

    def log_prior(self, x):
        """
        Returns natural-log of prior given a live point assuming
        uniform priors on each parameter.        
        """
        logP = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            logP -= np.log(self.bounds[n][1] - self.bounds[n][0])

        return logP


    def log_likelihood(self, x):
        """Natural-log-likelihood assumed in the inference.
        It implements the inference model settings according
        to the options specified by the user.
        """
        logL_GW = 0.0
        cosmo_par = [self.truths['h'], self.truths['om'],
                     self.truths['ol'], self.truths['w0'],
                     self.truths['w1']]
        if ('h' in self.model):
            cosmo_par[0] = x['h']
        if ('om' in self.model):
            cosmo_par[1:3] = x['om'], 1.0 - x['om']
        if ('ol' in self.model):
            cosmo_par[2] = x['ol']
        if ('w0' in self.model):
            cosmo_par[3] = x['w0']
        if ('w1' in self.model):
            cosmo_par[4] = x['w1']            
        else:
            pass                
        self.O = cs.CosmologicalParameters(*cosmo_par)

        if (self.event_class == 'dark_siren'):
            ll_vals = []
            for e in self.data:
                z_grid, mixture = self.gal_interp[str(e.ID)]
                likelihood = lk.lk_dark_single_event_trap(
                    e.dl_scat if self.dl_gw_true == 0 else e.dl, 
                    e.sigmadl, 
                    self.O,
                    e.zmin, e.zmax, 
                    z_grid, mixture,
                    self.com_vol_mode) 
                ll_vals.append(likelihood)
            ll = np.asarray(ll_vals, dtype=float)
            logL_events = np.log(np.clip(ll, 1e-300, None)).sum()

            logL_sel = 0.0
            if self.use_alpha:
                logL_sel = -self.N * np.log(self._alpha_eval(self.O.h, self.O.om))

            logL_GW = logL_events + logL_sel

        elif (self.event_class == 'MBHB'):
            ll_vals = [lk.lk_bright_single_event_trap(
                    self.hosts[e.ID], e.dl, e.sigmadl, self.O,
                    self.model_str, zmin=e.zmin, zmax=e.zmax)
                    for e in self.data]
            logL_GW = np.log(np.clip(ll_vals, 1e-300, None)).sum()

        self.O.DestroyCosmologicalParameters()

        return float(logL_GW)

usage="""\n\n %prog --config-file config.ini\n
    ######################################################################################################################################################
    IMPORTANT: This code requires the installation of the 'nessai' package: https://github.com/mj-will/nessai
               See the instructions in cosmolisa/README.md.
    ######################################################################################################################################################

    #=======================#
    # Input parameters      #
    #=======================#

    'com_vol_mode'         Default: 0.                                       Include a dV/dz factor in the likelihood (0: off, 1: with normalisation, 2: without normalisation).
    'data'                 Default: ''.                                      Data location.
    'dl_gw_true'           Default: 0.                                       Use the true luminosity distance of the GW event instead of the observed one.
    'equal_wj'             Default: 0.                                       Impose all galaxy angular weights equal to 1.
    'event_ID_list'        Default: ''.                                      String (without single/double quotation marks) of specific ID events to be read (separated by commas).
    'event_class'          Default: ''.                                      Class of the event(s) ['dark_siren', 'MBHB'].
    'max_hosts'            Default: 0.                                       Select events according to the allowed maximum number of hosts.
    'model'                Default: ''.                                      Specify the cosmological parameters to sample over ['h', 'om', 'ol', 'w0', 'wa'] and the type of analysis ['GW'] separated by a '+'.
    'one_host_sel'         Default: 0.                                       For each event, associate only the nearest-in-redshift host (between z_gal and event z_true).
    'outdir'               Default: './default_dir'.                         Directory for output.
    'postprocess'          Default: 0.                                       Run only the postprocessing. It works only with reduced_catalog=0.
    'prior_bounds'         Default: {"h": [0.6, 0.76], "om": [0.04, 0.5]}.   Prior bounds specified by the user. Must contain all the parameters specified in 'model'.
    'random'               Default: 0.                                       Run a joint analysis with N events, randomly selected.
    'reduced_catalog'      Default: 0.                                       Select randomly only a fraction of the catalog (4 yrs of observation, hardcoded).
    'screen_output'        Default: 0.                                       Print the output on screen or save it into a file.
    'sigma_pv'             Default: 0.0023.                                  Uncertainty associated to peculiar velocity value, equal to (vp / c), used in the computation of the GW redshift uncertainty (0.0015 in https://arxiv.org/abs/1703.01300).
    'single_z_from_GW'     Default: 0.                                       Impose a single host for each GW having redshift equal to z_true. It works only if one_host_sel = 1.
    'snr_range'            Default: ''.                                      Impose low-high cutoffs in SNR. It can be a single number (lower limit) or a string with SNR_min and SNR_max separated by a comma.
    'snr_selection'        Default: 0.                                       Select in SNR the N loudest/faintest (N<0/N>0) events, where N=snr_selection.
    'snr_threshold'        Default: 0.0.                                     Impose an SNR detection threshold X>0 (X<0) and select the events above (below) X.
    'truths_par'           Default: {"h": 0.673, "om": 0.315, "ol": 0.685}.  Cosmology truths values. If not specified, default values are used.
    'use_alpha'            Default: 0.                                       Use selection effects in the likelihood computation (0: off, 1: on).
    'z_gal_cosmo'          Default: 0.                                       If set to 1, read and use the cosmological redshift of the galaxies instead of the observed one.
    'z_gw_range'           Default: '1000.0'.                                Impose low-high cutoffs in redshift. It can be a single number (upper limit) or a string with z_min and z_max separated by a comma.
    # sampler settings
    # 'nlive'                Default: 1000.                                    Number of live points.
    # 'seed'                 Default: 0.                                       Random seed initialisation.
    # 'pytorch_threads'      Default: 1.                                       Number of threads that pytorch can use.
    # 'n_pool'               Default: None.                                    Threads for evaluating the likelihood.
    # 'checkpoint_int'       Default: 21600.                                   Time interval between sampler periodic checkpoint in seconds. Defaut: 21600 (6h).

"""

def main():
    """Main function to be called when cosmoLISA is executed."""
    run_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    args = parser.parse_args()

    config_path = Path(args.config_file)
    if not config_path.exists():
        parser.error(f"Config file {config_path} not found.")
    
    Config = configparser.ConfigParser()
    read_ok = Config.read(config_path)
    if not read_ok:
        parser.error(f"Failed to read config file {config_path}.")

    SECTION = "input parameters"
    if SECTION not in Config:
        parser.error(f"Config file missing '{SECTION}' section.")

    config_par = {
        'com_vol_mode': 0,
        'data': '',
        'dl_gw_true': 0,
        'equal_wj': 0,
        'event_ID_list': '',
        'event_class': '',
        'max_hosts': 0,
        'model': '',
        'one_host_sel': 0,
        'outdir': "./default_dir",
        'postprocess': 0,
        'prior_bounds': {"h": [0.6, 0.86], "om": [0.04, 0.5]},
        'random': 0,
        'reduced_catalog': 0,
        'screen_output': 0,
        'sigma_pv': 0.0023,
        'single_z_from_GW': 0,
        'snr_range': '',
        'snr_selection': 0,
        'snr_threshold': 0.0,
        'truth_par': {"h": 0.673, "om": 0.315, "ol": 0.685},
        'use_alpha': 0,
        'z_gal_cosmo': 0,
        'z_gw_range': "0.0",
        # sampler settings
        'nlive': 1000,
        'seed': 1234,
        'pytorch_threads': 1,
        'n_pool': 1,
        'checkpoint_int': 10800,
    }

    def _get(sect, key, default):
        if key not in Config[sect]:
            return default
        val = Config.get(sect, key)
        if key in ('truth_par', 'prior_bounds'):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON for '{key}' in [{sect}]")
        t = type(default)
        if t is bool:
            return Config.getboolean(sect, key)
        if t is int:
            return Config.getint(sect, key)
        if t is float:
            return Config.getfloat(sect, key)
        return val
    
    for k, v in list(config_par.items()):
        config_par[k] = _get(SECTION, k, v)

    outdir = Path(str(config_par.get('outdir', './default_dir')))
    (outdir / "nessai").mkdir(parents=True, exist_ok=True)
    (outdir / "Plots").mkdir(parents=True, exist_ok=True)

    dst_Config = outdir / config_path.name
    try:
        if config_path.resolve() != dst_Config.resolve():
            shutil.copy(config_path, dst_Config)
    except FileNotFoundError:
        pass
    output_sampler = str(outdir / "nessai")


    # ---------------- logging setup ----------------
    log = logging.getLogger("cosmolisa")
    log.setLevel(logging.INFO)

    # File handler (always write a log file)
    fh = logging.FileHandler(outdir / "run.log")
    fh.setLevel(logging.INFO)

    # Optional console handler if screen_output is truthy
    ch = None
    if bool(config_par.get('screen_output', 0)):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    if ch:
        ch.setFormatter(fmt)
        log.addHandler(ch)

    # ---------------- header / environment ----------------
    sep = "=" * 24
    log.info("\n%s\nRunning cosmoLISA\n%s", sep, sep)

    # Try to report nessai version if available
    try:
        import nessai 
        log.info("nessai version: %s", getattr(nessai, "__version__", "unknown"))
    except Exception as e:
        log.info("nessai not importable at this point (%s); continuing.", e)

    import cosmolisa.likelihood as lk
    log.info("cosmolisa likelihood module path: %s", lk.__file__)

    # ---------------- config echo ----------------
    log.info("Reading config file: %s", str(config_path))
    # Nicely formatted key/value dump (sorted by key for stable output)
    pretty_cfg = pformat(dict(sorted(config_par.items())), width=120, compact=True)
    log.info("Parsed configuration:\n%s", pretty_cfg)


    truths = {
        'h': 0.673,
        'om': 0.315,
        'ol': 0.685,
        'w0': -1.0,
        'w1': 0.0,
        }

    user_truths = config_par.get('truth_par', {})
    if isinstance(user_truths, dict):
        truths.update(user_truths)
    else:
        log.warning("Unexpected type for 'truth_par' in config: %r", type(user_truths))

    # Pretty print to the log
    log.info("\n%s\nTruth cosmology parameters:\n%s", sep, sep)
    for key in sorted(truths):
        log.info("  %-10s : %s", key, truths[key])
    log.info("%s\n", sep)

    omega_true = cs.CosmologicalParameters(truths['h'], truths['om'],
                                           truths['ol'], truths['w0'],
                                           truths['w1'])


    ###################################################################
    ### Reading the catalog according to the user's options.
    ###################################################################

    # Choose between dark or bright siren analysis
    if (config_par['event_class'] == "dark_siren"):
        events = readdata.read_dark_siren_event(
            config_par['data'],
            max_hosts=config_par['max_hosts'],
            snr_selection=config_par['snr_selection'],
            sigma_pv=config_par['sigma_pv'],
            z_gw_range=config_par['z_gw_range'],
            one_host_selection=config_par['one_host_sel'],
            z_gal_cosmo=config_par['z_gal_cosmo'],
            event_ID_list=config_par['event_ID_list'],
            snr_range=config_par['snr_range'],
            snr_threshold=config_par['snr_threshold'],
            reduced_cat=config_par['reduced_catalog'],
            single_z_from_GW=config_par['single_z_from_GW'],
            equal_wj=config_par['equal_wj'],
            omega_true=omega_true,
            logger=log)
    elif (config_par['event_class'] == "MBHB"):
        events = readdata.read_MBHB_event(config_par['data'], logger=log)
    else:
        log.info(f"Unknown event_class '{config_par['event_class']}'."
              " Exiting.\n")
        exit()

    if (len(events) == 0):
        log.info("The passed catalog is empty. Exiting.\n")
        exit()

    if (config_par['random'] != 0):
        events = readdata.pick_random_events(events, config_par['random'], logger=log)

    ###################################################################
    ### Modifying the event properties according to the user's options.
    ###################################################################

    gal_interp = {}
    for e in events:
        z_grid, mixture = lk.build_interpolant(e)
        z_grid = np.ascontiguousarray(z_grid, dtype=np.float64)
        mixture = np.ascontiguousarray(mixture, dtype=np.float64)
        gal_interp[str(e.ID)] = (z_grid, mixture)
        log.info(f"Built interpolant for event {e.ID}")

    events.sort(key=lambda e: e.ID)
    # = sorted(events, key=lambda e: getattr(e, 'ID'))

    log.info("\n%s\nDetailed list of the %d selected event(s):\n%s",
            sep, len(events), sep)

    for e in events:
        # Build pieces, then join with ' | ' so it stays readable
        parts = [
            f"ID: {e.ID:>3}",
            f"SNR: {e.snr:.2f}",
            f"z_true: {e.z_true:.5f}",
            f"dl: {e.dl:.3f} Mpc",
            f"dl scat: {e.dl_scat:.3f} Mpc",
            f"dl_true_host: {e.dl_true_host:.3f} Mpc",
            f"dl - dl_true_host: {(e.dl - e.dl_true_host):.3f} Mpc",
            f"sigmadl: {e.sigmadl:.3f} Mpc",
            f"hosts: {len(e.potential_galaxy_hosts)}",
            f"zmin: {e.zmin:.5f}",
            f"zmax: {e.zmax:.5f}",
        ]
        if config_par['event_class'] == "MBHB":
            # Append host redshift for bright sirens
            parts.append(f"z_host: {e.potential_galaxy_hosts[0].redshift:.5f}")

        log.info(" | ".join(parts))

    #FIXME:only snr_threshold 50, 80, 100 available for now
    import h5py
    with h5py.File("./sel_eff/alpha_grids.h5", "r") as f:
        h_values = f["h_values"][:]
        Om_values = f["Om_values"][:]
        alpha_grid = f[f"alpha_thr{config_par['snr_threshold']}"][:]

    i0 = np.argmin(np.abs(h_values - truths['h']))
    j0 = np.argmin(np.abs(Om_values - truths['om']))
    alpha_grid = alpha_grid / max(alpha_grid[i0, j0], 1e-12)

    alpha = RegularGridInterpolator(
        (h_values, Om_values),
        np.clip(alpha_grid, 1e-12, None),
        bounds_error=False,
        fill_value=np.nan)
    
    log.info("\n%s\nnessai will be initialised with:", sep)
    log.info("  nlive:                   %s", config_par["nlive"])
    log.info("  pytorch_threads:         %s", config_par["pytorch_threads"])
    log.info("  n_pool:                  %s", config_par["n_pool"])
    log.info("  periodic_checkpoint_int: %s", config_par["checkpoint_int"])
    log.info("%s\n", sep)

    C = CosmologicalModel(
        model=config_par['model'],
        data=events,
        gal_interp=gal_interp,
        truths=truths,
        prior_bounds=config_par['prior_bounds'],
        snr_threshold=config_par['snr_threshold'],
        z_threshold=float(config_par['z_gw_range']),
        event_class=config_par['event_class'],
        com_vol_mode=config_par['com_vol_mode'],
        dl_gw_true=config_par['dl_gw_true'],
        alpha=alpha,
        use_alpha=bool(config_par['use_alpha']),
        logger=log,
        )

    # FIXME: add all the settings options of nessai.
    if (config_par['postprocess'] == 0):
        sampler = FlowSampler(
            C,
            nlive=config_par['nlive'],
            pytorch_threads=config_par['pytorch_threads'],
            n_pool=config_par['n_pool'],
            seed=config_par['seed'],
            output=output_sampler,
            checkpoint_interval=config_par['checkpoint_int'],
            )

        sampler.run()
        log.info("\n"+sep+"\n")

        x = sampler.posterior_samples.ravel()

        # Save git info.
        with open("{}/git_info.txt".format(outdir), 'w+') as fileout:
            subprocess.call(['git', 'diff'], stdout=fileout)
 
        # Save content of installed files.
        files_to_save = []
        files_path = lk.__file__.replace(lk.__file__.split("/")[-1], "")
        for te in os.listdir(files_path):
            if not ".so" in te and not "__" in te:
                files_to_save.append(te)
        files_to_save.sort()
        output_file = open(os.path.join(outdir, "installed_files.txt"), 'w')
        for fi in files_to_save:
            f = open(f"{files_path}/{fi}", 'r')
            output_file.write("____________________\n")
            output_file.write(f"{fi}\n____________________\n")
            output_file.write(f.read())
            output_file.write(10*"\n")
    else:
        log.info(f"Reading the .h5 file... from {outdir}")
        import h5py
        filename = os.path.join(outdir,"raynest", "results.json")
        h5_file = h5py.File(filename, 'r')
        x = h5_file['combined'].get('posterior_samples')

    ###################################################################
    ###################          MAKE PLOTS         ###################
    ###################################################################

    params = [m for m in C.model if m not in ['GW']]

    if (len(params) == 1):
        plots.histogram(x, par=params[0],
                        truths=truths, outdir=outdir)
    else:
        plots.corner_plot(x, pars=params,
                          truths=truths, outdir=outdir)
        log.info("Making corner plots...")

    # Compute the run-time.
    if (config_par['postprocess'] == 0):
        run_time = (time.perf_counter() - run_time)/60.0
        log.info("\nRun-time (min): {:.2f}\n".format(run_time))


if __name__=='__main__':
    main()
