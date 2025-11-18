import numpy as np
import sys
import os


class Galaxy:
    """Galaxy class:
    Initialise a galaxy defined by its redshift, redshift error,
    weight determined by its angular position relative to
    the detector posterior.
    """
    def __init__(self, redshift, dredshift, weight):
        
        self.redshift = redshift
        self.dredshift = dredshift
        self.weight = weight


class Event:
    """Event class:
    Initialise a GW event based on its properties and 
    potential galaxy hosts.
    """
    def __init__(self, **kwargs):
        self.ID = kwargs.get('ID')
        self.dl = kwargs.get('dl')
        self.dl_scat = kwargs.get('dl_scat')
        self.sigmadl = kwargs.get('sigmadl')
        self.dl_true_host = kwargs.get('dl_true_host')
        self.redshifts = kwargs.get('redshifts', [])
        self.dredshifts = kwargs.get('dredshifts', [])
        self.weights = kwargs.get('weights', [])
        self.zmin = kwargs.get('zmin')
        self.zmax = kwargs.get('zmax')
        self.snr = kwargs.get('snr')
        self.z_true = kwargs.get('z_true')
        self.z_cosmo_true_host = kwargs.get('z_cosmo_true_host')

        self.potential_galaxy_hosts = [
            Galaxy(r, dr, w) for r, dr, w in 
            zip(self.redshifts, self.dredshifts, self.weights)]
        self.n_hosts = len(self.potential_galaxy_hosts)

        if self.dl is not None and self.sigmadl is not None:
            self.dmax = self.dl + 3.0 * self.sigmadl
            self.dmin = max(self.dl - 3.0 * self.sigmadl, 0.0)
        else:
            self.dmax = None
            self.dmin = None


def read_dark_siren_event(input_folder,
                          max_hosts=0, one_host_selection=0,
                          snr_selection=0,
                          snr_threshold=0.0, sigma_pv=0.0023,
                          event_ID_list='', snr_range='', 
                          z_gw_range=0.0,
                          z_gal_cosmo=0, dl_cutoff=0.0,
                          equal_wj=0,
                          reduced_cat=0, single_z_from_GW=0, 
                          logger=None, **kwargs):
    """
    Read dark_siren data to be passed to CosmologicalModel class.

    The file ID.dat contains a single row with the following columns:
    1. Event ID
    2. True Luminosity distance dL (Mpc)
    3. Scattered luminosity distance dL_scat (Mpc)
    4. Relative error on ln luminosity distance delta{dL}/dL
    5. Observed redshift of the true host (including peculiar velocity)
    6. Minimum redshift assuming the true cosmology
    7. Maximum redshift assuming the true cosmology
    8. Fiducial redshift (i.e., the redshift corresponding to the measured distance in the true cosmology)
    9. Minimum redshift adding the cosmological prior
    10. Maximum redshift adding the cosmological prior
    11. Luminosity distance of the true host (Mpc)
    12. Signal-to-noise ratio (SNR)

    The file ERRORBOX.dat contains information about potential galaxy hosts within the error box. 
    Each row corresponds to a possible host and includes the following columns:
    1. Best luminosity distance measured by the detector (same as column 2 in ID.dat)
    2. Cosmological redshift of the host candidate (excluding peculiar velocity)
    3. Observed redshift of the host candidate (including peculiar velocity)
    4. Weight of the host candidate based on its angular position relative to the errorbox center
    """
    all_files = os.listdir(input_folder)
    logger.info(f"\nReading {input_folder}")
    events_list = [f for f in all_files if "EVENT" in f]
    pv = sigma_pv

    events = []
    for k, ev in enumerate(events_list):
        # Different catalogs have different numbers of columns,
        # so a try/except is used.
        logger.info("Reading {0} out of {1} events\r".format(
            k+1, len(events_list)))
        try:
            data = np.genfromtxt(input_folder+"/"+ev+"/ID.dat", names=True)
            event_id = data["ID"]
            dl = data["trueDL"]
            dl_scat = data["DL_scat"]
            sigma_lndl = data["delta_DL_DL"]
            z_observed_true = data["z_host"]
            zmin_true = data["z_true_min"]
            zmax_true = data["z_true_max"]
            z_true = data["z_from_DL"]
            zmin = data["zmin"]
            zmax = data["zmax"]
            dl_true_host = data["DL_host"]
            snr = data["SNR"]
        except ValueError as err:
            logger.error("ID.dat parse error in %s: %s", ev, err)
            continue

        ID = int(event_id)
        dl = np.float64(dl)
        sigmadl = np.float64(sigma_lndl)*dl # linear propagation
        dl_scat = np.float64(dl_scat)
        dl_true_host = np.float64(dl_true_host)
        zmin = np.float64(zmin)
        zmax = np.float64(zmax)
        snr = np.float64(snr)
        z_cosmo_true_host = np.float64(z_observed_true)
        z_true = np.float64(z_true)
        try:
            try:
                data = np.genfromtxt(input_folder + "/" + ev 
                                     + "/ERRORBOX.dat", names=True)
                best_dl = data["trueDL"]
                zcosmo = data["zcosmo"]
                zobs = data["zobs"]
                weights = data["wsky"]
            except ValueError as err:
                logger.info(err)
                return None
            if not z_gal_cosmo:
                redshifts = np.atleast_1d(zobs)
            else:
                redshifts = np.atleast_1d(zcosmo)
            d_redshifts = pv * np.ones(len(redshifts))
            weights = np.atleast_1d(weights)
            events.append(Event(
                ID=ID,
                dl=dl,
                dl_scat=dl_scat,
                sigmadl=sigmadl,
                dl_true_host=dl_true_host,
                redshifts=redshifts,
                dredshifts=d_redshifts,
                weights=weights,
                zmin=zmin,
                zmax=zmax,
                snr=snr,
                z_true=z_true,
                z_cosmo_true_host=z_cosmo_true_host
            ))
        except ValueError as err:
            logger.error(f"Event {event_id} at a distance {dl} (error {sigmadl})"
                    f" has no hosts because of a parsing error: {err}\n")

    if (snr_selection != 0):
        new_list = sorted(events, key=lambda x: getattr(x, 'snr'))
        if (snr_selection > 0):
            events = new_list[:snr_selection]
        else:
            events = new_list[snr_selection:]
        logger.info(f"\nSelected {len(events)} events from SNR={events[0].snr}" 
                f" to SNR={events[-1].snr}:")            
        for e in events:
            logger.info("ID: {}  |  SNR: {}".format(str(e.ID).ljust(3),
                                                str(e.snr).ljust(9)))

    if (float(z_gw_range) != 0.0):
        if (',' in z_gw_range):
            z_hor_min, z_hor_max = z_gw_range.split(',')
        else:
            z_hor_min = 0.0
            z_hor_max = float(z_gw_range)
        events = [e for e in events 
                    if (z_hor_min <= e.z_true <= z_hor_max)]
        events = sorted(events, key=lambda x: getattr(x, 'z_true'))
        if (len(events) != 0):
            logger.info(f"\nSelected {len(events)} events from"
                    f" z={events[0].z_true} to"
                    f" z={events[len(events)-1].z_true}"
                    f" (z_hor_min, z_hor_max=[{z_hor_min},{z_hor_max}]):")
            for e in events:
                logger.info("ID: {}  |  z_true: {}".format(str(e.ID).ljust(3),
                                                str(e.z_true).ljust(7))) 
        else:
            logger.info("Zero events found in the redshift window"
                    f" [{z_hor_min},{z_hor_max}].")

    if (dl_cutoff != 0):
        logger.info("\nSelecting events according to"
                f" dL(omega_true,e.zmax) < {dl_cutoff} (Mpc):")
        events = [e for e in events 
            if (kwargs['omega_true'].LuminosityDistance(e.zmax)
            < dl_cutoff)]
        logger.info("\nSelected {} events from dl={} to dl={} (Mpc)."
            .format(len(events), events[0].dl, events[len(events)-1].dl))  


    if max_hosts != 0:
        events = [e for e in events if e.n_hosts <= max_hosts]
        events = sorted(events, key=lambda x: getattr(x, 'n_hosts'))
        logger.info(f"\nSelected {len(events)} events having hosts from"
                f" n={events[0].n_hosts} to"
                f" n={events[len(events)-1].n_hosts}"
                f" (max hosts imposed={max_hosts}):")
        for e in events:
            logger.info("ID: {}  |  n_hosts: {}".format(str(e.ID).ljust(3),
                                                str(e.n_hosts).ljust(7)))

    if snr_threshold != 0.0:
        if (reduced_cat == 0):
            if (snr_threshold > 0):
                events = [e for e in events if e.snr > snr_threshold]
                logger.info("Applied SNR > %.2f", snr_threshold)
            else:
                events = [e for e in events if e.snr < abs(snr_threshold)]
                logger.info("Applied SNR < %.2f", abs(snr_threshold))
            events.sort(key=lambda e: e.snr)
            if events:
                logger.info("Selected %d events (SNR range: %.2f to %.2f).",
                            len(events), events[0].snr, events[-1].snr)
            else:
                logger.info("No events passed the SNR threshold.")
                return []
            for e in events:
                logger.info("ID: {}  |  SNR: {}".format(str(e.ID).ljust(3),
                                                str(e.snr).ljust(7)))
        else:
            # Draw a number of events in the 4-year scenario.
            N = int(np.random.poisson(len(events)*4./10.))
            logger.info(f"\nReduced number of events: {N}")
            selected_events = []
            k = 0
            while k < N and not(len(events) == 0):
                idx = np.random.randint(len(events))
                selected_event = events.pop(idx)
                logger.info("Drawn event {0}: ID={1} - SNR={2:.2f}".format(k+1,
                    str(selected_event.ID).ljust(3), selected_event.snr))
                if (snr_threshold > 0.0):
                    if (selected_event.snr > snr_threshold):
                        logger.info("Selected: ID="
                            +"{0}".format(str(selected_event.ID).ljust(3))
                            +" - SNR={0:.2f}".format(selected_event.snr)
                            +" > {0:.2f}".format(snr_threshold))
                        selected_events.append(selected_event)
                    else: pass
                    k += 1
                else:
                    if (selected_event.snr < abs(snr_threshold)):
                        logger.info("Selected: ID="
                            +"{0}".format(str(selected_event.ID).ljust(3))
                            +" - SNR={0:.2f}".format(selected_event.snr)
                            +" < {0:.2f}".format(snr_threshold))
                        selected_events.append(selected_event)
                    else: pass
                    k += 1
            events = selected_events
            events = sorted(selected_events, 
                            key=lambda x: getattr(x, 'snr'))
            logger.info("\nSelected {} events".format(len(events))
                +" from SNR={}".format(events[0].snr)
                +" to SNR={}:".format(events[len(events)-1].snr))
            for e in events:
                logger.info("ID: {}  |  dl: {}".format(str(e.ID).ljust(3),
                                                    str(e.dl).ljust(9)))

    if (event_ID_list != ''):
        print("event id list is", event_ID_list)
        ID_list = event_ID_list.split(',')
        print("ID list", ID_list)
        events = [e for e in events if str(e.ID) in ID_list]

    if (one_host_selection == 1):
        for e in events:
            z_differences = []
            for gal in e.potential_galaxy_hosts:
                z_diff = abs(e.z_true - gal.redshift)
                z_differences.append(z_diff)
                if (z_diff == min(z_differences)):
                    selected_gal = gal 
            e.potential_galaxy_hosts = [selected_gal]
        logger.info("\nUsing only the nearest host to the GW source:")
        events = sorted(events, key=lambda x: getattr(x, 'ID'))
        for e in events:
            logger.info(f"ID: {str(e.ID).ljust(3)}  |  "
                    f"SNR: {str(e.snr).ljust(9)}  |  "
                    f"dl: {str(e.dl).ljust(7)} Mpc  |  "
                    f"z_true: {str(e.z_true).ljust(7)} |  "
                    "z_nearest_host: "
                    f"{str(e.potential_galaxy_hosts[0].redshift).ljust(7)}"
                    " |  hosts:" 
                    f"{str(len(e.potential_galaxy_hosts)).ljust(4)}")
        if (single_z_from_GW == 1):
            logger.info("\nSimulating a single potential host with redshift"
            " equal to z_true.")
            for e in events:
                e.potential_galaxy_hosts[0].redshift = e.z_true
                e.potential_galaxy_hosts[0].weight = 1.0

    if equal_wj == 1:
        logger.info("\nImposing all the galaxy angular weights equal to 1.")
        for e in events:
            for g in e.potential_galaxy_hosts:
                g.weight = 1.0


    if snr_range != '':
        snr_min, snr_max = snr_range.split(',')
        events = [e for e in events if float(snr_min) <= e.snr <= float(snr_max)]


    analysis_events = events
    del events_list

    return analysis_events



def read_MBHB_event(input_folder, event_number=None, logger=None):
    """Read MBHB data to be passed to CosmologicalModel class.
    #########################################################
    If data is stored in two files:
    The file ID.dat (no header) has a single row containing:
    1-event ID
    2-luminosity distance dL (Mpc) (scattered)
    3-relative error on luminosity distance delta{dL}/dL 
        (not including propagated z error)
    The file ERRORBOX.dat (no header) contains:
    1-event redshift 
    2-absolute redshift error
    #########################################################
    If data is stored in a single file:
    The file ID.dat (with header) contains:
    1-z_true: the true binary redshift
    2-z_shifted: shifted redshift because of the error in the 
        EM measurement
    3-error_z: redshift error	
    4-d_LGpc: luminosity distance dL in Gpc
    5-d_L_shiftedGpc: shifted dL because of noise,
        lensing, and peculiar velocity errors
    6-sigma_dl_posterior1sigmaGpc: error from the posterior
        distribution
    7-sigma_dl_fisherGpc: error coming from the Fisher
    8-sigma_dl_pvGpc: error coming from the peculiar velocity
    9-sigma_dl_lensingGpc: error coming from lensing
    10-sigma_dl_combinedGpc: error obtained combining
        sigma_dl_posterior, lensing, and peculiar velocity
    11-string_index: source index
    12-LSST_detected: (boolean variable) 1 if the source is observed
        by LSST, 0 otherwise
    13-SKA_detected: the same as above (SKA)
    14-Athena_detected: the same as above (Athena)
    """
    all_files = os.listdir(input_folder)
    logger.info("Reading %s", input_folder)
    events_list = [f for f in all_files if "EVENT" in f]
    analysis_events = []

    # Two-file format
    try:
        if event_number is None:
            for k, ev in enumerate(events_list):
                logger.info("Reading %d / %d", k+1, len(events_list))
                with open(os.path.join(input_folder, ev, "ID.dat"), "r") as fh:
                    event_id, dl, rel_sigmadl = fh.readline().split(None)
                ID = int(event_id)
                dl = float(dl)
                sigmadl = float(rel_sigmadl) * dl

                errbox = os.path.join(input_folder, ev, "ERRORBOX.dat")
                try:
                    redshift, d_redshift = np.loadtxt(errbox, unpack=True)
                    redshift = np.atleast_1d(redshift)
                    d_redshift = np.atleast_1d(d_redshift)
                    weights = np.ones_like(redshift, dtype=float)
                    zmin = np.maximum(redshift - 5.0 * d_redshift, 0.0)
                    zmax = redshift + 5.0 * d_redshift
                    analysis_events.append(Event(
                        ID, dl, sigmadl, 1.0, 1.0,
                        redshift, d_redshift, weights,
                        float(zmin.min()), float(zmax.max()),
                        -1.0, -1.0, -1.0, [0]
                    ))
                except Exception as e:
                    logger.warning("Event %s (dL=%.3f, err=%.3f) has no hosts: %s",
                                   event_id, dl, sigmadl, e)
                    continue
        else:
            ev = events_list[event_number]
            with open(os.path.join(input_folder, ev, "ID.dat"), "r") as fh:
                event_id, dl, rel_sigmadl = fh.readline().split(None)
            ID = int(event_id)
            dl = float(dl)
            sigmadl = float(rel_sigmadl) * dl
            try:
                redshift, d_redshift = np.loadtxt(os.path.join(input_folder, ev, "ERRORBOX.dat"), unpack=True)
                redshift = np.atleast_1d(redshift)
                d_redshift = np.atleast_1d(d_redshift)
                weights = np.ones_like(redshift, dtype=float)
                zmin = np.maximum(redshift - 10.0 * d_redshift, 0.0)
                zmax = redshift + 10.0 * d_redshift
                analysis_events = [Event(
                    ID, dl, sigmadl, 1.0, 1.0,
                    redshift, d_redshift, weights,
                    float(zmin.min()), float(zmax.max()),
                    -1.0, -1.0, -1.0, [0]
                )]
                logger.info("Selected event %s at dL=%.3f (err=%.3f), hosts=%d",
                            event_id, dl, sigmadl, len(redshift))
            except Exception as e:
                logger.error("Event %s at dL=%.3f (err=%.3f) has no hosts: %s",
                             event_id, dl, sigmadl, e)
                return []
    # One-file format
    except Exception:
        for k, ev in enumerate(events_list):
            logger.info("Reading %d / %d", k+1, len(events_list))
            params = np.genfromtxt(os.path.join(input_folder, ev, "ID.dat"), names=True)
            ID = int(params["ID"])
            dl = float(params["d_LGpc"]) * 1e3
            sigmadl = float(params["sigma_dl_fisherGpc"]) * 1e3
            redshift = np.atleast_1d(params["z_shifted"])
            d_redshift = np.atleast_1d(params["error_z"])
            snr = float(params["SNR"])
            weights = np.ones_like(redshift, dtype=float)
            zmin = np.maximum(redshift - 5.0 * d_redshift, 0.0)
            zmax = redshift + 5.0 * d_redshift
            analysis_events.append(Event(
                ID, dl, sigmadl, 1.0, 1.0,
                redshift, d_redshift, weights,
                float(zmin.min()), float(zmax.max()),
                snr, -1.0, -1.0, [0]
            ))

    logger.info("%d MBHB events loaded", len(analysis_events))
    return analysis_events



def pick_random_events(events, number, logger=None):
    logger.info(f"\nSelecting {number} random events for joint analysis.")
    if (number >= len(events)):
        logger.info(f"Required {number} random events, but the catalog has only"
              f" {len(events)}. Running on {len(events)} events.")
        number = len(events)
    events = np.random.choice(events, size=number, replace=False)
    return events

def sample_from_pdf(x, pdf, N_draws):
    """Generate samples from a PDF through the inversion method.

    Parameters
    ----------
    x: np.array
        Parameter to sample.
    pdf: np.array
        PDF of the parameter to sample.
    N_draws: int
        Size of the sample.

    Returns
    -------
    x_sampled: np.array
        Sample of x given pdf(x).
    """

    from scipy.interpolate import interp1d

    x = np.asarray(x, dtype=float)
    pdf = np.asarray(pdf, dtype=float)

    pdf = np.clip(pdf, 0.0, None)
    area = np.trapz(pdf, x)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("PDF has zero/invalid integral.")
    cdf = np.cumsum((pdf[:-1] + pdf[1:]) * 0.5 * np.diff(x))
    cdf = np.concatenate([[0.0], cdf])
    cdf /= cdf[-1]  # normalize to [0,1]

    inv_cdf = interp1d(cdf, x, bounds_error=False, assume_sorted=True,
                       fill_value=(x[0], x[-1]))
    u = np.random.random(N_draws)
    return inv_cdf(u)