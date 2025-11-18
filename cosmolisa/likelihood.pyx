"""
# cython: profile=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
"""
cimport cython
from libc.math cimport exp, sqrt, M_SQRT1_2, M_2_SQRTPI, pi
from cosmolisa.cosmology cimport CosmologicalParameters

#######################################################################
#                          DARK SIREN
#######################################################################


@cython.boundscheck(False) # Disable bounds checking for array accesses
@cython.wraparound(False) # Disable negative indexing for arrays
@cython.nonecheck(False)
@cython.cdivision(True) # Enables C-style division
def lk_dark_single_event_trap(const double meandl,
                              const double sigmadl,
                              CosmologicalParameters omega,
                              const double zmin,
                              const double zmax,
                              const double[::1] z_grid,
                              const double[::1] mixture,
                              int vol_mode):
    """
    Trapezoidal integration over z in [zmin, zmax] with galaxy prior from (z_grid, mixture).
    Returns likelihood (not log). All heavy work is C-level; no Python in the loop.
    """
    cdef int N = 256 
    cdef double dz = (zmax - zmin) / N
    cdef double I = 0.0
    cdef double Z = 0.0
    cdef int i
    cdef double z, w, dv, det
    cdef double z0 = z_grid[0]
    cdef double dz_grid = z_grid[1] - z_grid[0]
    cdef double inv_dz_grid = 1.0 / dz_grid
    cdef Py_ssize_t M = z_grid.shape[0]

    with nogil:
        for i in range(N + 1): # endpoints included
            z = zmin + i * dz
            w = 0.5 if (i == 0 or i == N) else 1.0
 
            z_prior = lininterp_uniform(z, z0, inv_dz_grid, mixture, M)
            if vol_mode == 0: # no comoving volume factor
                zp_num = z_prior
                zp_den = z_prior
            elif vol_mode == 1: # com vol in numerator and denominator
                dv = omega._ComovingVolumeElement(z)
                zp_num = z_prior * dv
                zp_den = z_prior * dv
            elif vol_mode == 2: # com vol only in numerator
                dv = omega._ComovingVolumeElement(z)
                zp_num = z_prior * dv
                zp_den = z_prior

            det = _detector_term(z, meandl, sigmadl, omega)

            I += w * det * zp_num
            Z += w * zp_den

    if Z <= 0.0:
        return 0.0
    # dz cancels in ratio
    return I / Z

@cython.boundscheck(False) # Disable bounds checking for array accesses
@cython.wraparound(False) # Disable negative indexing for arrays
@cython.nonecheck(False)
@cython.cdivision(True) # Enables C-style division
cdef inline double _detector_term(const double z,
                                const double meandl,
                                const double sigmadl,
                                CosmologicalParameters omega) nogil:
    cdef double dl = omega._LuminosityDistance(z)
    cdef double sig2 = sigmadl * sigmadl + _sigma_weak_lensing(z, dl)**2
    cdef double invs = 1.0 / sqrt(sig2)
    # 1/sqrt(2*pi) precomputed: M_SQRT1_2*0.5*M_2_SQRTPI
    cdef double norm = M_SQRT1_2 * 0.5 * M_2_SQRTPI * invs
    cdef double d = dl - meandl

    return norm * exp(-0.5 * d * d * (invs * invs))


#######################################################################
#                          BRIGHT SIREN
#######################################################################

def lk_bright_single_event_trap(const double[:,::1] hosts,
                            const double meandl,
                            const double sigmadl,
                            CosmologicalParameters omega,
                            str model,
                            const double zmin,
                            const double zmax):
    """Likelihood function p( Di | O, M, I) for a single bright GW 
    event of data Di assuming cosmological model M and parameters O.
    Following the formalism of <arXiv:2102.01708>.
    Use EM host data to compute the likelihood.
    Parameters:
    ===============
    hosts: :obj: 'numpy.array' with shape Nx4. The columns are
        redshift, redshift_error, angular_weight, magnitude
    meandl: :obj: 'numpy.double': LISA mean of the luminosity distance dL
    sigmadl: :obj: 'numpy.double': LISA standard deviation of dL
    omega: :obj: 'lal.CosmologicalParameter': cosmological parameter
        structure O
    event_redshift: :obj: 'numpy.double': redshift for the GW event
    zmin: :obj: 'numpy.double': minimum GW redshift
    zmax: :obj: 'numpy.double': maximum GW redshift
    """

    return _lk_bright_single_event_trap(hosts, meandl, sigmadl, omega,
                                      model, zmin, zmax)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _lk_bright_single_event_trap(const double[:,::1] hosts,
                            const double meandl,
                            const double sigmadl,
                            CosmologicalParameters omega,
                            str model,
                            const double zmin,
                            const double zmax):

    cdef int i
    cdef int N = 100
    cdef double dz = (zmax-zmin)/N
    cdef double z  = zmin + dz
    cdef double I = (0.5
        * (_lk_bright_single_event_integrand_trap(zmin, hosts, meandl,
                                                sigmadl, omega, model)
        + _lk_bright_single_event_integrand_trap(zmax, hosts, meandl,
                                               sigmadl, omega, model)))
    for i in range(1, N):
        I += _lk_bright_single_event_integrand_trap(z, hosts, meandl,
                                                  sigmadl, omega, model)
        z += dz
    return I*dz

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _lk_bright_single_event_integrand_trap(
                                        const double event_redshift,
                                        const double[:,::1] hosts,
                                        const double meandl,
                                        const double sigmadl,
                                        CosmologicalParameters omega,
                                        str model) nogil:

    cdef double dl
    cdef double L_EM = 0.0
    cdef double L_detector = 0.0
    cdef double sigma_z, score_z
    cdef double OneOverSqrtTwoPi = M_SQRT1_2*0.5*M_2_SQRTPI

    # GW likelihood: N(dl - meandl; sigmadl^2)
    dl = omega._LuminosityDistance(event_redshift)
    cdef double weak_lensing_error = _sigma_weak_lensing(event_redshift, dl)
    cdef double SigmaSquared = sigmadl**2 + weak_lensing_error**2
    cdef double SigmaNorm = OneOverSqrtTwoPi * 1/sqrt(SigmaSquared)
    L_detector = (SigmaNorm * exp(-0.5*(dl-meandl)*(dl-meandl)
                  / SigmaSquared))

    # Redshift prior: N(z - zEM; sigmaz^2)
    # Read sig_z_EM from EM data.
    sigma_z = hosts[0,1]

    score_z = (event_redshift - hosts[0,0])/sigma_z
    L_EM = (OneOverSqrtTwoPi * (1/sigma_z)
                * exp(-0.5*score_z*score_z))
    
    return L_detector * L_EM


##########################################################
#                                                        #
#                   Other functions                      #
#                                                        #
##########################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double lininterp_uniform(double z, double z0, double inv_dz,
                                     const double[:] y, Py_ssize_t n) nogil:
    # clamp
    if z <= z0:
        return y[0]
    cdef double t = (z - z0) * inv_dz      # fractional index
    if t >= n - 1:
        return y[n - 1]
    cdef Py_ssize_t i = <Py_ssize_t> t
    cdef double frac = t - i
    return (1.0 - frac) * y[i] + frac * y[i + 1]


cpdef build_interpolant(event):
    """Build an interpolant for the galaxy redshift prior of the event.
    Parameters
    ----------
    event: Event
        The event for which to build the interpolant.
    Returns
    -------
    interpolant: function
        A function of the galaxy weighted mixture model.
    """
    import numpy as np

    cdef int M = 2000
    cdef double zmin = event.zmin
    cdef double zmax = event.zmax

    z_grid = np.linspace(zmin, zmax, M, dtype=np.float64)    
    mixture = np.zeros(M, dtype=np.float64)

    cdef int i
    cdef double sigma_z, score_z, p_gal, z
    cdef double sqrt2pi = sqrt(2.0 * pi)

    for gal in event.potential_galaxy_hosts:
        sigma_z = gal.dredshift * (1.0 + gal.redshift)
        if sigma_z <= 0.0:
            continue
        for i in range(M):
            z = z_grid[i]
            score_z = (z - gal.redshift) / sigma_z
            p_gal = (gal.weight / (sqrt2pi * sigma_z)) * exp(-0.5 * score_z * score_z)
            mixture[i] += p_gal

    return z_grid, mixture


def sigma_weak_lensing(const double z, const double dl):
    return _sigma_weak_lensing(z, dl)

cdef inline double _sigma_weak_lensing(const double z, 
                                       const double dl) nogil:
    """Weak lensing error. From <arXiv:1601.07112v3>,
    Eq. (7.3) corrected by a factor 0.5 
    to match <arXiv:1004.3988v2>.
    Parameters:
    ===============
    z: :obj:'numpy.double': redshift
    dl: :obj:'numpy.double': luminosity distance
    """
    cdef double t = 1.0 - (1.0+z)**(-0.25)
    return 0.5*0.066*dl*(t/0.25)**1.8


cpdef double find_redshift(CosmologicalParameters omega, double dl):
    from scipy.optimize import newton
    return newton(objective, 1.0, args=(omega,dl))

cdef double objective(double z, CosmologicalParameters omega, double dl):
    return dl - omega._LuminosityDistance(z)
