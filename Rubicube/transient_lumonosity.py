import sys
from params import p, p_cosmology
from transient_rates import R_SFR, get_N_k_firesong

import numpy  as np
import matplotlib.pyplot as plt
import astropy.units as u
import scipy
import astropy.cosmology as acosmo

def rate_integral(phi_z, z_max=10, p=p_cosmology):  #, t_obs=1*u.yr):

    cosmo = acosmo.LambdaCDM(H0=p['h0']* u.km / u.s / u.Mpc, 
                            Om0=p['omega_M'],
                            Ode0=p['omega_L'])#, Tcmb0=2.725 * u.K,


    z_range = np.linspace(p['z_min'], z_max, 10000)
    dVdz = cosmo.differential_comoving_volume(z_range)
    y = phi_z(z_range) * dVdz

    return scipy.integrate.trapezoid(y, z_range)

def get_phi_z1(phi_z, R0, z_max=10, p=p_cosmology, gamma=p_cosmology['nu_diff_gamma'], t_obs=1*u.yr, debug=False):
    R0 = R0.to(u.Mpc**-3 * u.yr ** -1) 
    cosmo = acosmo.LambdaCDM(H0=p['h0']* u.km / u.s / u.Mpc, 
                            Om0=p['omega_M'],
                            Ode0=p['omega_L'])#, Tcmb0=2.725 * u.K,
    norm = rate_integral(phi_z, z_max=z_max, p=p)#, t_obs=t_obs)

    #source integral
    z_range = np.linspace(p['z_min'], z_max, 100000)
    dVdz = cosmo.differential_comoving_volume(z_range)
    d_L = cosmo.luminosity_distance(z_range).to(u.cm)
    dNdz = phi_z(z_range)

    

    y = (1 + z_range) ** (gamma + 2) / (d_L ** 2) * dNdz * dVdz / norm


    source_int = scipy.integrate.trapezoid(y, z_range)

   
    #combine all
    nu_diff_phi = p['nu_diff_phi'] *u.GeV ** -1 * u.cm ** -2 * u.s ** -1 * u.sr ** -1#GeV/cm2/s/sr.

    
    N_tot_z1 = get_N_k_firesong(phi_z, R0, z_max=1, time_window=t_obs)
    d_L_z1 = cosmo.luminosity_distance(1).to(u.cm)

    phi_1 = nu_diff_phi * 4 * np.pi * u.sr/ (N_tot_z1 * d_L_z1 ** 2 * source_int) 


    if debug:
        print('dVdz: ', dVdz)
        print('d_L: ', d_L)
        print('dNdz: ', dNdz)
        print('y: ', y)
        print('source_int: ', source_int)
        print('d_L_z1: ', d_L_z1**2)

    return phi_1

def get_phi(z, phi_z, R0, gamma=p_cosmology['nu_diff_gamma'], z_max=10, p=p_cosmology, t_obs=1*u.yr):
    '''
    Obtains the neutrino flux normalization of a source at a redshift z
    '''
    phi_1 = get_phi_z1(phi_z, R0, z_max=z_max, p=p, t_obs=t_obs)

    cosmo = acosmo.LambdaCDM(H0=p['h0']* u.km / u.s / u.Mpc, 
                            Om0=p['omega_M'],
                            Ode0=p['omega_L'])#, Tcmb0=2.725 * u.K,
    d_L_z1 = cosmo.luminosity_distance(1)
    d_L = cosmo.luminosity_distance(z)


    phi_z = phi_1 * (d_L_z1 / d_L) ** 2 * (1+z) ** (gamma + 2)

    return phi_z


def get_phi_from_phi_z1(z, phi_1, gamma=p_cosmology['nu_diff_gamma'],  p=p_cosmology):
    '''
    Obtains the neutrino flux normalization of a source at a redshift z from the normalization phi_z1 (obtained from get_[hi_z1]
    '''
    # phi_1 = get_phi_z1(phi_z, R0, z_max=z_max, p=p, t_obs=t_obs)

    cosmo = acosmo.LambdaCDM(H0=p['h0']* u.km / u.s / u.Mpc, 
                            Om0=p['omega_M'],
                            Ode0=p['omega_L'])#, Tcmb0=2.725 * u.K,
    d_L_z1 = cosmo.luminosity_distance(1)
    d_L = cosmo.luminosity_distance(z)


    phi_z = phi_1 * (d_L_z1 / d_L) ** 2 * (1+z) ** (gamma + 2)

    return phi_z


def get_E_nu_iso(phi_k_1, gamma=2.53, E_0=1e5 * u.GeV, E_min=3, E_max=8, p=p_cosmology, debug=False):

    #get cosmology stuff
    cosmo = acosmo.LambdaCDM(H0=p['h0']* u.km / u.s / u.Mpc, 
                            Om0=p['omega_M'],
                            Ode0=p['omega_L'])#, Tcmb0=2.725 * u.K,
    d_L = cosmo.luminosity_distance(1).to(u.cm)
    E_range = np.logspace(E_min, E_max, 10000) * u.GeV
    I_e_func = E_range * (E_range / E_0) ** (-1 * gamma)
    I_e = scipy.integrate.trapezoid(I_e_func, E_range)

    if debug:
        print('\n I_e after int: ', I_e)

    return phi_k_1 * 4 * np.pi * d_L ** 2 * I_e

