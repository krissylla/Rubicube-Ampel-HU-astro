import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import scipy

from astropy import units as u

from params import p

def effective_area(E_nu, dec, channel='all-gold-bronze', p=p):
    '''
    Interpoolates the effective area at a given energy, declination,
    and channel type. 
    E_nu: float or array. in GeV
    dec: float or array. in degrees
    channel: string, alert channel that was issued. It can be:
            gfu-bronze, gfu-gold, hese-bronze, hese-gold, ehe-gold, 
            all-gold, all-gold-bronze. 
    p: dict. contains the genral RubiCube paremters, including the path to the effective areas folder.

    Returns:
    computed_areas: np.array, in m^2.
    '''

    
    path_eff_area = os.path.join(p['Eff_area']['dir'], p['Eff_area'][channel])
    with open(path_eff_area,  'r') as file:
        table  = pd.read_table(file)

    if isinstance(dec, float) or isinstance(dec, int):
        dec = np.array([dec])
    if isinstance(E_nu, float) or isinstance(E_nu, int):
        E_nu = np.array([E_nu])
    
    
    if isinstance(E_nu, np.ndarray) and len(dec) != len(E_nu):
        if len(dec) == 1:
            dec = np.ones(len(E_nu)) * dec
        else:
            raise ValueError('dec array should have dimension 1 or same dimensions a E_nu.')
    computed_areas = np.zeros_like(dec)

    masks = {
        ' -90--30': (dec >= -90) & (dec < -30),
        ' -30--5 ': (dec >= -30) & (dec < -5),
        ' -5-0 ': (dec >= -5)  & (dec < 0),
        ' 0-30 ': (dec >= 0)   & (dec < 30),
        ' 30-90 ': (dec >= 30)  & (dec <= 90)
    }
    for key in masks.keys():
        if np.any(masks[key]):
            area_values = table[key].values
            f = interp1d(table["#E "].values, area_values)
            # print(masks[key])
            computed_areas[masks[key]] = f(E_nu[masks[key]]) 
    return computed_areas * u.m ** 2


def signal_energy_pdf(E_i, delta_j, gamma=2.13, E_max=9, channel='all-gold-bronze', p=p):
    '''
    Computes the energy component of the signal pdf for the bayesian likelihood. 
    We assume that the population emits neutrinos folllowing a powerlaw of dN/dEdt = phi_0 * (E / 1e5 GeV) ^ -gamma. 
    The energy pdf is computed as: 
    S_energy = (E_i/E0) ^ -gamma * A_eff / (int E/E0) ^ -gamma * A_eff dE)

    Params:
    -------
    E_i: float, reconnstructed energy of the neutrino in GeV. 
    delta_j: float, dclination of the evaluated source in degree. 
    gamma: float, powerlaw of the neutrino spectrum of the source population. By default we assume 
        the single-powerlaw fit from the IceCube diffuse neutrino flux.
    channel: str, alert channel. Its used to retrieve the effective area
    p: dict, params. used to compute the effective area. 
    Returns:
    --------
    S_energy: float, signal energy pdf evaulated at the neutrino energy and souce declination. This value should be normalized
         so that \int S_energy dE = 1
    '''

    E_range = np.logspace(3, E_max, 1000) #in GeV
    E_0 = 1e5
    if isinstance(E_i, np.ndarray) and isinstance(delta_j, np.ndarray):
        S_energy = []
        for E, delta in zip(E_i, delta_j):
            A_eff_range = effective_area(E_range, delta, channel=channel, p=p).value
            y = (E_range / E_0) ** (-1 * gamma) * A_eff_range
            norm = scipy.integrate.trapezoid(y, E_range)

            A_eff_i = effective_area(E, delta, channel=channel, p=p).value
            S_energy.extend((E / E_0) ** (-1 * gamma) * A_eff_i / norm)
        S_energy = np.array(S_energy)
    else:
        A_eff_range = effective_area(E_range, delta_j, channel=channel, p=p).value
        y = (E_range / E_0) ** (-1 * gamma) * A_eff_range
        norm = scipy.integrate.trapezoid(y, E_range)

        A_eff_i = effective_area(E_i, delta_j, channel=channel, p=p).value
        S_energy = (E_i / E_0) ** (-1 * gamma) * A_eff_i / norm

    return S_energy




    

if __name__ == '__main__':
    #test if area is working correctly:
    for channel in p['Eff_area'].keys():
        if channel == 'dir':
            continue
        print(channel, ' channel: ', effective_area(1e5, 30, channel=channel), '\n')