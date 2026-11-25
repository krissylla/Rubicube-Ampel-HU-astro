import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from scipy.special import factorial


import sys
sys.path.insert(1, '/Users/cristozilleruelo/Rubicube-Ampel-HU-astro/Rubicube')

from transient_lumonosity import  get_phi_from_phi_z1
from params import p, p_transient
from ic_processing import effective_area



def read_luminosity_from_csv(path_csv, k):

    '''
    Reads trhe comp[uted luyminosity model from the csv file. It uses the functions in transient_luminosity to compute his originally
    '''
    df = pd.read_csv(path_csv)
    i = list(df['Transient']).index(k)
    luminosity = {
        'R0' : df['R0 (1 / Gpc^3 yr)'][i] * u.Gpc**-3  * u.yr**-1, 
        'gamma': df['gamma'][i],
        'phi_z1': df['phi_z1 (1 / GeV s cm^2)'][i] * u.GeV ** -1 * u.s ** -1 * u.cm ** -2,
        'E_nu_iso': df['E_nu_iso (erg / s)'][i] * u.erg * u.s ** -1,
    }
    return luminosity


def get_n_jk(E_i, z, k, dec, p=p, p_transient=p_transient):

    '''
    TODO: n_jk is in units of 1 / GeV. To do this correclty, I need to integrate within 
        an energy range. For now, I multiply by the neutrino energy to obtain a value


    Estimated number of neutrinos within a given time window dependent on the transient type:

    dN/dE = phi(z) * (E / E_0) ** gamma * T_k 

    Note: The formula should have an effective area term, A_eff, ihn cm^2 to fully return as unitless number. However,m since this term will also appear in the 
        Denomiator in the background signal term it will get cancelled out, so for now we don't include it. 

    Params: 
    -------
    E_i: astropy.Quantity, neutrino energy in GeV
    z: float, redshift of the source
    k: str, transient type of the source. 

    Returns
    -------
    n_kj = float, estimated number of neutrinos from source j assuming type k, in the relevant time window. 
    '''
    path_csv = p['neutrino_luminosity_path']
    transient_model = read_luminosity_from_csv(path_csv, k)
    phi_z = get_phi_from_phi_z1(z, 
                                transient_model['phi_z1'], 
                                gamma=transient_model['gamma'])
    T_k = p_transient['T_k'][k].to(u.s) #as phi should be in s^-1

    A_eff = effective_area(E_i, dec) #in m^2

    #TODO: TEMPORAL FIX OF THE 1/GeV UNIT PORBLEM
    n_kj = E_i * phi_z * (E_i.to(u.GeV) / p_transient['nu_diff_E0'].to(u.GeV)) ** transient_model['gamma'] * T_k * A_eff.to(u.cm**2)
    # n_kj = phi_z * (E_i.to(u.GeV) / p_transient['nu_diff_E0'].to(u.GeV)) ** transient_model['gamma'] * T_k * A_eff.to(u.cm**2)
    
    return n_kj



def Poission(lam, k):
    return lam ** k * np.exp(-1 * k) / factorial(k)


    
##### PDFs

def spatial_signal_pdf(neutrino, transient):
    '''
    Reads in the spatial probability from the neutrino map
    
    '''

    p_spatial = neutrino.get_spatial_probabilty(transient)

    return p_spatial

def temporal_signal_pdf(neutrino, transient, k, p=p_transient):
    '''
    Uniform temporal probability of the neutrino arrival time. 
    p= 1/ delta t if t_nu in [t_peak]
    '''
    delta_t = p['T_k'][k].to(u.s) #as phi should be in s^-1
    t_nu = neutrino.t
    t_peak = transient.t_peak #both should be a Time oject

    if t_peak - delta_t/2 <= t_nu <= t_peak + delta_t/2:
        p_temporal = 1 / delta_t
    else: 
        p_temporal = 0

    return p_temporal

def emission_signal_pdf(neutrino, transient, k):

    '''
    Computes the posission probability of detecting 1 neutrino assuming a Poission probaility of mean = expected number of neutrinos 
        assuming the transient is of type k
    '''

    E_i = neutrino.E_i
    z = transient.z
    dec = transient.dec

    n_jk = get_n_jk(E_i, z, k, dec)

    print("N_jk: ", n_jk)
    p_emission =Poission(lam=n_jk, k=1)

    return p_emission