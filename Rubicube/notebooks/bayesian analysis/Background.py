import numpy as np
import astropy.units as u
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import factorial


import sys
sys.path.insert(1, '/Users/cristozilleruelo/Rubicube-Ampel-HU-astro/Rubicube')

from params import p_bayesian_analysis, p
from ic_processing import effective_area

## utilty functions
def zenith_angle(declination):

    '''Converts declination into zenith angle for IceCube detector '''
    if  not isinstance(declination,  u.Quantity):
        declination = declination * u.deg

    return declination.to(u.deg) + 90 * u.deg 

def cos_zenith(declination):
    
    return np.cos(zenith_angle(declination).to(u.rad))


def get_nu_bkg_flux(E_nu, dec, p=p):

    '''
    Reads the atmospheric neutrino model saved in the csv file.
    Computed originally with nuflux, assuming the honda2006 model. 
    '''
    #check and set the correct units
    if not isinstance(E_nu, u.quantity.Quantity):
        E_nu = E_nu * u.GeV
    if not isinstance(dec, u.quantity.Quantity):
        dec = dec * u.degree

    path_csv = p['neutrino_bkg_model_path']
    with open(path_csv, 'r') as file:
        nu_model = pd.read_csv(file)

    #first find declination bin
    dec_bins = nu_model.columns[1:]
    for dec_bin in dec_bins:
        min_dec, max_dec = dec_bin.split(' - ')
        if float(min_dec) <= dec.value <= float(max_dec):
            correct_bin = dec_bin #+1 because first column are energy values
            break
    nu_model[[nu_model.columns[0], correct_bin]]

    nu_model[nu_model.columns[0]].values

    #now we interpolate at the correct energy
    f = interp1d(nu_model[nu_model.columns[0]].values, 
                nu_model[correct_bin].values)

    bkg_flux = f(E_nu.to(u.GeV).value)

    #flux units in GeV^-1 cm^-2 s^-1 as stated in: https://docs.icecube.aq/nuflux/main/pythonapi.html#nuflux.FluxFunction.getFlux

    return bkg_flux * u.GeV ** -1 * u.cm ** -2 * u.s ** -1


def Poission(lam, k):
    return lam ** k * np.exp(-1 * k) / factorial(k)



### Pdfs

def spatial_bkg_pdf():
    '''
    Probability of obtaining a neutrino in this particular region of the sky. 
    We assume a uniform distribution'''
    return 1 / (4 * np.pi)
    

def temporal_bkg_pdf(p=p_bayesian_analysis):
    '''
    Represents the probability of obtaining an alert in the given time window. We define it as 
    one over the time windoe in which we search for active transients/
    '''
    t_obs = p['t_obs'].to(u.s)
    p_temporal_bkg = 1 / t_obs #1 / time window where we search for trans
    return p_temporal_bkg

def get_n_bkg(neutrino, transient, p=p_bayesian_analysis, debug=False):
    '''
    We read in the nuflux values from the preivously computed csv file

    TODO: Same issue as in the signal emission prob, where n_bkg is in units 
        of 1/GeV. For now, we multiply by the neutrino energy but we need a better solution
    
    '''

    dec = transient.dec #in degrees
    transient_cos_zenith = cos_zenith(dec)

    E_i = neutrino.E_i

    t_obs = p['t_obs'].to(u.s)

    A_eff = effective_area(E_i, dec).to(u.cm**2)

    #get bkg model
    # flux, nu_type = get_bkg_flux_model(p=p)
    flux_at_dec = get_nu_bkg_flux(E_i.to(u.GeV), dec)
    #get expeced number of backround events
    # n_bkg = flux_at_dec * t_obs * A_eff

    #1/GeV fix for now:
    n_bkg = flux_at_dec * t_obs * A_eff * E_i


    if debug: 
        print("\nDebug emission_bkf_pdf")
        print('dec: ', dec)
        print('transient_cos_zenith: ', transient_cos_zenith)
        print('E_i: ', E_i)
        print('t_obs: ', t_obs)
        print('A_eff: ', A_eff)
        print('flux at E, dec: ', flux_at_dec)
        print('n_bkg: ', n_bkg)

    return n_bkg


def emission_bkg_pdf(neutrino, transient, p=p_bayesian_analysis, debug=False):
    '''
    TODO!!: Fix nuflux enviorment solvency
    '''
    n_bkg = get_n_bkg(neutrino, transient, p=p, debug=debug)
    p_emission_bkg = Poission(lam=n_bkg, k=1)

    if debug: 
        print('Poiss(n_bkg, 1): ', p_emission_bkg)
    return p_emission_bkg



