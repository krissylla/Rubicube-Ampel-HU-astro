from astropy.time import Time
import astropy.units as u

import sys
sys.path.insert(1, '/Users/cristozilleruelo/Rubicube-Ampel-HU-astro/Rubicube/')

from ic_visualization import read_map, get_moc_map_metadata
from ic_processing import evaluate_map


class Neutrino:
    
    def __init__(self, alert_info, E_i):

        self.E_i = E_i

        self.map_name = alert_info['alert_name']
        self.skymap = alert_info['skymap']
        self.t = Time(alert_info['header']['DATE-OBS'])

        self.ra = None
        self.dec = None
        self.metadata = None

        #exclusion box
        self.ra_box = None
        self.dec_box = None

        self.load_map(alert_info)
        self.set_exclusion_box()


    
    def load_map(self, alert_info):

        '''
        Alert info is the product of ic_processing.read_map
        '''

        metadata = get_moc_map_metadata(alert_info['skymap'])
        self.ra = metadata['RA']
        self.dec = metadata['DEC']
        self.metadata

        return None

    def set_exclusion_box(self):
        '''
        Sets a ra and dec box such around the best fit location:
        that if a transient is further from it then the spatial probability is 0.
        This is done to save computational time. 

        TODO: set it from the average RA_DEC_err, I.e 10* error
        '''  
        
        self.ra_box = [self.ra - 10 * u.deg, self.ra + 10 * u.deg]
        self.dec_box = [self.dec - 10 * u.deg, self.dec + 10 * u.deg]
        


    def get_spatial_probabilty(self, transient):
        '''
        Evaluates the source on the map
        '''
        if transient.ra < self.ra_box[0] or transient.ra > self.ra_box[1] or transient.dec < self.dec_box[0] or transient.dec > self.dec_box[1]:
            return 0
        
        p_spatial = evaluate_map(self.skymap, transient.ra, transient.dec)

        return p_spatial

    def __repr__(self):
        '''
        We pring the neutrino parameters
        '''

        string = f"Neutrino {self.map_name} \nEnergy: {self.E_i} \nt_nu: {self.t}"
        return string
        