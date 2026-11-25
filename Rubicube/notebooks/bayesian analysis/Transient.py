import astropy.units as u
from astropy.time import Time


class Transient:

    # transient_params: dict | None

    def __init__(self, transient_params):
        self.ra = transient_params['ra']
        self.dec = transient_params['dec']
        self.z = transient_params['z']

        self.p_k = transient_params['p_k']

        self.t_peak = Time(transient_params['t_peak'])

        self.name = transient_params['name']

        self.check_units()

    def check_units(self):
        '''
        Transofrms quantities to the desired units. If no units are provided, then we assume a unit
        '''


        if not isinstance(self.ra, u.quantity.Quantity):
            self.ra = self.ra * u.deg #we assume its in degrees
        else:
            self.ra = self.ra.to(u.deg)

        if not isinstance(self.dec, u.quantity.Quantity):
            self.dec = self.dec * u.deg #we assume its in degrees
        else:
            self.dec = self.dec.to(u.deg)

        

