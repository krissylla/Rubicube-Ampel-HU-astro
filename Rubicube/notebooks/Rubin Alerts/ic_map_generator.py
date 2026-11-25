import sys
sys.path.insert(1, '/Users/cristozilleruelo/Rubicube-Ampel-HU-astro/Rubicube/')

from astropy.time import Time
import astropy.units as u
import numpy as np
import healpy as hp
import mhealpy as mhp


def gaus_prob(dec, ra, dec_in, ra_in, one_sig_err):
        ''' 
        Adapted from https://github.com/icecube/realtime/blob/185383567df283e768e5944985306528ae0be3d3/realtime_scripts/resources/scripts/kafka_gcn_trackalert_test.py#L143
            Credit: Erik Blaufus and Angela Zegarelli
        '''
        #p(x)=1/(2πσ²)D/2 * exp(-||x||²/2σ²)
        ang_sep = ang_dist(dec, ra, dec_in, ra_in)
        prob = (1/(2*np.pi*one_sig_err**2)) * np.exp(-1.0*(ang_sep**2)/(2*one_sig_err**2))
        return prob

def IndexToDeclRa(nside, index):
    ''' 
    Adapted from https://github.com/icecube/realtime/blob/185383567df283e768e5944985306528ae0be3d3/realtime_scripts/resources/scripts/kafka_gcn_trackalert_test.py#L143
    Credit: Erik Blaufus and Angela Zegarelli
    '''
    theta,phi=hp.pixelfunc.pix2ang(nside,index)
    #return -np.degrees(theta-np.pi/2.),np.degrees(np.pi*2.-phi)
    return -(theta-np.pi/2.), phi

def ang_dist(dec1,ra1,dec2,ra2):
    '''
        Adapted from https://github.com/icecube/realtime/blob/185383567df283e768e5944985306528ae0be3d3/realtime_scripts/resources/scripts/kafka_gcn_trackalert_test.py#L143
        Credit: Erik Blaufus and Angela Zegarelli
    '''
    a = (np.sin((dec2 - dec1)/2.0))**2
    b = np.cos(dec1)*np.cos(dec2)
    c = (np.sin((ra2 - ra1)/2.0))**2
    ang_diff = 2.0*np.arcsin(np.sqrt((a + b * c)))
    return ang_diff


def generate_map(alert=None, alert_type ="Gold", nside=512, save_path=None):
#from mhealpy import HealpixBase, HealpixMap
    '''
    Adapted from https://github.com/icecube/realtime/blob/185383567df283e768e5944985306528ae0be3d3/realtime_scripts/resources/scripts/kafka_gcn_trackalert_test.py#L143
    Credit: Erik Blaufus and Angela Zegarelli

    alert: dict: {
    'name': 
    'ra': 
    'dec':
    'ra_dec_error': 
    't_mjd': 
    'stream': 'GOLD' or 'BRONZE',
    }
    '''
    if alert is None:
            return None
    dec_in = np.radians(alert['dec'])
    theta_in = -dec_in + np.pi/2.0
    ra_in  = np.radians(alert['ra'])
    one_sig_err = np.radians(alert['ra_dec_error'])  # in radians

    npix = hp.nside2npix(nside) # Calculate total number of pixels for given nside

    # Create an empty map (e.g., filled with zeros)
    healpix_map = np.zeros(npix)
    # create a MUC map as well
    mEq = mhp.HealpixBase(order = 12, scheme='RING',  coordsys='C')
    disc_pix = mEq.query_disc(mhp.ang2vec(theta_in, ra_in), np.radians(2.0))
    m = mhp.HealpixMap.moc_from_pixels(mEq.nside, disc_pix, coordsys='C',density=True)

    for i in range(npix):
        dec, ra = IndexToDeclRa(nside, i)
        healpix_map[i] = gaus_prob(dec, ra, dec_in, ra_in, one_sig_err)


    for pix in range(m.npix):
        theta,phi= m.pix2ang(pix)
        dec,ra = -(theta-np.pi/2.),phi
        m[pix] = gaus_prob(dec, ra, dec_in, ra_in, one_sig_err)

    m_normalized_map = m/np.sum(m)
    # now make into a normalized density...
    for pix in range(m.npix):
        m_normalized_map[pix] = m_normalized_map[pix]/m_normalized_map.pixarea(pix).to_value()



    #normalize to 1.0 over all pixels    
    hp_normalized_map = healpix_map/(np.sum(healpix_map)*hp.pixelfunc.nside2pixarea(nside))
    #update the headers
    _scale_90 = 2.56708542 # GFU scale factor for 50% to 90% containment
    extra_header = [
            # ('RUNID',alert['id'][0].split('_')[0]),
            # ('EVENTID',alert['id'][0].split('_')[1]),
            ('SENDER', 'Simulated AMPEL Map'),
            ('DATE-OBS', str(alert['t_mjd']),'UTC date of the observation'),
            ('MJD-OBS',  alert['t_mjd'].mjd,'modified Julian date of the observation'),
            # ('EVENT-TYPE', alert['pipeline'],'Event Type'),
            ('ALERT-STREAM', alert['stream'], 'Alert Stream'),
            ('RA', alert['ra'],'Degree'),
            ('DEC', alert['dec'],'Degree'),
            ('RA_ERR_PLUS_50', alert['ra_dec_error']/_scale_90,'50% containment error high'),
            ('RA_ERR_MINUS_50', alert['ra_dec_error']/_scale_90,'50% containment error low'),
            ('DEC_ERR_PLUS_50', alert['ra_dec_error']/_scale_90,'50% containment error high'),
            ('DEC_ERR_MINUS_50', alert['ra_dec_error']/_scale_90,'50% containment error low'),
            ('RA_ERR_PLUS_90', alert['ra_dec_error'],'90% containment error high'),
            ('RA_ERR_MINUS_90', alert['ra_dec_error'],'90% containment error low'),
            ('DEC_ERR_PLUS_90', alert['ra_dec_error'],'90% containment error high'),
            ('DEC_ERR_MINUS_90', alert['ra_dec_error'],'90% containment error low'),
            ('CONTOUR_AREA_50', np.pi*(alert['ra_dec_error']/_scale_90)**2, '50% contour area (sqdeg)'),
            ('CONTOUR_AREA_90', np.pi*(alert['ra_dec_error'])**2, '90% contour area (sqdeg)')
            ]
    
    #write out the files
    # base_filename = alert['event_name'][0] + '_skymap_probdensity'
    # std_healpix_name = base_filename + '_flat.fits.gz'
    # mhealpix_name = base_filename + '_multiorder.fits.gz'
    if save_path is None:
        save_path = f'{alert['name']}_simulated_test.fits'
    # hp.write_map('/tmp/' + std_healpix_name, hp_normalized_map, column_names=["PROB"], 
    #                 extra_header=extra_header, column_units= 'pix-1', overwrite=True)
    # hp.write_map(save_path, columnames=['PROBDENSITY'], extra_header=extra_header, column_units=['pix-1'], 
    #              overwrite=True)
    m_normalized_map.write_map(save_path, column_names=["PROBDENSITY"], 
                                extra_header=extra_header, overwrite=True)
    return m_normalized_map, save_path