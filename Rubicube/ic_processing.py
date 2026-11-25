import os
import requests #for url downloading
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import mhealpy as mhp
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import QTable
import astropy_healpix as ah
from ligo.skymap.io.fits import read_sky_map
import ligo.skymap.plot #used impliciteely
import pandas as pd
from scipy.interpolate import interp1d

from params import p

import warnings #for import lal inside ligo.skymap
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")



def read_map(map_name, map_dir=p['map_dir'], add_units=False):
    '''
    Opens a map at map_path. If map_path is a url, it downloads the map in the save_dir folder
    add_units: add the units of 1/sr to the map. This is only for simulated maps
    '''

    #check if map_name is an url
    if 'https' in map_name:
        file_name = map_name.split('/')[-1]
        alert_name = file_name.split('_')[0]

        #first we download the map
        response = requests.get(map_name)
        map_path = os.path.join(map_dir, file_name)
        with open(map_path, 'wb') as file: #also deletes previously existing file and redownloads alert. 
            file.write(response.content)
    
    #if not url, then it could be a full path:
    elif os.path.exists(map_name):
        map_path = map_name
        alert_name = os.path.basename(map_path).split('_')[0]
    
    #Check if its just the filename:
    elif os.path.exists(os.path.join(map_dir, map_name)):
        map_path = os.path.join(map_dir, map_name)
        alert_name = os.path.basename(map_path).split('_')[0]
    
    #In case just the alert name like IceCube-260610A. We can check if this alert was already downloaded.
    #this will be usefull in case I want to pull up a specific alert on comand
    else:
        downloaded_maps = {alert.split('_')[0]: alert for alert in os.listdir(map_dir)}
        if map_name in downloaded_maps.keys():
            print('Opening downloaded alert: ', downloaded_maps[map_name])
            map_path = os.path.join(map_dir, downloaded_maps[map_name])
            alert_name = os.path.basename(map_path).split('_')[0]
        else:
            raise AttributeError(f'Incorrect Map name inpu: {map_name} \n. Please provide a url, map_path, or alert name in the format IceCube-YYMMDDA')


    #we open the map
    skymap = QTable.read(map_path)
    _, header = fits.getdata(map_path, header=True)

    #Check if array has masked values. We fill them with 0.0
    try:
        if skymap['PROBDENSITY'].mask.any():
            print("Map has masked values. reeplacing them with 0")
            skymap['PROBDENSITY'] = skymap['PROBDENSITY'].filled(0)
    except AttributeError:
        pass #array has no masked values

    if add_units: 
        skymap['PROBDENSITY'] = skymap['PROBDENSITY'] * u.sr ** -1
    
    map_data = {
        'skymap': skymap,
        'header': header,
        'map_path': map_path,
        'alert_name': alert_name
    }

    return map_data

def get_moc_map_metadata(skymap):
    '''
    Not sure if useful, but returns a dictionary with the metavalues for the multi-ordered map
    '''
    level, ipix = ah.uniq_to_level_ipix(skymap['UNIQ'])
    nside = ah.level_to_nside(level)
    pixel_area = ah.nside_to_pixel_area(nside)
    


    #we get ra and dec from maximum value of map
    arg_max = np.max(skymap['PROBDENSITY'])
    max_i = list(skymap['PROBDENSITY']).index(arg_max)
    max_ipix = ipix[max_i]
    ra, dec = hp.pix2ang(nside[max_i], max_ipix, lonlat=True, nest=True)
    ra = ra * u.deg
    dec = dec * u.deg

    metadata = {
            'PROBDENSITY': skymap['PROBDENSITY'],
            'UNIQ': skymap['UNIQ'],
            'LEVEL': level,
            'IPIX': ipix,
            'NSIDE': nside,
            'PIXELAREA': pixel_area,
            'RA': ra,
            'DEC': dec
        }

    return metadata


def get_p_value_contour(skymap, ax, p):

    skymap.sort("PROBDENSITY", reverse=True)
    sorted_metadata = get_moc_map_metadata(skymap)
    prob = sorted_metadata['PIXELAREA'] * skymap["PROBDENSITY"]
    cumprob = np.cumsum(prob)

    # searchsorted gives the FIRST index where cumprob >= target.
    i_p = cumprob.searchsorted(p)
    
    
    m_prob = mhp.HealpixMap(data=cumprob, uniq=skymap['UNIQ'], density=True)
    img = m_prob.get_wcs_img(ax)#, coord = coord, rasterize = rasterize)
    im = ax.contour(img, levels=[p], colors=['cyan'], linestyles=['solid'], label=f'p = {p}') 
    # ax.legend(framealpha=1)

    area_p = sorted_metadata['PIXELAREA'][:i_p].sum().to(u.deg**2)
    
    return area_p

def get_50_90_contour(skymap, ax):
    skymap.sort("PROBDENSITY", reverse=True)
    sorted_metadata = get_moc_map_metadata(skymap)
    prob = sorted_metadata['PIXELAREA'] * skymap["PROBDENSITY"]
    cumprob = np.cumsum(prob)
    # searchsorted gives the FIRST index where cumprob >= target.
    i_50 = cumprob.searchsorted(0.5)
    i_90 = cumprob.searchsorted(0.9)
    
    m_prob = mhp.HealpixMap(data=cumprob, uniq=skymap['UNIQ'], density=True)
    img = m_prob.get_wcs_img(ax)#, coord = coord, rasterize = rasterize)
    im = ax.contour(img, levels=[0.5, 0.9], colors=['black', 'black'], linestyles=['solid', 'dashed']) #nested=True,

    area_50 = sorted_metadata['PIXELAREA'][:i_50].sum().to(u.deg**2)
    area_90 = sorted_metadata['PIXELAREA'][:i_90].sum().to(u.deg**2)

    return area_50, area_90

def deres(debug_regions):
    """
    Compresses a list of HEALPix regions into the coarsest possible resolution 
    by recursively merging groups of 4 sibling pixels into their parent.
    
    Parameters:
    -----------
    debug_regions : list of dicts
        Format: [{'nside': 1024, 'pixels': [...]}, {'nside': 512, 'pixels': [...]}]
        
    Returns:
    --------
    list of dicts
        The minimized and coarsened pixel regions.
    """
    # 1. Group all input pixels into sets mapped by their hierarchical order (order = log2(nside))
    pixel_sets = {}
    for region in debug_regions:
        nside = region['nside']
        pixels = region['pixels']
        order = int(np.round(np.log2(nside)))
        if order not in pixel_sets:
        # Using a set allows fast membership checks and removal during merging
            pixel_sets[order] = set()
        pixel_sets[order].update(pixels)
        
    # 2. Iteratively merge from fine (high order) to coarse (low order) 
    # until no more groups of 4 siblings can be combined.
    changed = True
    while changed:
        changed = False
        orders = sorted(pixel_sets.keys(), reverse=True)
        
        for order in orders:
            if order <= 0:
                continue
            current_pixels = pixel_sets.get(order, set())
            if not current_pixels:
                continue
                
            # Map parent pixel indices to their set of present children (0, 1, 2, or 3)
            parent_dict = {}
            for pix in current_pixels:
                parent = pix // 4
                remainder = pix % 4
                if parent not in parent_dict:
                    parent_dict[parent] = set()
                parent_dict[parent].add(remainder)
                
            # Find parents that have all 4 children present ({0, 1, 2, 3})
            parents_to_promote = [p for p, children in parent_dict.items() if children == {0, 1, 2, 3}]
            
            if parents_to_promote:
                # Remove the 4 children from the current finer order
                for parent in parents_to_promote:
                    for i in range(4):
                        current_pixels.remove(4 * parent + i)
                        
                # Add the parent to the next coarser order (order - 1)
                lower_order = order - 1
                if lower_order not in pixel_sets:
                    pixel_sets[lower_order] = set()
                pixel_sets[lower_order].update(parents_to_promote)
                
                # Flag that a merge occurred so the loop runs again 
                # (allowing newly formed parents to cascade upward if possible)
                changed = True
                
    # 3. Format the result back into your original list-of-dicts structure
    compressed_regions = []
    for order in sorted(pixel_sets.keys()):
        if pixel_sets[order]:
            compressed_regions.append({
                'nside': int(2**order),
                'pixels': sorted(list(pixel_sets[order]))
            })
            
    return compressed_regions

def is_source_in_contour_area(skymap, ra, dec, p=0.9, return_bool=False):
    '''
    Checks if a source's position is inside the top p contoour area. 

    m: mhealpy HealpixMap object 
    ra: float or 1darray
    dec: float or 1darray
    p: contour area to check. p=0.9 corresponds to the 90% percent contour area

    Returns:
    accepted_ra, accepted_dec: np.array. Returns the ra & dec posiitons of the objects that 
        where within the p contour region. Returns empty array if no sources accepted the criteria
    '''

    if isinstance(ra, float) or isinstance(ra, int):
        ra = np.array(ra)
    if isinstance(dec, float) or isinstance(dec, int):
        dec = np.array(dec)

    # skymap = QTable({
    #     'PROBDENSITY': m.data,
    #     'UNIQ': m.uniq
    # })

    #get uniq at ra, dec value
    metadata = get_moc_map_metadata(skymap)

    if 'ABSPROB' in skymap.keys():
        m = mhp.HealpixMap(data=skymap['ABSPROB'], nside=metadata['NSIDE'])
    else:
        
        abs_prob = skymap['PROBDENSITY'] * metadata['PIXELAREA']
        m = mhp.HealpixMap(data=abs_prob, nside=metadata['NSIDE'])

    pix_at_radec = m.ang2pix(ra, dec, lonlat=True)
    uniq_at_radec = m.pix2uniq(pix_at_radec)


    #get area contour
    
    skymap.sort("PROBDENSITY", reverse=True)
    cumprob = np.cumsum(skymap["PROBDENSITY"] * metadata['PIXELAREA'])
    # searchsorted gives the FIRST index where cumprob >= target.
    i_p = cumprob.searchsorted(p)

    accepted_uniq = skymap['UNIQ'][:i_p]

    
    bool_array = np.isin(uniq_at_radec, accepted_uniq)
    if return_bool:
        return bool_array

    accepted_ra, accepted_dec = ra[bool_array], dec[bool_array]
    return accepted_ra, accepted_dec

    
    

    #evaluate if uniq is in area

def get_map_area(map_data, p=0.9):
    '''
    Computes the containment area up to a value of p. p=0.9 corresponds to the 90% probability area
    '''

    skymap = map_data['skymap']
    skymap.sort("PROBDENSITY", reverse=True)
    sorted_metadata = get_moc_map_metadata(skymap)
    prob = sorted_metadata['PIXELAREA'] * skymap["PROBDENSITY"]
    cumprob = np.cumsum(prob)
    # searchsorted gives the FIRST index where cumprob >= target.
    p_i = cumprob.searchsorted(p)

    p_area = sorted_metadata['PIXELAREA'][:p_i].sum().to(u.deg**2)

    return p_area

def get_neutrino_properties(map_data, alert):
    '''
    Returns a dict with the relevant neutrino properties for the Rubicube framework from the json alert'''

    neutrino_properties = dict()

    neutrino_properties['E'] = alert['nu_energy']
    neutrino_properties['ra'] = alert['RA']
    neutrino_properties['dec'] = alert['DEC']
    neutrino_properties['t_nu'] = alert['DATE-OBS']
    neutrino_properties['A_90'] = get_map_area(map_data, p=0.9)

    return neutrino_properties

def get_max_ra_dec_from_map(m):
    ''''
    Computes the ra and dec coordinate for the maximum value in the healpy map. 
    
    
    m: HealpixMap object from mhealpy. 
    '''
    arg_max = np.argmax(m.data)
    ra, dec = hp.pix2ang(m.nside, arg_max, lonlat=True, nest=m.is_nested)
    return ra, dec

def evaluate_map(skymap, ra, dec):
    '''
    Reads the absolute spatial probability from the map at a sky location. 
    
    Params:
    -------
    skymap: QTable object, original healpix map where skymap[PROBDENSITY] is in 1/sr
    ra: float, right assention in degrees. It could also have astropy.unit of degree
    dec: float, declination in degrees. It could also have astropy.unit of degree

    Returns:
    -------
    spatial_prob: float, spatial probaility at ra, dec. 
    '''
    if not isinstance(ra, u.quantity.Quantity):
        ra = ra * u.deg
    if not isinstance(dec, u.quantity.Quantity):
        dec = dec * u.deg


    ra = ra.to(u.deg).value
    dec = dec.to(u.deg).value 
    
    metadata = get_moc_map_metadata(skymap)
    abs_prob = metadata['PROBDENSITY'] * metadata['PIXELAREA']

    mhpx_map = mhp.HealpixMap(data=abs_prob, uniq=metadata['UNIQ'], density=False)

    spatial_prob = mhpx_map.get_interp_val(ra, dec, lonlat=True)

    
    return spatial_prob

def select_pixels(map_url, p_region=0.9, map_dir=p['map_dir'], nest=True):
    #first we download the map
    file_name = map_url.split('/')[-1]
    title = file_name.split('_')[0]
    
    response = requests.get(map_url)
    map_path = os.path.join(map_dir, file_name)
    with open(map_path, 'wb') as file: #also deletes previously existing file and redownloads alert. 
        file.write(response.content)
    hpx, header = read_sky_map(map_path, nest=nest)

    skymap = QTable.read(map_path)
    level, ipix = ah.uniq_to_level_ipix(skymap['UNIQ'])
    nside = ah.level_to_nside(level)
    prob_density = skymap['PROBDENSITY'].to_value(u.deg**-2)

    skymap.sort("PROBDENSITY", reverse=True)
    level, ipix = ah.uniq_to_level_ipix(skymap["UNIQ"])
    nside = ah.level_to_nside(level)
    pixel_area = ah.nside_to_pixel_area(nside)
    prob = pixel_area * skymap["PROBDENSITY"]
    cumprob = np.cumsum(prob)

    #get required level
    region_index = cumprob.searchsorted(p_region)

    level, ipix = ah.uniq_to_level_ipix(skymap["UNIQ"])
    nside = ah.level_to_nside(level)
    masked_list = []
    nsides = []
    for nside_i in nside[:region_index]:
        if int(nside_i) not in nsides:
            nsides.append(int(nside_i))
    for nside_i in nsides:
        masked_list.append({'nside': nside_i,
                        'pixels': []})
    for nside_i, ipix_i in zip(nside[:region_index], ipix[:region_index]):
        for element in masked_list:
            if element['nside'] == int(nside_i):
                element['pixels'].append(int(ipix_i))
    
    #masked_list [{nside_1: [pixel_1, ...]}, {nside_2: [pixel_1, ....]}, ...]
    sel_pixels_lat = []
    sel_pixels_lon = []
    pixel_selection = {}
    for dict_nside in masked_list:
        nside = dict_nside['nside']
        pixels = dict_nside['pixels']
        lon, lat = ah.healpix_to_lonlat(pixels, nside=nside, order='nested')
        sel_pixels_lat.extend(lat.to('deg') / u.deg)
        sel_pixels_lon.extend(lon.to('deg') / u.deg)
        pixel_selection[float(nside)] = [list(lat.to('deg') / u.deg),
                                         list(lon.to('deg') / u.deg)]

    return pixel_selection

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
    elif isinstance(dec, u.quantity.Quantity):
        dec = np.array([dec.to(u.deg).value])

    if isinstance(E_nu, float) or isinstance(E_nu, int):
            E_nu = np.array([E_nu])
    elif isinstance(E_nu, u.quantity.Quantity):
        E_nu = np.array([E_nu.value])
    
    
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


# def check_source_in_map(ra, dec, map_url, p_region=0.9, map_dir=p['map_dir'], nest=True, source_name='Source', plot=True, **kwargs):
#     '''
#     Params
#     ra: (lon), float in degrees
#     dec: (lat), float in degrees
#     '''

#     source_in_pixelrange = False
    
#     #first we get pixel selection
    
#     pixel_selection = select_pixels(map_url, p_region=p_region, map_dir=map_dir, nest=nest)
#     for nside in pixel_selection.keys():
#         lat, lon = pixel_selection[nside]
#         area = hp.nside2pixarea(int(nside), degrees=True)
#         #we assume its a square
#         dx = np.sqrt(area) / 2
#         for lon_i, lat_i in zip(lon, lat):
#             lon_range =[float(lon_i - dx), float(lon_i + dx)]
#             lat_range = [float(lat_i - dx), float(lat_i + dx)]
#             if lon_range[0] <= ra <= lon_range[1] and lat_range[0] <= dec <= lat_range[1]:
#                 source_in_pixelrange=True
#                 print('Source is inside specified region')
#                 break
#         if not source_in_pixelrange:
#             print('Source is not inside specified region')

#     if plot:
#         plot_IC_alert(map_url, 
#                         map_dir=map_dir, 
#                         plot_source={'ra': ra,
#                                     'dec': dec,
#                                     'name': source_name},
#                         **kwargs)
#     return source_in_pixelrange