import os
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

from params import p
from ic_processing import read_map, get_moc_map_metadata, get_p_value_contour, \
                        get_50_90_contour

import warnings #for import lal inside ligo.skymap
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")



def plot_map_on_axis(healpix_map, ax, plot_ztf_lsst=True, clabel=''):
    '''
    Plots a map onto an axis 'ax' which already has mollwedie projection
    '''
    
    if plot_ztf_lsst:
        lsst_limit =  [np.linspace(0,360, 50), np.ones(50) * 30]
        ztf_limit = [np.linspace(0,360, 50), np.ones(50) * -30]
        ax.plot(lsst_limit[0], lsst_limit[1], c='purple', linestyle='dotted', transform=ax.get_transform('world'))
        ax.plot(ztf_limit[0], ztf_limit[1], c='green', linestyle='dotted', transform=ax.get_transform('world'))

    if isinstance(ax, ligo.skymap.plot.allsky.AstroDegreesMollweideAxes):
        ax.text(175, 20, r'$\downarrow$LSST', c='purple', transform=ax.get_transform('world'))
        ax.text(175, -28, r'$\uparrow$ZTF', c='green', transform=ax.get_transform('world'))
    img = healpix_map.get_wcs_img(ax)#, coord = coord, rasterize = rasterize)
    im = ax.imshow(img, cmap='OrRd')
    if isinstance(ax, ligo.skymap.plot.allsky.AstroDegreesZoomAxes):
        c = plt.colorbar(im, label=clabel)




def plot_sources_on_ax(ax, plot_source):
        if isinstance(plot_source, dict):
            ax.plot(plot_source['ra'],
                        plot_source['dec'], 'o',
                        label=plot_source['name'],
                        c='cyan',  
                        transform=ax.get_transform('world'))    
            ax.legend(framealpha=1)
        
        elif isinstance(plot_source, list):
            for source in plot_source:
                ax.plot(source['ra'],
                            source['dec'], 'o',
                            label=source['name'],
                            transform=ax.get_transform('world'))    
                ax.legend(framealpha=1)
        return None

def plot_full_map(map_data, r=3, plot_prob_density=False, save_dir=p['save_dir'],
             coord='icrs', nest=True, show_fig=True, plot_ztf_lsst=True, plot_source=False,
             plot_only_zoomed_frame=False, p_value=None):
    '''
     
    '''
    
    #plotting info for ligo.skymap.plot
    ligo_coord = {
        'E': 'geo',
        'icrs': 'astro',
        'G': 'galactic'
    }
    astropy_frame = {
        'icrs': 'icrs',
        'G': 'galactic'
    }

    #get data and metadata
    skymap = map_data['skymap']
    header = map_data['header']
    alert_name = map_data['alert_name']
    metadata = get_moc_map_metadata(skymap)
    ra, dec = metadata['RA'], metadata['DEC']


    #zoom box center
    center = SkyCoord(ra.to('rad'), dec.to('rad'), frame=astropy_frame[coord])
    
    fig = plt.figure(figsize=(6, 6), dpi=100)

    ax = plt.axes(
        [0.05, 0.05, 0.8, 0.9],
        projection=f'{ligo_coord[coord]} degrees mollweide',
        center=SkyCoord(0 * u.rad, 0 * u.rad, frame=astropy_frame[coord]))
    ax_inset = plt.axes(
        [1.03, 0.3, 0.42, 0.42],
        projection=f'{ligo_coord[coord]} degrees zoom',
        center=center,
        radius=r * u.deg)
    
    if coord == 'icrs':
        for key in ['ra', 'dec']:
            ax_inset.coords[key].set_ticks_visible(False)
    
    ax_inset.set_xlabel("")
    ax_inset.set_ylabel("")
    ax.grid()
    ax.locator_params(nbins=10) #test!
    ax.mark_inset_axes(ax_inset)
    ax.connect_inset_axes(ax_inset, 'upper left')
    ax.connect_inset_axes(ax_inset, 'lower left')
    ax_inset.scalebar((0.1, 0.1), 1 * u.deg).label()
    ax_inset.compass(0.9, 0.1, 0.2)
    ax_inset.grid()
    ax.set_title(alert_name)
    ax_inset.set_title(fr'RA: {ra.value:.2f}$\degree$  DEC: {dec.value:.2f}$\degree$')


    #map processing
    if plot_prob_density:
        prob_density = skymap['PROBDENSITY'].to_value(u.deg**-2)
        m = mhp.HealpixMap(data=prob_density, uniq=skymap['UNIQ'], density=True)
        clabel = r'Prob density [$deg^{-2}$]'
    else:
        #we show absolute probability = prob_density * pixel_area
        prob = metadata['PIXELAREA'] * skymap['PROBDENSITY']
        m = mhp.HealpixMap(data=prob, uniq=skymap['UNIQ'], density=False)
        clabel = 'Absolute Probability'

    plot_map_on_axis(m, ax, plot_ztf_lsst=plot_ztf_lsst)
    plot_map_on_axis(m, ax_inset, plot_ztf_lsst=plot_ztf_lsst, clabel=clabel)

    _ = get_50_90_contour(skymap, ax)
    area_50, area_90 = get_50_90_contour(skymap, ax_inset)

    #get our own p_value search
    if p_value is not None:
        area_p = get_p_value_contour(skymap, ax_inset, p_value)
        #we add it to the text box
        text_str = f'Area 50%:\n {area_50:.3f} \n\nArea 90%:\n {area_90:.3f} \n\n Queried Area {p_value * 100:.0f}%:\n {area_p:.3f}'
        text_box_position = [1.9, 0.15]
    else:
        #text box just includes 90% and 50% regions
        text_str = f'Area 50%:\n {area_50:.3f} \n\nArea 90%:\n {area_90:.3f}'
        text_box_position = [1.9, 0.35]

    

    #area text
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(text_box_position[0], text_box_position[1], text_str, fontsize=14, bbox=props, transform=ax.transAxes)#, horizontalalignment='right', va='top')

    plot_sources_on_ax(ax_inset, plot_source)

    # if plot_source is not None or p_value is not None:
    #     ax_inset.legend(framealpha=1)

    handles, labels = ax.get_legend_handles_labels()
    if handles:  # Only show legend if there are labeled artists
            ax.legend(handles, labels, framealpha=1)

    if isinstance(save_dir, str):
        fig_path = os.path.join(save_dir, f'{alert_name}.png')
        plt.savefig(fig_path, bbox_inches='tight')
    if show_fig:
        plt.show()
        
    return fig_path, area_90, m

def plot_selected_regions(ax, hpx_regions):
    '''
    Used for debug, shows the selected pixels in the query
    '''
    for level in hpx_regions:
        nside = level['nside']
        pixels = level['pixels']

        for pixel in pixels:
            data = np.zeros(hp.nside2npix(nside))
            data[pixel] += 1

            healpix_map = mhp.HealpixMap(data=data, nside=nside, scheme='NESTED')

            img = healpix_map.get_wcs_img(ax)#, coord = coord, rasterize = rasterize)

            #in a boolean map of values 0 or 1, level=0.5 is identified when the map sharply passes from 0 to 1, which works for outlining the pixels
            _ = ax.contour(img, levels=[0.5], colors=['green'], linestyles=['solid']) #nested=True,
        
    return None

def plot_zoomed_IC_alert(map_data, r=3, plot_prob_density=False, save_dir=p['save_dir'],
             coord='icrs', show_fig=True, plot_ztf_lsst=True, plot_source=False,
             plot_only_zoomed_frame=False, p_value=None, debug=None):
    '''
    '''
    #plotting info
    ligo_coord = {
        'E': 'geo',
        'icrs': 'astro',
        'G': 'galactic'
    }
    astropy_frame = {
        'icrs': 'icrs',
        'G': 'galactic'
    }
    
    #get data and metadata
    skymap = map_data['skymap']
    header = map_data['header']
    alert_name = map_data['alert_name']
    metadata = get_moc_map_metadata(skymap)
    ra, dec = metadata['RA'], metadata['DEC']


    #zoom box center
    center = SkyCoord(ra.to('rad'), dec.to('rad'), frame=astropy_frame[coord])
    
    fig = plt.figure(figsize=(6, 6), dpi=100)


    ax = plt.axes(
        [0.05, 0.05, 0.8, 0.9],
        projection=f'{ligo_coord[coord]} degrees zoom',
        center=center,
        radius=r * u.deg)
    
    if coord == 'icrs':
        for key in ['ra', 'dec']:
            # ax_inset.coords[key].set_ticklabel_visible(False)
            ax.coords[key].set_ticks_visible(False)
    ax.set_xlabel("RA")
    ax.set_ylabel("DEC")
    ax.grid()    

    #map processing
    if plot_prob_density:
        prob_density = skymap['PROBDENSITY'].to_value(u.deg**-2)
        m = mhp.HealpixMap(data=prob_density, uniq=skymap['UNIQ'], density=True)
        clabel = r'Prob density [$deg^{-2}$]'
    else:
        #we show absolute probability = prob_density * pixel_area
        prob = metadata['PIXELAREA'] * skymap['PROBDENSITY']
        m = mhp.HealpixMap(data=prob, uniq=skymap['UNIQ'], density=False)
        clabel = 'Absolute Probability'

    plot_map_on_axis(m, ax, plot_ztf_lsst=plot_ztf_lsst, clabel=clabel)
    area_50, area_90 = get_50_90_contour(skymap, ax)

    

    plot_sources_on_ax(ax, plot_source)


    if type(debug) == list:
        plot_selected_regions(ax, debug)

    #get our own p_value search
    if p_value is not None:
        _ = get_p_value_contour(skymap, ax, p_value)
    
    ra_title = ra / u.deg
    dec_title = dec / u.deg
    ax.set_title(fr'RA: {ra_title:.2f}$\degree$  DEC: {dec_title:.2f}$\degree$')
    pos = ax.get_position()
    plt.suptitle(alert_name, x=pos.x0 + pos.width / 2.0, y=0.9)
    # plt.suptitle(alert_name)

    # if plot_source is not None or p_value is not None or debug is not None:
    #     ax.legend(framealpha=1)
    
    # handles, labels = ax.get_legend_handles_labels()
    # if handles:  # Only show legend if there are labeled artists
    #     ax.legend(handles, labels, framealpha=1)

    if save_dir:
        fig_path = os.path.join(save_dir, f'{alert_name}_mw_zoom.png')
        plt.savefig(fig_path, bbox_inches='tight')
    if show_fig:
        plt.show()
    
    if save_dir:
        return fig_path
    return None







    

   



if __name__ == '__main__':
    map_data =read_map('IceCube-260610A')
    plot_full_map(map_data, show_fig=False, plot_prob_density=True)
    # print(evaluate_map(map_data['skymap'], 309, 37))
