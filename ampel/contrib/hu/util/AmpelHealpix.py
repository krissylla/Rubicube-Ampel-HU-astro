#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File:                Ampel-HU-astro/ampel/contrib/hu/util/AmpelHealpix.py
# License:             BSD-3-Clause
# Author:              jakob nordin
# Date:                27.03.2023
# Last Modified Date:  07.02.2024
# Last Modified By:    ernstand@physik.hu-berlin.de


import math
import os
from base64 import b64encode
from collections import defaultdict
from datetime import datetime
from hashlib import blake2b

import healpy as hp
import numpy as np
import requests
from astropy import units as u
from astropy.time import Time

#From cris
from astropy.table import QTable
from mhealpy import HealpixMap
import astropy_healpix as ah


# ruff: noqa: T201


class AmpelHealpix:
    """
    - Obtain and load (GW) Healpix map.
    - Get pixels above some probability threshold.
    - Get probability for some (ra,dec).
    - Get mean distance & distance uncertainty
    """

    # Disk storage
    save_dir: str = "."

    def __init__(
        self,
        map_name: str,
        map_url: None | str = None,
        save_dir: None | str = None,
        nside: None | int = None,
    ):
        self.map_name = map_name
        self.map_url = map_url
        # print(self.map_url)
        if save_dir:
            self.save_dir = save_dir
        self.nside = nside

        self._get_map()
        # Attribues
        self.credible_levels = np.empty(0)
        self.trigger_time: None | float = None

        #cris
        self.sorted_credible_levels = None

    def _get_map(self, clobber=False) -> int:
        path = os.path.join(self.save_dir, self.map_name)
        # print(path)
        if os.path.exists(path) and not clobber:
            # print("Map exists and found: ", path)
            return 1

        # Retrieve mapfile.
        assert self.map_url
        map_data = requests.get(self.map_url)
        with open(path, "wb") as fh:
            fh.write(map_data.content)

        return 0
    


    def process_map(self) -> str:
        """
        Load map and determine prob values.
        """

        skymap = QTable.read( os.path.join(self.save_dir, self.map_name) )
        headers = skymap.meta
        prob_density = skymap['PROBDENSITY'].to_value(u.deg**-2)
        m = HealpixMap(data=prob_density, uniq=skymap['UNIQ'], density=True)
        pixel_areas = m.pixarea().to_value(u.deg**2) #* (180/np.pi)**2  # Convert steradians to deg^2


        # Determine trigger time
        # New icecube / mhealpy format
        if isinstance(headers, dict) and "DATE-OBS" in headers:
            trigger_time = headers["DATE-OBS"]
            trigger_time = datetime.fromisoformat(trigger_time)
        else:
            # Old ligo format
            trigger_time = next(
                datetime.fromisoformat(header[1])
                for header in headers
                if header[0] == "DATE-OBS"
            )

        # Check resolution and possible change 
        if self.nside and self.nside < m.nside:
            print("Downgrading map from nside ", m.nside, " to ", self.nside) #TODO: Do we need this?
            m = m.ud_grade(
                nside_out=self.nside,
                power=-1,   # weigted average, conserves total probability
            )
        else:
            self.nside = m.nside


        #Info only for GW detections. If IC map, we don't need this:
        try:
            # Will likely fail for new mhealpy format
            self.dist = next(header[1] for header in headers if header[0] == "DISTMEAN")
        except StopIteration:
            self.dist = None

        try:
            # Only for GW, not mo icecube alerts
            self.dist_unc = next(header[1] for header in headers if header[0] == "DISTSTD")
        except StopIteration:
            self.dist_unc = None

        seed_list = [header[1] for header in headers if header[0] == "SEED"]
        if len(seed_list) > 0:
            self.seed = seed_list[0]
        else:
            self.seed = self.map_name.replace(".fits.gz", "").replace(",", "_")

        # Find credible levels

        # mhealpy, trying to take pixel area into account
        idx = np.flipud( np.argsort(m.data) ) #TODO: from single-order map, not sure if we need this

        skymap.sort("PROBDENSITY", reverse=True)

        #CRIS: Check if array has masked values. We fill them with 0.0
        try:
            if skymap['PROBDENSITY'].mask.any():
                print("Map has masked values. reeplacing them with 0")
                skymap['PROBDENSITY'] = skymap['PROBDENSITY'].filled(0)
        except AttributeError:
            pass #array has no masked values
        level, ipix = ah.uniq_to_level_ipix(skymap["UNIQ"])
        nside = ah.level_to_nside(level)
        pixel_area = ah.nside_to_pixel_area(nside)
        prob = pixel_area * skymap["PROBDENSITY"]
        sorted_credible_levels = np.cumsum(prob)

        credible_levels = np.empty_like(sorted_credible_levels)
        credible_levels[idx] = sorted_credible_levels


        self.credible_levels = credible_levels
        self.sorted_credible_levels = sorted_credible_levels
        self.trigger_time = Time(trigger_time).jd

        return b64encode(
            blake2b(sorted_credible_levels, digest_size=7).digest()
        ).decode("utf-8")

    def get_triggertime(self):
        if self.trigger_time is None:
            self.process_map()
        return self.trigger_time

    def get_pixelmask(self, pvalue_limit):
        """
        Return pixels with total probability up to some limit.
        """

        if not self.nside:
            raise ValueError("First get and process map before using.")

        # Create mask for pixel selection
        assert len(self.credible_levels)
        mask = np.zeros(len(self.credible_levels), dtype=int)
        mask[self.credible_levels <= pvalue_limit] = 1

        return mask.nonzero()[0].tolist()

    def get_maparea(self, pvalue_limit: float, healpix_regions=None) -> float:
        """
        TODO: this only works for single order maps, can we do this with self.sorted_credible_levels?
        Return sky area for probability threshold in square degrees.
        """

        if not healpix_regions:
            pixels = self.get_pixelmask(pvalue_limit)
            # Combine pixels when possible
            deresdict = deres(self.nside, pixels)
            healpix_regions = [
                {"nside": nside, "pixels": members} for nside, members in deresdict.items()
            ]
        # calculate relevant map area
        hp_area = 0.0
        for region in healpix_regions:
            npix_from_nside = 12 * region["nside"] ** 2
            hp_area += len(region["pixels"]) / npix_from_nside
        hp_area *= 360**2 / np.pi
        # print("USED MAP AREA: ", hp_area)

        return hp_area

    def get_cumprob(self, ra: float, dec: float) -> float:
        """
        Obtain probability for a specific coordinate based on loaded map.
        ra, dec in degrees.
        """

        if not self.nside:
            raise ValueError("First get and process map before using.")
        assert len(self.credible_levels)

        theta = 0.5 * np.pi - np.deg2rad(dec)
        phi = np.deg2rad(ra)
        alertpix = hp.pixelfunc.ang2pix(
            hp.npix2nside(len(self.credible_levels)), theta, phi, nest=True
        )
        return self.credible_levels[alertpix]

    def get_mapdist(self) -> tuple[float, float]:
        """
        Obtain mean distance + std from healpix map in Mpc.
        """
        if not self.nside:
            raise ValueError("First get and process map before using.")

        return self.dist, self.dist_unc

    #cris attempt
    # def get_hpx_regions(self, p_value):
    #     '''
    #     This function should combine get_pixel_mask and deres
    #     We will use the slef.sorted_credible_levels to get the limit set by the p_value
    #     p_value = 0.9 -> 90% confidence region in the IC map
    #     p_value = 0.5 -> 50% confidence region in the IC map
    #     '''
    #     i_region = self.sorted_credible_levels.searchsorted(p_value)

    #     #TODO: we calculate this bvadly again, we should keep this info from process_map 

    #     skymap = QTable.read(os.path.join(self.save_dir, self.map_name) )
    #     prob_density = skymap['PROBDENSITY'].to_value(u.deg**-2)
    #     skymap.sort("PROBDENSITY", reverse=True)
    #     level, ipix = ah.uniq_to_level_ipix(skymap["UNIQ"])
    #     nside = ah.level_to_nside(level)
    #     masked_list = []
    #     nsides = []
    #     for nside_i in nside[:i_region]:
    #         if int(nside_i) not in nsides:
    #             nsides.append(int(nside_i))
    #     for nside_i in nsides:
    #         masked_list.append({'nside': nside_i,
    #                         'pixels': []})
    #     for nside_i, ipix_i in zip(nside[:i_region], ipix[:i_region]):
    #         for element in masked_list:
    #             if element['nside'] == int(nside_i):
    #                 element['pixels'].append(int(ipix_i))
        
    #     return masked_list

    def get_hpx_regions(self, p_value):

        ##DEBUG
        print('reading map from: ', os.path.join(self.save_dir, self.map_name))

        skymap = QTable.read(os.path.join(self.save_dir, self.map_name) )
            
        skymap.sort("PROBDENSITY", reverse=True)
        
        level, ipix = ah.uniq_to_level_ipix(skymap["UNIQ"])
        nside = ah.level_to_nside(level)
    
    
        prob_density = skymap['PROBDENSITY']
        pixel_area = ah.nside_to_pixel_area(nside)
    
        prob = pixel_area * prob_density
        
        cumprob = np.cumsum(prob)
        i_region = cumprob.searchsorted(p_value)
        
        selected_nside = nside[:i_region]
        selected_ipix = ipix[:i_region]
        
        nside_selection = list(set(selected_nside)) 
        masked_dict = {nside_i: [] for nside_i in nside_selection}
        
        for i in range(len(selected_nside)):
            masked_dict[selected_nside[i]].append(int(selected_ipix[i]))
    
        masked_list = []
        for nside_i in masked_dict.keys():
            masked_list.append({
                'nside': int(nside_i),
                'pixels': masked_dict[nside_i]
            })
            
        return masked_list


# def deres(nside, ipix, min_nside=1):
#     """
#     Decompose a set of (nested) HEALpix indices into sets of complete superpixels at lower resolutions.
#     :param nside: nside of given indices
#     :param ipix: pixel indices
#     :min_nside: minimum nside of complete pixels

#     Copied from Ampel-ZTF-archive.
#     """
#     remaining_pixels = set(ipix)
#     decomposed = defaultdict(list)
#     for log2_nside in range(int(math.log2(min_nside)), int(math.log2(nside)) + 1):
#         super_nside = 2**log2_nside
#         # number of base_nside pixels per nside superpixel
#         scale = (nside // super_nside) ** 2
#         # sort remaining base_nside pixels by superpixel
#         by_superpixel = defaultdict(list)
#         for pix in remaining_pixels:
#             by_superpixel[pix // scale].append(pix)
#         # represent sets of pixels that fill a superpixel
#         # as a single superpixel, and remove from the working set
#         for superpix, members in by_superpixel.items():
#             if len(members) == scale:
#                 decomposed[super_nside].append(superpix)
#                 remaining_pixels.difference_update(members)

#     return dict(decomposed)

def deres(healpix_regions):
    """
    Addition by Cris Z:
    An adaptation of the deres function from AmpelHealpix.py that now works for multi-ordered maps.
    
    Compresses a list of HEALPix regions into the coarsest possible resolution 
    by recursively merging groups of 4 sibling pixels into their parent.
    
    Parameters:
    -----------
    healpix_regions : list of dicts
        Format: [{'nside': 1024, 'pixels': [...]}, {'nside': 512, 'pixels': [...]}]
        
    Returns:
    --------
    list of dicts
        The minimized and coarsened pixel regions.
    """
    # 1. Group all input pixels into sets mapped by their hierarchical order (order = log2(nside))
    pixel_sets = {}
    for region in healpix_regions:
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

def main():
    # map_name = "/mnt/c/Users/Public/Documents/Uni/master/masterarbeit/ampel/new_BBH_mergers_O3a_IAS_pipeline/skymapGW190704_104834.fits.gz"
    # map_name = "/mnt/c/Users/Public/Documents/Uni/master/masterarbeit/ampel/ampel-results/ligo-kilonova/tmp/S190408an.fits.gz,0"
    map_name = "/mnt/c/Users/Public/Documents/Uni/master/masterarbeit/ampel/Ampel-HU-astro/simulated_maps/tmp_gaussmap.fits.gz"
    map_name = "/mnt/c/Users/Public/Documents/Uni/master/masterarbeit/ampel/ampel-results/ligo-kilonova/tmp/S190426c.fits.gz,1"
    ah = AmpelHealpix(
        # map_name="S191222n.fits.gz",
        map_name=map_name,
        # map_url="https://gracedb.ligo.org/api/superevents/S191222n/files/LALInference.fits.gz",
    )
    hashit = ah.process_map()
    print(hashit)
    pixels = ah.get_pixelmask(0.9)  # noqa: F841
    print(ah.get_maparea(0.9))
    print(ah.get_mapdist())

    print(ah.trigger_time)

    print(ah.get_cumprob(13, 48))
    print(ah.get_cumprob(200, -14.88))
    print(ah.get_cumprob(68.58494, 33.89))


if __name__ == "__main__":
    main()
