#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 2023
@author: Bridget Hass (bhass@battelleecology.org)

Functions to download AOP data either for an entire site (by_file_aop), or
for a portion of a site (by_tile_aop), specified by the UTM coordinates
(easting and northing) of the AOP data tiles that you want to download.

Adapted from R neonUtilities byFileAOP.R and byTileAOP.R, along with other helper functions.
https://github.com/NEONScience/NEON-utilities/blob/main/neonUtilities/R/byFileAOP.R
https://github.com/NEONScience/NEON-utilities/blob/main/neonUtilities/R/byTileAOP.R
written by:
@author: Claire Lunch (clunch@battelleecology.org)
@author: Christine Laney (claney@battelleecology.org)

"""

from time import sleep
import os
import re
import platform
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import requests
import importlib
import importlib_resources
from . import __resources__
from .helper_mods.api_helpers import get_api
from .helper_mods.api_helpers import download_file
from .helper_mods.metadata_helpers import convert_byte_size
from .get_issue_log import get_issue_log
from .citation import get_citation

# display the log info messages, only showing the message (otherwise it would print INFO:root:'message')
logging.basicConfig(level=logging.INFO, format='%(message)s')


# check that token was used
def check_token(response):
    """
    This function checks the API response headers for a rate limit. If the rate limit is found to be '200',
    it prints a warning message indicating that the API token was not recognized and the public rate limit was applied.

    Parameters
    --------
    response: API get response object containing status code and json data. Return of get_api function.

    Returns
    --------
    None

    Raises
    --------
    None

    """

    if 'x-ratelimit-limit' in response.headers and response.headers['x-ratelimit-limit'] == '200':
        logging.info(
            'API token was not recognized. Public rate limit applied.\n')

# %%


def get_file_urls(urls, token=None):
    """
    This function retrieves all the files from a list of NEON data URLs.

    Parameters
    --------
    urls: list of str
        A list of URLs from which the files will be retrieved.

    token: str, optional
        User-specific API token generated within neon.datascience user accounts. If provided, it will be used for authentication.

    Returns
    --------
    all_file_url_df: pandas.DataFrame
        A DataFrame containing information about all the files retrieved from the URLs.
        The DataFrame includes columns for 'name', 'size', 'crc32c', and 'url' and 'release' of the files.

    release: str
        The release information retrieved from the response JSON.

    Raises
    --------
    None

    Examples
    --------
    >>> urls = ['https://data.neonscience.org/api/v0/data/DP3.30015.001/SCBI/2022-05',
                'https://data.neonscience.org/api/v0/data/DP3.30015.001/SCBI/2023-06']
    >>> get_file_urls(urls, 'my-api-token')
    # This will return a DataFrame with all the data files contained in SCBI 2022-05 and 2023-06 and the release information.

    Notes
    --------
    The function makes API calls to each URL in the 'urls' list and retrieves the file information.
    It also retrieves the release information from the response JSON.
    If the API call fails, it prints a warning message and continues with the next URL.
    """

    all_file_url_df = pd.DataFrame()
    releases = []
    for url in urls:
        response = get_api(api_url=url, token=token)
        if response is None:
            logging.info(
                "Data file retrieval failed. Check NEON data portal for outage alerts.")

        # get release info
        release = response.json()['data']['release']
        releases.append(release)

        file_url_dict = response.json()['data']['files']
        file_url_df = pd.DataFrame(data=file_url_dict)
        file_url_df['release'] = release

        # drop md5 and crc32 columns, which are all NaNs
        file_url_df.drop(columns=['md5', 'crc32'], inplace=True)

        # append the new dataframe to the existing one
        all_file_url_df = pd.concat(
            [all_file_url_df, file_url_df], ignore_index=True)

    return all_file_url_df, list(set(releases))

# %%


def get_shared_flights(site):
    """
    This function retrieves shared flights information for a NEON site from
    the shared_flights.csv lookup file, which contains the mapping between
    collocated AOP sites (terrestrial:terrestrial and aquatic:terrestrial).
    If the site is found in the shared flights data, it prints a message and
    updates the site to the corresponding "flightSite".

    Parameters
    --------
    site: str
        The 4-letter NEON site code for which the collocated site information is to be retrieved.

    Returns
    --------
    site: str
        The collocated site that AOP data is published under. If the original
        site is found in the shared flights lookup, it is updated to the
        corresponding flight site. Otherwise, it remains the same.

    Raises
    --------
    None

    Examples
    --------
    >>> get_shared_flights('TREE')
    'TREE is part of the flight box for STEI. Downloading data from STEI.'

    Notes
    --------
    The function reads the shared flights data from a CSV file named
    'shared_flights.csv' located in the '__resources__' directory.

    """
    shared_flights_file = (importlib_resources.files(
        __resources__) / 'shared_flights.csv')

    shared_flights_df = pd.read_csv(shared_flights_file)

    shared_flights_dict = shared_flights_df.set_index(
        ['site'])['flightSite'].to_dict()
    if site in shared_flights_dict:
        flightSite = shared_flights_dict[site]
        if site in ['TREE', 'CHEQ', 'KONA', 'DCFS']:
            logging.info(
                f'{site} is part of the flight box for {flightSite}. Downloading data from {flightSite}.')
        else:
            logging.info(
                f'{site} is an aquatic site and is sometimes included in the flight box for {flightSite}. Aquatic sites are not always included in the flight coverage every year.\nDownloading data from {flightSite}. Check data to confirm coverage of {site}.')
        site = flightSite
    return site


def get_neon_sites():
    """This function gets a list of the valid NEON sites from the 
    neon_field_site_metadata.csv file for validation, and adds the AOP CHEQ 
    site, which is an AOP site name and is associated with STEI & TREE."
    """
    neon_sites_file = (importlib_resources.files(
        __resources__) / "neon_field_site_metadata_20250214.csv")
    neon_sites_df = pd.read_csv(neon_sites_file)

    neon_sites_list = list(neon_sites_df['field_site_id'])
    neon_sites_list.append('CHEQ')

    return neon_sites_list


def get_data_product_name(dpid):
    dpid_api_response = get_api(
        f'https://data.neonscience.org/api/v0/products/{dpid}')
    product_name = dpid_api_response.json()['data']['productName']
    return product_name

# %% functions to validate inputs for by_file_aop and by_tile_aop


def validate_dpid(dpid):
    dpid_pattern = "DP[1-4]{1}.[0-9]{5}.00[1-2]{1}"
    if not re.fullmatch(dpid_pattern, dpid):
        raise ValueError(
            f'{dpid} is not a properly formatted data product ID. The correct format is DP#.#####.00#')

# TODO: change to read a list of valid AOP DPIDs
# def validate_aop_dpid(dpid):
#     aop_dpid_pattern = "DP[1-3]{1}.300[1-9]{1}.00[1-2]{1}"
#     if not re.fullmatch(aop_dpid_pattern, dpid):
#         raise ValueError(
#             f'{dpid} is not a valid AOP data product ID. AOP data products follow the format DP#.300##.00#.')


# List of valid AOP data product IDs (2024 +, includes the .002 spectrometer revisions)
valid_aop_dpids = ['DP1.30001.001',  # L1 waveform lidar
                   # L1 & L3 discrete lidar
                   'DP1.30003.001', 'DP3.30015.001', 'DP3.30024.001', 'DP3.30025.001',
                   # L1 & L3 camera
                   'DP1.30010.001', 'DP3.30010.001',
                   # L1 spectrometer (.001 & .002)
                   'DP1.30006.001', 'DP1.30006.002', 'DP1.30008.001',
                   # L2 spectrometer, .001
                   'DP2.30011.001', 'DP2.30012.001', 'DP2.30014.001', 'DP2.30019.001', 'DP2.30026.001',
                   # L2 spectrometer, .002
                   'DP2.30011.002', 'DP2.30012.002', 'DP2.30014.002', 'DP2.30019.002', 'DP2.30026.002',
                   # L3 spectrometer, .001
                   'DP3.30006.001', 'DP3.30011.001', 'DP3.30012.001', 'DP3.30014.001', 'DP3.30019.001', 'DP3.30026.001',
                   # L3 spectrometer, .002
                   'DP3.30006.002', 'DP3.30011.002', 'DP3.30012.002', 'DP3.30014.002', 'DP3.30019.002', 'DP3.30026.002']

# List of valid Level 3 AOP data product IDs
valid_aop_l3_dpids = [
    dpid for dpid in valid_aop_dpids if dpid.startswith('DP3')]

# List of suspended AOP data product IDs (will need to change once these data products become active again)
suspended_aop_dpids = ['DP2.30018.001', 'DP3.30018.001',  # canopy nitrogen
                       'DP2.30020.001', 'DP3.30020.001',  # canopy xanthophyll cycle
                       'DP2.30022.001', 'DP3.30022.001',  # canopy lignin
                       'DP2.30016.001', 'DP3.30016.001']  # total biomass map

# request with suspended data (no data available)
# eg. https://data.neonscience.org/api/v0/products/DP3.30016.001
# suspended biomass data product
# productStatus: FUTURE
# releases: []
# siteCodes: NoneType object


def validate_aop_dpid(dpid):
    """
    Validates the given AOP data product ID against a pattern and a list of valid IDs.

    Parameters:
    - dpid (str): The data product ID to validate.

    Raises:
    - ValueError: If the dpid does not match the expected pattern or is not in the list of valid IDs.
    """
    # Regular expression pattern for AOP data product IDs
    aop_dpid_pattern = "DP[1-3]{1}.300[0-2]{1}[0-9]{1}.00[1-2]{1}"

    # Check if the dpid matches the pattern
    if not re.fullmatch(aop_dpid_pattern, dpid):
        raise ValueError(
            f'{dpid} is not a valid AOP data product ID. AOP data products follow the format DP#.300##.00#.')

    # Check if the dpid is in the list of suspended AOP dpids
    if dpid in suspended_aop_dpids:
        raise ValueError(
            f'{dpid} has been suspended and is not currently available, see https://data.neonscience.org/data-products/{dpid} for more details.')  # ' Valid AOP IDs are: {", ".join(valid_aop_dpids)}.')

    # Check if the dpid is in the list of valid AOP dpids
    if dpid not in valid_aop_dpids:
        valid_aop_dpids.sort()
        valid_aop_dpids_string = "\n".join(valid_aop_dpids)
        raise ValueError(
            f'{dpid} is not a valid AOP data product ID. Valid AOP IDs are listed below:\n{valid_aop_dpids_string}')


def validate_aop_l3_dpid(dpid):
    """
    Validates the given AOP data product ID against a pattern and a list of valid Level 3 AOP IDs.

    Parameters:
    - dpid (str): The data product ID to validate.

    Raises:
    - ValueError: If the dpid does not start with DP3 or is not in the list of valid Level 3 AOP data product IDs.
    """
    # Check if the dpid starts with DP3
    if not dpid.startswith('DP3'):
        raise ValueError(
            f'{dpid} is not a valid Level 3 (L3) AOP data product ID. Level 3 AOP products follow the format DP3.300##.00#')

    # Check if the dpid is in the list of valid AOP dpids
    if dpid not in valid_aop_l3_dpids:
        valid_aop_l3_dpids.sort()
        valid_aop_l3_dpids_string = "\n".join(valid_aop_l3_dpids)
        # dpid_dict = {valid_aop_l3_dpids[i]: get_data_product_name(
        #     valid_aop_l3_dpids[i]) for i in range(len(valid_aop_l3_dpids))}
        # formatted_dpid_dict = '\n'.join(
        #     f'{key}: {value}' for key, value in dpid_dict.items())

        raise ValueError(
            f'{dpid} is not a valid Level 3 (L3) AOP data product ID. Valid L3 AOP IDs are listed below:\n{valid_aop_l3_dpids_string}')
        # below prints out the corresponding data product names for each ID.
        # f'{dpid} is not a valid Level 3 (L3) AOP data product ID. Valid L3 AOP products are listed below.\n{formatted_dpid_dict}')


def check_field_spectra_dpid(dpid):
    if dpid == 'DP1.30012.001':
        raise ValueError(
            f'{dpid} is the Field spectral data product, which is published as tabular data. Use zipsByProduct() or loadByProduct() to download these data.')


def validate_site_format(site):
    site_pattern = "[A-Z]{4}"
    if not re.fullmatch(site_pattern, site):
        raise ValueError(
            f'{site} is an invalid site format. A four-letter NEON site code is required. NEON site codes can be found here: https://www.neonscience.org/field-sites/explore-field-sites')


def validate_neon_site(site):
    neon_sites = get_neon_sites()

    if site not in neon_sites:
        raise ValueError(
            f'{site} is not a valid NEON site code. A complete list of NEON site codes can be found here: https://www.neonscience.org/field-sites/explore-field-sites')


def validate_year(year):
    # year = str(year)
    year_pattern = "20[1-9][0-9]"
    if not re.fullmatch(year_pattern, year):
        raise ValueError(
            f'{year} is an invalid year. Year is required in the format "2017" or 2017, eg. AOP data are available from 2013 to present.')


def check_aop_dpid(response_dict, dpid):
    if response_dict['data']['productScienceTeamAbbr'] != 'AOP':
        logging.info(
            f'{dpid} is not a remote sensing product. Use zipsByProduct()')
        return


def get_site_year_urls(response_dict, site, year):
    site_info = next(
        item for item in response_dict['data']['siteCodes'] if item["siteCode"] == site)
    site_urls = site_info['availableDataUrls']
    site_year_urls = [url for url in site_urls if str(year) in url]
    return site_year_urls


# %% functions to display available dates and tile extents


def list_available_dates(dpid, site):
    """
    list_available_dates displays the available releases and dates for a given product and site
    --------
     Inputs:
         dpid: the data product code (eg. 'DP3.30015.001' - CHM)
         site: the 4-digit NEON site code (eg. 'JORN')
    --------
    Returns:
    prints the Release Tag (or PROVISIONAL) and the corresponding available dates (YYYY-MM) for each tag

    Examples:
    --------
    >>> list_available_dates('DP3.30015.001','JORN')
    RELEASE-2025 Available Dates: 2017-08, 2018-08, 2019-08, 2021-08, 2022-09

    >>> list_available_dates('DP3.30015.001','HOPB')
    PROVISIONAL Available Dates: 2024-09
    RELEASE-2025 Available Dates: 2016-08, 2017-08, 2019-08, 2022-08

    >>> list_available_dates('DP1.10098.001','HOPB')
    ValueError: There are no data available for the data product DP1.10098.001 at the site HOPB.

    Created on Feb 17 2025

    @author: Bridget Hass

    """
    product_url = "http://data.neonscience.org/api/v0/products/" + dpid
    response = get_api(api_url=product_url)  # add input for token?

    # raise value error and print message if dpid isn't formatted as expected
    validate_dpid(dpid)

    # raise value error and print message if site is not a 4-letter character
    site = site.upper()  # make site upper case (if it's not already)
    validate_site_format(site)

    # raise value error and print message if site is not a valid NEON site
    validate_neon_site(site)

# get available releases & months:
    for i in range(len(response.json()['data']['siteCodes'])):
        if site in response.json()['data']['siteCodes'][i]['siteCode']:
            available_releases = response.json(
            )['data']['siteCodes'][i]['availableReleases']

# display available release tags (including provisional) and dates for each tag
    try:
        for entry in available_releases:
            release = entry['release']
            available_months = ', '.join(entry['availableMonths'])
            logging.info(f"{release} Available Dates: {available_months}")
    except UnboundLocalError as e:
        # if the available_releases variable doesn't exist, this error will show up:
        # UnboundLocalError: local variable 'available_releases' referenced before assignment
        raise ValueError(
            f'There are no data available for the data product {dpid} at the site {site}.')


def get_tile_bounds(file_url_df, all_bounds=False):
    """
    Extracts and calculates the bounding coordinates from a DataFrame of file
    names containing UTM coordinates.The input dataframe can be generated from
    the function get_file_urls.

    This function filters out file names that end with the data extensions
    ('.tif', '.h5', '.laz', '.zip') and extracts UTM x and y coordinates from
    the data file names. It calculates the minimum and maximum x and y
    coordinates to determine the bounding box of the tiles. Additionally, it
    returns a sorted list of unique x_y coordinate pairs.

    Parameters:
    - file_url_df (pd.DataFrame): A DataFrame containing a 'name' column with
      file names that include UTM coordinates.
    - all_bounds (bool, optional): A flag to indicate whether to return a list
      of all UTM coordinates. Default is set to False.

    Returns:
    - x_bounds (tuple): A tuple containing the minimum and maximum x coordinates (min_x, max_x).
    - y_bounds (tuple): A tuple containing the minimum and maximum y coordinates (min_y, max_y).
    - sorted_coords (list): A sorted list of unique (x, y) coordinate tuples extracted from the file names.

    Example:
    >>> file_url_df, releases = get_file_urls(site_year_urls, token=token)
    >>> x_bounds, y_bounds, sorted_coords = get_tile_bounds(file_url_df)
    """

    # Regular expression to match UTM coordinates in the format 'xxxxxx_yyyyyyy'
    utm_pattern = re.compile(r'(\d{6})_(\d{7})')

    # lists to store x and y coordinates
    x_coords = []
    y_coords = []
    unique_coords = set()

    # filter out rows where 'name' ends with '.tif' , '.h5' or '.laz'
    # this will exclude shapefiles, just in case they don't match
    data_df = file_url_df[file_url_df['name'].str.endswith(
        ('.tif', '.h5', '.laz', '.zip'))]

    # Iterate over each name in the DataFrame
    for name in data_df['name']:
        match = utm_pattern.search(name)
        if match:
            x, y = match.groups()
            x_coords.append(int(x))
            y_coords.append(int(y))
            unique_coords.add((int(x), int(y)))

    # Convert the set to a sorted list
    sorted_coords = sorted(unique_coords)

    # Calculate min and max for x and y coordinates
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    x_bounds = (min_x, max_x)
    y_bounds = (min_y, max_y)

    return x_bounds, y_bounds, sorted_coords


def get_aop_tile_extents(dpid,
                         site,
                         year,
                         token=None):
    """
    This function displays the tile extents for a given product, site, and year
    and returns a complete list of the UTM coordinates

    Some NEON sites have irregular polygon flight plan boundaries, or may have
    incomplete coverage in a given year, so the list of coordinates provides a
    complete picture of the available tiles.

    Parameters
    --------
    dpid: str
        The identifier of the NEON data product to pull, in the form DPL.PRNUM.REV, e.g. DP3.30015.001.
        This must be a DP3 (Level 3) data product to work.

    site: str
        The four-letter code of a single NEON site, e.g. 'CLBJ'.

    year: str or int
        The four-digit year of data collection.

    token: str, optional
        User-specific API token from data.neonscience.org user account. See
        https://data.neonscience.org/data-api/rate-limiting/ for details about
        API rate limits and user tokens.

    Returns
    --------
    sorted_coords: list, a sorted list of the UTM x, y pairs of all tiles
    prints the minimum and maximum UTM x and y coordinates for the SW corners of the tiles encompassing the full site

    Examples
    --------
    >>> tile_extents = get_aop_tile_extents(dpid="DP3.30015.001", site="MCRA", year="2021",all_bounds=True)
    # This returns a list of the UTM x,y extent for all CHM tiles at the site MCRA collected in 2021.
    # It also displays the minimum and maximum UTM Easting and Northing (x and y) values for this product - site -year.

    Created on Feb 17 2025

    @author: Bridget Hass

    """

    # raise value error and print message if dpid isn't formatted as expected
    validate_dpid(dpid)

    # raise value error and print message if dpid isn't a valid AOP L3 product
    validate_aop_l3_dpid(dpid)

    # raise value error and print message if field spectra data product is entered
    check_field_spectra_dpid(dpid)

    # raise value error and print message if site is not a 4-letter character
    site = site.upper()  # make site upper case (if it's not already)
    validate_site_format(site)

    # raise value error and print message if site is not a valid NEON site
    validate_neon_site(site)

    # raise value error and print message if year input is not valid
    year = str(year)  # cast year to string (if it's not already)
    validate_year(year)

    # if token is an empty string, set to None
    if token == '':
        token = None

    # query the products endpoint for the product requested
    response = get_api(
        "http://data.neonscience.org/api/v0/products/" + dpid, token)

    # exit function if response is None (eg. if no internet connection)
    if response is None:
        logging.info('No response from NEON API. Check internet connection')
        return

    # check that token was used
    if token and 'x-ratelimit-limit' in response.headers:
        check_token(response)
        # if response.headers['x-ratelimit-limit'] == '200':
        #     print('API token was not recognized. Public rate limit applied.\n')

    # get the request response dictionary
    response_dict = response.json()

    # error message if dpid is not an AOP data product
    check_aop_dpid(response_dict, dpid)

    # replace collocated site with the AOP site name it's published under
    site = get_shared_flights(site)

    # get the urls for months with data available, and subset to site & year
    site_year_urls = get_site_year_urls(response_dict, site, year)

    # error message if nothing is available
    if len(site_year_urls) == 0:
        logging.info(
            f"There are no {dpid} data available at the site {site} in {year}. \nTo display available dates for a given data product and site, use the function list_available_dates().")
        return

    # get file url dataframe for the available month urls
    file_url_df, releases = get_file_urls(site_year_urls, token=token)

    # get the number of files in the dataframe, if there are no files to download, return
    if len(file_url_df) == 0:
        # print("No data files found.")
        logging.info("No data files found.")
        return

    # corner_tiles = get_corner_tiles(file_url_df)
    x_bounds, y_bounds, sorted_coords = get_tile_bounds(file_url_df)

    logging.info(f'Easting Bounds: {x_bounds}')
    logging.info(f'Northing Bounds: {y_bounds}')

    # return the sorted_coords list
    return sorted_coords


# %%


def by_file_aop(dpid,
                site,
                year,
                include_provisional=False,
                check_size=True,
                savepath=None,
                chunk_size=1024,
                token=None):
    """
    This function queries the NEON API for AOP data by site, year, and product, and downloads all
    files found, preserving the original folder structure. It downloads files serially to
    avoid API rate-limit overload, which may take a long time.

    Parameters
    --------
    dpid: str
        The identifier of the NEON data product to pull, in the form DPL.PRNUM.REV, e.g. DP3.30001.001.

    site: str
        The four-letter code of a single NEON site, e.g. 'CLBJ'.

    year: str or int
        The four-digit year of data collection.

    include_provisional: bool, optional
        Should provisional data be downloaded? Defaults to False. See
        https://www.neonscience.org/data-samples/data-management/data-revisions-releases
        for details on the difference between provisional and released data.

    check_size: bool, optional
        Should the user approve the total file size before downloading? Defaults to True.
        If you have sufficient storage space on your local drive, when working
        in batch mode, or other non-interactive workflow, use check_size=False.

    savepath: str, optional
        The file path to download to. Defaults to None, in which case the working directory is used.

    chunk_size: integer, optional
        Size in bytes of chunk for chunked download. Defaults to 1024.

    token: str, optional
        User-specific API token from data.neonscience.org user account. See
        https://data.neonscience.org/data-api/rate-limiting/ for details about
        API rate limits and user tokens.

    Returns
    --------
    None; data are downloaded to the local directory specified.

    Examples
    --------
    >>> by_file_aop(dpid="DP3.30015.001", site="MCRA", year="2021", savepath="./test_download")
    # This will download 2021 canopy height model data from McRae Creek to the './test_download' directory.

    Notes
    --------
    The function creates a folder in the 'savepath' directory, containing all AOP files meeting the query criteria.
    If 'savepath' is not provided, data are downloaded to the working directory.

    Windows Path Length Limitations:
    When using this function to download files from the NEON data portal, you may encounter path length limitations 
    on Windows systems. Windows has a default maximum path length of 260 characters, which can cause download 
    functions to fail if this limit is exceeded. If the file path exceeds 260 characters on a Windows system, 
    the package will issue a warning.If you see this warning and no files are downloaded, either change your 
    working or savepath directory to be closer to the root directory, or enable long paths on Windows. You can 
    choose to ignore or filter these warnings using Python's warnings module if you prefer not to see them.

    Created on Feb 28 2024

    @author: Bridget Hass
    """

    # raise value error and print message if dpid isn't formatted as expected
    validate_dpid(dpid)

    # raise value error and print message if dpid isn't formatted as expected
    validate_aop_dpid(dpid)

    # raise value error and print message if field spectra data are attempted
    check_field_spectra_dpid(dpid)

    # raise value error and print message if site is not a 4-letter character
    site = site.upper()  # make site upper case (if it's not already)
    validate_site_format(site)

    # raise value error and print message if site is not a valid NEON site
    validate_neon_site(site)

    # raise value error and print message if year input is not valid
    year = str(year)  # cast year to string (if it's not already)
    validate_year(year)

    # if token is an empty string, set to None
    if token == '':
        token = None

    # query the products endpoint for the product requested
    response = get_api(
        "http://data.neonscience.org/api/v0/products/" + dpid, token)

    # exit function if response is None (eg. if no internet connection)
    if response is None:
        logging.info('No response from NEON API. Check internet connection')
        return

    # check that token was used
    if token and 'x-ratelimit-limit' in response.headers:
        check_token(response)
        # if response.headers['x-ratelimit-limit'] == '200':
        #     print('API token was not recognized. Public rate limit applied.\n')

    # get the request response dictionary
    response_dict = response.json()

    # error message if dpid is not an AOP data product
    check_aop_dpid(response_dict, dpid)

    # replace collocated site with the AOP site name it's published under
    site = get_shared_flights(site)

    # get the urls for months with data available, and subset to site & year
    site_year_urls = get_site_year_urls(response_dict, site, year)

    # error message if nothing is available
    if len(site_year_urls) == 0:
        logging.info(
            f"There are no {dpid} data available at the site {site} in {year}.\nTo display available dates for a given data product and site, use the function list_available_dates().")
        # print("There are no data available at the selected site and year.")
        return

    # get file url dataframe for the available month urls
    file_url_df, releases = get_file_urls(site_year_urls, token=token)

    # get the number of files in the dataframe, if there are no files to download, return
    if len(file_url_df) == 0:
        # print("No data files found.")
        logging.info("No data files found.")
        return

    # if 'PROVISIONAL' in releases and not include_provisional:
    if include_provisional:
        # log provisional included message
        logging.info(
            "Provisional data are included. To exclude provisional data, use input parameter include_provisional=False.")
    else:
        # log provisional not included message and filter to the released data
        # logging.info(
        #     "Provisional data are not included. To download provisional data, use input parameter include_provisional=True.")
        file_url_df = file_url_df[file_url_df['release'] != 'PROVISIONAL']
        if len(file_url_df) == 0:
            logging.info(
                "Provisional data are not included. To download provisional data, use input parameter include_provisional=True.")

    num_files = len(file_url_df)
    if num_files == 0:
        logging.info(
            "No data files found. Available data may all be provisional. To download provisional data, use input parameter include_provisional=True.")
        return

    # get the total size of all the files found
    download_size_bytes = file_url_df['size'].sum()
    # print(f'download size, bytes: {download_size_bytes}')
    download_size = convert_byte_size(download_size_bytes)
    # print(f'download size: {download_size}')

    # report data download size and ask user if they want to proceed
    if check_size:
        if input(f"Continuing will download {num_files} files totaling approximately {download_size}. Do you want to proceed? (y/n) ") != "y":
            print("Download halted.")
            return

    # create folder in working directory to put files in
    if savepath is not None:
        download_path = savepath + "/" + dpid
    else:
        download_path = os.getcwd() + "/" + dpid
    os.makedirs(download_path, exist_ok=True)

    # serially download all files, with progress bar
    files = list(file_url_df['url'])
    print(
        f"Downloading {num_files} files totaling approximately {download_size}\n")
    sleep(1)
    for file in tqdm(files):
        download_file(url=file, savepath=download_path,
                      chunk_size=chunk_size, token=token)

    # download issue log table
    ilog = get_issue_log(dpid=dpid, token=None)
    if ilog is not None:
        ilog.to_csv(f"{download_path}/issueLog_{dpid}.csv", index=False)

    # download citations
    if "PROVISIONAL" in releases:
        try:
            cit = get_citation(dpid=dpid, release="PROVISIONAL")
            with open(f"{download_path}/citation_{dpid}_PROVISIONAL.txt",
                      mode="w+", encoding="utf-8") as f:
                f.write(cit)
        except Exception:
            pass

    rr = re.compile("RELEASE")
    rel = [r for r in releases if rr.search(r)]
    if len(rel) == 0:
        releases = releases
    if len(rel) == 1:
        try:
            cit = get_citation(dpid=dpid, release=rel[0])
            with open(f"{download_path}/citation_{dpid}_{rel[0]}.txt",
                      mode="w+", encoding="utf-8") as f:
                f.write(cit)
        except Exception:
            pass

    return

# %%


def by_tile_aop(dpid,
                site,
                year,
                easting,
                northing,
                buffer=0,
                include_provisional=False,
                check_size=True,
                savepath=None,
                chunk_size=1024,
                token=None,
                verbose=False):
    """
    This function queries the NEON API for AOP data by site, year, product, and
    UTM coordinates, and downloads all files found, preserving the original
    folder structure. It downloads files serially to avoid API rate-limit
    overload, which may take a long time.

    Parameters
    --------
    dpid: str
        The identifier of the NEON data product to pull, in the form DPL.PRNUM.REV, e.g. DP3.30001.001.

    site: str
        The four-letter code of a single NEON site, e.g. 'CLBJ'.

    year: str or int
        The four-digit year of data collection.

    easting: int or list of int
        A number or list containing the easting UTM coordinate(s) of the locations to download.

    northing: int or list of int
        A number or list containing the northing UTM coordinate(s) of the locations to download.

    buffer: int, optional
        Size, in meters, of the buffer to be included around the coordinates when determining which tiles to download. Defaults to 0.

    include_provisional: bool, optional
        Should provisional data be downloaded? Defaults to False. See
        https://www.neonscience.org/data-samples/data-management/data-revisions-releases
        for details on the difference between provisional and released data.

    check_size: bool, optional
        Should the user approve the total file size before downloading? Defaults to True.
        If you have sufficient storage space on your local drive, when working
        in batch mode, or other non-interactive workflow, use check_size=False.

    savepath: str, optional
        The file path to download to. Defaults to None, in which case the working directory is used.

    chunk_size: int, optional
        Size in bytes of chunk for chunked download. Defaults to 1024.

    token: str, optional
        User-specific API token from data.neonscience.org user account. See
        https://data.neonscience.org/data-api/rate-limiting/ for details about
        API rate limits and user tokens.

    verbose: bool, optional
        If set to True, the function will print information about the downloaded tiles.

    Return
    --------
    None; data are downloaded to the local directory specified.

    Example
    --------
    >>> by_tile_aop(dpid="DP3.30015.001", site="MCRA",
                    easting=[566456, 566639], northing=[4900783, 4901094],
                    year="2021", savepath="../../test_download")
    # This will download any tiles overlapping the specified UTM coordinates for
    # 2021 canopy height model data from McRae Creek to the './test_download' directory.

    Notes
    --------
    The function creates a folder in the 'savepath' directory, containing all AOP files meeting the query criteria.
    If 'savepath' is not provided, data are downloaded to the working directory.

    Windows Path Length Limitations:
    When using this function to download files from the NEON data portal, you may encounter path length limitations 
    on Windows systems. Windows has a default maximum path length of 260 characters, which can cause download 
    functions to fail if this limit is exceeded. If the file path exceeds 260 characters on a Windows system, 
    the package will issue a warning.If you see this warning and no files are downloaded, either change your 
    working or savepath directory to be closer to the root directory, or enable long paths on Windows. You can 
    choose to ignore or filter these warnings using Python's warnings module if you prefer not to see them.

    Created on Feb 28 2024

    @author: Bridget Hass

    """

    # raise value error and print message if dpid isn't formatted as expected
    validate_dpid(dpid)

    # raise value error and print message if dpid isn't a valid AOP L3 product
    validate_aop_l3_dpid(dpid)

    # raise value error and print message if field spectra data are attempted
    check_field_spectra_dpid(dpid)

    # raise value error and print message if site is not a 4-letter character
    site = site.upper()  # make site upper case (if it's not already)
    validate_site_format(site)

    # raise value error and print message if site is not a valid NEON site
    validate_neon_site(site)

    # raise value error and print message if year input is not valid
    year = str(year)  # cast year to string (if it's not already)
    validate_year(year)

    # convert easting and northing to lists, if they are not already
    if type(easting) is not list:
        easting = [easting]
    if type(northing) is not list:
        northing = [northing]

    # convert to floats, and display error message if easting and northing lists are not numeric
    try:
        easting = [float(e) for e in easting]
    except ValueError as e:
        logging.info(
            'The easting is invalid, this is required as a number or numeric list format, eg. 732000 or [732000, 733000]')
        print(e)

    try:
        northing = [float(e) for e in northing]
    except ValueError as e:
        logging.info(
            'The northing is invalid, this is required as a number or numeric list format, eg. 4713000 or [4713000, 4714000]')
        print(e)

    # link easting and northing coordinates - as a list of tuples ?
    # coord_tuples = [(easting[i], northing[i]) for i in range(0, len(easting))]

    # error message if easting and northing vector lengths don't match (also handles empty/NA cases)
    # there should not be any strings now that everything has been converted to a float
    easting = [e for e in easting if not np.isnan(e)]
    northing = [n for n in northing if not np.isnan(n)]

    if len(easting) != len(northing):
        logging.info(
            'Easting and northing list lengths do not match, and/or contain null values. Cannot identify paired coordinates.')
        return

    # if token is an empty string, set to None
    if token == '':
        token = None

    # query the products endpoint for the product requested
    response = get_api(
        "http://data.neonscience.org/api/v0/products/" + dpid, token)

    # exit function if response is None (eg. if no internet connection)
    if response is None:
        logging.info('No response from NEON API. Check internet connection')
        return

#   # check that token was used
    if token and 'x-ratelimit-limit' in response.headers:
        check_token(response)

    # get the request response dictionary
    response_dict = response.json()
    # error message if dpid is not an AOP data product
    if response_dict['data']['productScienceTeamAbbr'] != 'AOP':
        print(
            f'{dpid} is not a remote sensing product. Use zipsByProduct()')
        return

    # replace collocated site with the site name it's published under
    site = get_shared_flights(site)

    # get the urls for months with data available, and subset to site & year
    site_year_urls = get_site_year_urls(response_dict, site, year)

    # error message if nothing is available
    if len(site_year_urls) == 0:
        logging.info(
            f"There are no {dpid} data available at the site {site} in {year}.\nTo display available dates for a given data product and site, use the function list_available_dates().")
        return

    # get file url dataframe for the available month url(s)
    file_url_df, releases = get_file_urls(site_year_urls, token=token)

    # get the number of files in the dataframe, if there are no files to download, return
    if len(file_url_df) == 0:
        logging.info("No data files found.")
        return

    # if 'PROVISIONAL' in releases and not include_provisional:
    if include_provisional:
        # print provisional included message
        logging.info(
            "Provisional data are included. To exclude provisional data, use input parameter include_provisional=False.")
    else:
        # print provisional not included message
        file_url_df = file_url_df[file_url_df['release'] != 'PROVISIONAL']
        logging.info(
            "Provisional data are not included. To download provisional data, use input parameter include_provisional=True.")

        # get the number of files in the dataframe after filtering for provisional data, if there are no files to download, return
        num_files = len(file_url_df)
        if num_files == 0:
            logging.info(
                "No data files found. Available data may all be provisional. To download provisional data, use input parameter include_provisional=True.")
            return

    # BLAN edge-case - contains plots in 18N and plots in 17N; flight data are all in 17N
    # convert easting & northing coordinates for Blandy (BLAN) to 17N

    if site == 'BLAN' and any([e <= 250000.0 for e in easting]):
        # check that pyproj is installed
        try:
            from pyproj import Proj, CRS
            # importlib.import_module('pyproj')
        except ImportError:
            logging.info(
                "Package pyproj is required for this function to work at the BLAN site. Install and re-try")
            return

        crs17 = CRS.from_epsg(32617)  # utm zone 17N
        crs18 = CRS.from_epsg(32618)  # utm zone 18N

        proj18to17 = Proj.from_crs(crs_from=crs18, crs_to=crs17)

        # link easting and northing coordinates so it's easier to parse the zone for each
        coord_tuples = [(easting[i], northing[i])
                        for i in range(0, len(easting))]

        coords17 = [(e, n) for (e, n) in coord_tuples if e > 250000.0]
        coords18 = [(e, n) for (e, n) in coord_tuples if e <= 250000.0]

        # apply the projection transformation from 18N to 17N for each coordinate tuple

        coords18_reprojected = [proj18to17.transform(
            coords18[i][0], coords18[i][1]) for i in range(len(coords18))]

        coords17.extend(coords18_reprojected)

        # re-set easting and northing
        easting = [c[0] for c in coords17]
        northing = [c[1] for c in coords17]

        logging.info('Blandy (BLAN) plots include two UTM zones, flight data '
                     'are all in 17N. Coordinates in UTM zone 18N have been '
                     'converted to 17N to download the correct tiles. You '
                     'will need to make the same conversion to connect '
                     'airborne to ground data.')

    # function to round down to the nearest 1000, in order to determine
    # lower left coordinate of AOP tile to be downloaded
    def round_down1000(val):
        return int(np.floor(val/1000)*1000)

    # function to get the coordinates of the tiles including the buffer
    def get_buffer_coords(easting, northing, buffer):
        # apply the buffer to the easting and northings
        buffer_min_e = easting - buffer
        buffer_min_n = northing - buffer
        buffer_max_e = easting + buffer
        buffer_max_n = northing + buffer

        new_coords = [(buffer_min_e, buffer_min_n), (buffer_min_e, buffer_max_n),
                      (buffer_max_e, buffer_min_n), (buffer_max_e, buffer_max_n)]

        return new_coords

    # get the tiles corresponding to the new coordinates (mins and maxes)
    # if verbose:
    #     print('getting coordinates of tiles, including the buffer')
    buffer_coords = []
    for e, n in zip(easting, northing):
        buffer_coords.extend(get_buffer_coords(e, n, buffer))

    buffer_coords_rounded = [
        (round_down1000(c[0]), round_down1000(c[1])) for c in buffer_coords]
    # remove duplicate coordinates
    buffer_coords_set = list(set(buffer_coords_rounded))
    buffer_coords_set.sort()

    utm17_eastings_str = ', '.join([str(round(e, 2)) for e in easting])
    utm17_northings_str = ', '.join(str(round(n, 2)) for n in northing)

    if verbose:
        logging.info(f'UTM 17N Easting(s): {utm17_eastings_str}')
        logging.info(f'UTM 17N Northing(s): {utm17_northings_str}')
        # logging.info('Buffer:', buffer)
        logging.info(
            'UTM (x, y) lower-left coordinates of tiles to be downloaded:')
        for coord in buffer_coords_set:
            logging.info(coord)

    # create the list of utm "easting_northing" strings that will be used to match to the tile names
    coord_strs = ['_'.join([str(c[0]), str(c[1])])
                  for c in buffer_coords_set]

    # append the .txt file to include the README - IS THIS NEEDED?
    coord_strs.append('.txt')

    # subset the dataframe to include only the coordinate strings matching coord_strs
    # if verbose:
    #     print('finding the tiles')
    file_url_df_subset = file_url_df[file_url_df['name'].str.contains(
        '|'.join(coord_strs))]

    file_url_df_subset2 = file_url_df_subset[~file_url_df_subset['name'].str.endswith(
        '.txt')]

    # if any coordinates were not included in the data, print a warning message
    # Warning: the following coordinates are outside the bounds of site-year:
    unique_coords_to_download = set(
        file_url_df_subset2['name'].str.extract(r'_(\d+_\d+)_')[0])

    coord_strs.remove('.txt')
    # compare two lists:
    coords_not_found = list(
        set(coord_strs).difference(list(unique_coords_to_download)))
    if len(coords_not_found) > 0:
        print('Warning, the following coordinates fall outside the bounds of the site, so will not be downloaded:')
        for coord in coords_not_found:
            print(','.join(coord.split('_')))

    # get the number of files in the dataframe, if there are no files to download, return
    num_files = len(file_url_df_subset)
    if num_files == 0:
        print("No data files found.")
        return

    # get the total size of all the files found
    download_size_bytes = file_url_df_subset['size'].sum()

    download_size = convert_byte_size(download_size_bytes)

    # ask whether to continue download, depending on size
    if check_size:
        if input(f"Continuing will download {num_files} files totaling approximately {download_size}. Do you want to proceed? (y/n) ") != "y":
            print("Download halted")
            return

    # create folder in working directory to put files in
    if savepath is not None:
        download_path = savepath + "/" + dpid
    else:
        download_path = os.getcwd() + "/" + dpid
    # print('download path', download_path)
    os.makedirs(download_path, exist_ok=True)

    # serially download all files, with progress bar
    files = list(file_url_df_subset['url'])
    print(
        f"Downloading {num_files} files totaling approximately {download_size}\n")
    sleep(1)
    for file in tqdm(files):
        download_file(url=file, savepath=download_path,
                      chunk_size=chunk_size, token=token)

    # download issue log table
    ilog = get_issue_log(dpid=dpid, token=None)
    if ilog is not None:
        ilog.to_csv(f"{download_path}/issueLog_{dpid}.csv", index=False)

    # download citations
    if "PROVISIONAL" in releases:
        try:
            cit = get_citation(dpid=dpid, release="PROVISIONAL")
            with open(f"{download_path}/citation_{dpid}_PROVISIONAL.txt",
                      mode="w+", encoding="utf-8") as f:
                f.write(cit)
        except Exception:
            pass

    rr = re.compile("RELEASE")
    rel = [r for r in releases if rr.search(r)]
    if len(rel) == 0:
        releases = releases
    if len(rel) == 1:
        try:
            cit = get_citation(dpid=dpid, release=rel[0])
            with open(f"{download_path}/citation_{dpid}_{rel[0]}.txt",
                      mode="w+", encoding="utf-8") as f:
                f.write(cit)
        except Exception:
            pass

    return
