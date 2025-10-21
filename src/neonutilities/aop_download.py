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

Updated Oct 2025 to include options to skip files if they already exist locally and 
to validate existing files against files published on the Data Portal with checksums.

"""

# %% imports
from datetime import datetime
import glob
import importlib_resources
import logging
import pandas as pd
import numpy as np
import os
import re

# import time
from time import sleep
from tqdm import tqdm

# local imports
from . import __resources__
from .helper_mods.api_helpers import get_api
from .helper_mods.api_helpers import download_file, calculate_crc32c
from .helper_mods.metadata_helpers import convert_byte_size
from .get_issue_log import get_issue_log
from .citation import get_citation

# display the log info messages, only showing the message (otherwise it would print INFO:root:'message')
logging.basicConfig(level=logging.INFO, format="%(message)s")


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

    if (
        "x-ratelimit-limit" in response.headers
        and response.headers["x-ratelimit-limit"] == "200"
    ):
        logging.info("API token was not recognized. Public rate limit applied.\n")


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
    The function makes API calls to each URL in the 'urls' list and retrieves the file
    information. It also retrieves the release information from the response JSON.
    If the API call fails, it prints a warning message and continues with the next URL.
    """

    all_file_url_df = pd.DataFrame()
    releases = []
    for url in urls:
        response = get_api(api_url=url, token=token)
        if response is None:
            logging.info(
                "NEON data file retrieval failed. Check NEON data portal for outage alerts."
            )

        # get release info
        release = response.json()["data"]["release"]
        releases.append(release)

        file_url_dict = response.json()["data"]["files"]
        file_url_df = pd.DataFrame(data=file_url_dict)
        file_url_df["release"] = release

        # drop md5 and crc32 columns, which are all NaNs
        file_url_df.drop(columns=["md5", "crc32"], inplace=True)

        # append the new dataframe to the existing one
        all_file_url_df = pd.concat([all_file_url_df, file_url_df], ignore_index=True)

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
    shared_flights_file = (
        importlib_resources.files(__resources__) / "shared_flights.csv"
    )

    shared_flights_df = pd.read_csv(shared_flights_file)

    shared_flights_dict = shared_flights_df.set_index(["site"])["flightSite"].to_dict()
    if site in shared_flights_dict:
        flightSite = shared_flights_dict[site]
        if site in ["TREE", "CHEQ", "KONA", "DCFS"]:
            logging.info(
                f"{site} is part of the NEON flight box for {flightSite}. Downloading data from {flightSite}."
            )
        else:
            logging.info(
                f"{site} is a NEON aquatic site and is sometimes included in the flight box for {flightSite}. Aquatic sites are not always included in the flight coverage every year.\nDownloading data from {flightSite}. Check data to confirm coverage of {site}."
            )
        site = flightSite
    return site


def get_neon_sites():
    """This function gets a list of the valid NEON sites from the
    neon_field_site_metadata.csv file for validation, and adds the AOP CHEQ
    site, which is an AOP site name and is associated with STEI & TREE."
    """
    neon_sites_file = (
        importlib_resources.files(__resources__)
        / "neon_field_site_metadata_20250214.csv"
    )
    neon_sites_df = pd.read_csv(neon_sites_file)

    neon_sites_list = list(neon_sites_df["field_site_id"])
    neon_sites_list.append(
        "CHEQ"
    )  # append the CHEQ site, a separate flight area than STEI-TREE

    return neon_sites_list


def get_data_product_name(dpid):
    dpid_api_response = get_api(f"https://data.neonscience.org/api/v0/products/{dpid}")
    product_name = dpid_api_response.json()["data"]["productName"]
    return product_name


def check_exists_and_checksum(row, download_path):
    """
    Check if a file exists locally and if its CRC32C checksum matches the expected value.
    Used in the skip_if_exists option of by_tile_aop and by_file_aop.

    Parameters
    ----------
    row : pandas.Series
        A row from a DataFrame containing at least a 'url' and optionally a 'crc32c' field.
    download_path : str
        The root directory where files are downloaded.

    Returns
    -------
    pandas.Series
        A Series of two boolean values:
        - exists_locally: True if the file exists locally, False otherwise.
        - checksum_matches: True if the local file's CRC32C matches the expected value, False otherwise.

    Example
    -------
    >>> file_url_df[["exists_locally", "checksum_matches"]] = file_url_df.apply(
    ...     lambda row: check_exists_and_checksum(row, download_path), axis=1
    ... )
    """
    # print(row["url"])
    pathparts = row["url"].split("/")
    file_path = os.path.join(download_path, *pathparts[3:])
    # file_path = os.path.join(download_path,"\\".join(pathparts[3:]))
    # print(file_path)
    exists = os.path.exists(file_path)
    # print(exists)
    if not exists:
        # print(f'{pathparts[-1:]} does not exist!')
        return pd.Series([False, False])
    expected_crc32c = row.get("crc32c")
    # print(expected_crc32c)
    if expected_crc32c:
        local_crc32c = calculate_crc32c(file_path)
        matches = local_crc32c.zfill(8) == str(expected_crc32c).zfill(8)
        # return pd.Series({"exists_locally": True, "checksum_matches": matches})
        return pd.Series([True, matches])
    return pd.Series([True, False])
    # return pd.Series({"exists_locally": True, "checksum_matches": False})


# %% functions to validate inputs for by_file_aop and by_tile_aop


def validate_dpid(dpid):
    dpid_pattern = "DP[1-4]{1}.[0-9]{5}.00[1-2]{1}"
    if not re.fullmatch(dpid_pattern, dpid):
        raise ValueError(
            f"{dpid} is not a properly formatted NEON data product ID. The correct format is DP#.#####.00#"
        )


# %% function to get all AOP data product IDs (active = valid, inactive = future / suspended)


def get_aop_dpids():
    """
    This function retrieves all active and inactive AOP data product IDs from the NEON API.

    Returns
    -------
    active_dpids: list
        A list of all active AOP data product IDs (productStatus = "ACTIVE")

    inactive_dpids: list
        A list of all inactive AOP data product IDs (productStatus = "FUTURE" / suspended products)

    Raises
    ------
    None

    Examples
    --------
    >>> active_dpids = get_active_dpids()
    # This will return a list of all active NEON data product IDs.
    """
    response = get_api("https://data.neonscience.org/api/v0/products")

    response_dict = response.json()
    # all_neon_dpids = [item["productCode"] for item in response_dict["data"]]
    # active_neon_dpids = [item["productCode"] for item in response_dict["data"] if item["productStatus"] == "ACTIVE"]
    active_aop_dpids = [
        item["productCode"]
        for item in response_dict["data"]
        if (
            item["productScienceTeamAbbr"] == "AOP"
            and item["productStatus"] == "ACTIVE"
        )
    ]
    inactive_aop_dpids = [
        item["productCode"]
        for item in response_dict["data"]
        if (
            item["productScienceTeamAbbr"] == "AOP"
            and item["productStatus"] == "FUTURE"
        )
    ]

    return active_aop_dpids, inactive_aop_dpids


def validate_aop_dpid(dpid):
    """
    Validates the given AOP data product ID against a pattern and a list of active DPIDs, determined from the API.

    Parameters:
    - dpid (str): The data product ID to validate.

    Raises:
    - ValueError: If the dpid does not match the expected pattern or is not in the list of active (valid) DPIDs.
    """
    # Regular expression pattern for AOP DPIDs
    aop_dpid_pattern = "DP[1-3]{1}.300[0-2]{1}[0-9]{1}.00[1-2]{1}"

    # Check if the dpid matches the pattern
    if not re.fullmatch(aop_dpid_pattern, dpid):
        raise ValueError(
            f"{dpid} is not a valid NEON AOP data product ID. AOP data products follow the format DP#.300##.00#."
        )

    active_aop_dpids, inactive_aop_dpids = get_aop_dpids()

    # Check if the dpid is in the list of suspended (FUTURE) AOP DPIDs
    if dpid in inactive_aop_dpids:
        raise ValueError(
            f"NEON {dpid} has been suspended and is not currently available, see https://data.neonscience.org/data-products/{dpid} for more details."
        )

    # Check if the dpid is in the list of valid AOP DPIDs
    if dpid not in active_aop_dpids:
        active_aop_dpids.sort()
        valid_aop_dpids_string = "\n".join(active_aop_dpids)
        raise ValueError(
            f"NEON {dpid} is not a valid AOP data product ID. Valid AOP IDs are listed below:\n{valid_aop_dpids_string}"
        )


def validate_aop_l3_dpid(dpid):
    """
    Validates the given AOP data product ID against expected pattern to check if it is downloadable by tile.
    If the dpid does not match the expected pattern or is not in the list of active Level 3 AOP data product IDs,
    it will  raise a value error with a descriptive message.

    Parameters:
    - dpid (str): The data product ID to validate.

    Raises:
    - ValueError: If the dpid is not in the list of AOP data product IDs that are downloadable by tile.
    """
    # Check if the dpid starts with DP3 or is DP1.30003.001
    if not (dpid.startswith("DP3") or dpid == "DP1.30003.001"):
        raise ValueError(
            f"NEON {dpid} is not a valid Level 3 (L3) AOP data product ID. Level 3 AOP products follow the format DP3.300##.00#, with the exception of DP1.30003.001 (the discrete classified lidar point cloud data product)."
        )

    # Get the list of active AOP data product IDs
    active_aop_dpids, _ = get_aop_dpids()

    # List of valid Level 3 AOP data product IDs
    valid_aop_l3_dpids = [
        dpid
        for dpid in active_aop_dpids
        if dpid.startswith("DP3") or dpid == "DP1.30003.001"
    ]

    # Check if the dpid is in the list of valid AOP dpids
    if dpid not in valid_aop_l3_dpids:
        valid_aop_l3_dpids.sort()
        valid_aop_l3_dpids_string = "\n".join(valid_aop_l3_dpids)

        raise ValueError(
            f"NEON {dpid} is not a valid Level 3 (L3) AOP data product ID. Valid L3 AOP IDs are listed below:\n{valid_aop_l3_dpids_string}"
        )
        # below prints out the corresponding data product names for each ID.
        # f'{dpid} is not a valid Level 3 (L3) AOP data product ID. Valid L3 AOP products are listed below.\n{formatted_dpid_dict}')


def check_field_spectra_dpid(dpid):
    if dpid == "DP1.30012.001":
        raise ValueError(
            f"NEON {dpid} is the Field spectral data product, which is published as tabular data. Use zips_by_product() or loadByProduct() to download these data."
        )


def validate_site_format(site):
    site_pattern = "[A-Z]{4}"
    if not re.fullmatch(site_pattern, site):
        raise ValueError(
            f"{site} is an invalid NEON site format. A four-letter NEON site code is required. NEON site codes can be found here: https://www.neonscience.org/field-sites/explore-field-sites"
        )


def validate_neon_site(site):
    neon_sites = get_neon_sites()

    if site not in neon_sites:
        raise ValueError(
            f"{site} is not a valid NEON site code. A complete list of NEON site codes can be found here: https://www.neonscience.org/field-sites/explore-field-sites"
        )


def validate_year(year):
    # year = str(year)
    # year_pattern = "20?(1[2-9]|2[0-9])"
    # if not re.fullmatch(year_pattern, year):
    #     raise ValueError(
    #         f'{year} is an invalid year. Year is required in the format "2017" or 2017, eg. NEON AOP data are available from 2013 to present.'
    #     )
    """
    Validates that the year is between 2012 and the current year.
    """
    # First, check with regex for exactly 4 digits
    if not re.fullmatch(r"\d{4}", str(year)):
        raise ValueError(
            f"{year} is an invalid year. Year is required in the format '2017' or 2017, e.g. NEON AOP data are available from 2012 to present."
        )

    # Convert year to an integer and check if it's between 2012 and the current year
    year_int = int(year)
    current_year = datetime.now().year
    if not (2012 <= year_int <= current_year):
        raise ValueError(
            f"{year} is an invalid year. Year must be between 2012 and {current_year}."
        )


def validate_overwrite(overwrite):
    """
    Validates that overwrite is one of the accepted options: 'yes', 'no', or 'prompt'.
    Raises a ValueError if not.
    """
    valid_options = {"yes", "no", "prompt"}
    if overwrite not in valid_options:
        raise ValueError(f"overwrite must be one of {valid_options}. ")


def validate_skip_if_exists(skip_if_exists):
    """
    Validates that skip_if_exists is a boolean (True or False).
    Raises a ValueError if not.
    """
    if not isinstance(skip_if_exists, bool):
        raise ValueError(
            f"skip_if_exists must be a boolean (True or False). "
            f"Received: '{skip_if_exists}' of type {type(skip_if_exists)}"
        )


def check_aop_dpid(response_dict, dpid):
    if response_dict["data"]["productScienceTeamAbbr"] != "AOP":
        logging.info(
            f"NEON {dpid} is not a remote sensing product. Use zipsByProduct()"
        )
        return


def get_site_year_urls(response_dict, site, year):
    site_info = next(
        item for item in response_dict["data"]["siteCodes"] if item["siteCode"] == site
    )
    site_urls = site_info["availableDataUrls"]
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
    --------
        Usage:
        --------
        >>> list_available_dates('DP3.30015.001','JORN')
        RELEASE-2025 Available Dates: 2017-08, 2018-08, 2019-08, 2021-08, 2022-09

        >>> list_available_dates('DP3.30015.001','HOPB')
        PROVISIONAL Available Dates: 2024-09
        RELEASE-2025 Available Dates: 2016-08, 2017-08, 2019-08, 2022-08

        >>> list_available_dates('DP1.10098.001','HOPB')
        ValueError: There are no data available for the data product DP1.10098.001 at the site HOPB.
    """
    product_url = "https://data.neonscience.org/api/v0/products/" + dpid
    response = get_api(api_url=product_url)  # add input for token?

    # raise value error and print message if dpid isn't formatted as expected
    validate_dpid(dpid)

    # raise value error and print message if site is not a 4-letter character
    site = site.upper()  # make site upper case (if it's not already)
    validate_site_format(site)

    # raise value error and print message if site is not a valid NEON site
    validate_neon_site(site)

    # check if product is active
    if response.json()["data"]["productStatus"] != "ACTIVE":
        raise ValueError(
            f"NEON {dpid} is not an active data product. See https://data.neonscience.org/data-products/{dpid} for more details."
        )

    # get available releases & months:
    for i in range(len(response.json()["data"]["siteCodes"])):
        if site in response.json()["data"]["siteCodes"][i]["siteCode"]:
            available_releases = response.json()["data"]["siteCodes"][i][
                "availableReleases"
            ]

    # display available release tags (including provisional) and dates for each tag
    try:
        for entry in available_releases:
            release = entry["release"]
            available_months = ", ".join(entry["availableMonths"])
            logging.info(f"{release} Available Dates: {available_months}")
    except UnboundLocalError:
        # if the available_releases variable doesn't exist, this error will show up:
        # UnboundLocalError: local variable 'available_releases' referenced before assignment
        raise ValueError(
            f"There are no NEON data available for the data product {dpid} at the site {site}."
        )


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
    utm_pattern = re.compile(r"(\d{6})_(\d{7})")

    # lists to store x and y coordinates
    x_coords = []
    y_coords = []
    unique_coords = set()

    # filter out rows where 'name' ends with '.tif' , '.h5' or '.laz'
    # this will exclude shapefiles, just in case they don't match
    data_df = file_url_df[
        file_url_df["name"].str.endswith((".tif", ".h5", ".laz", ".zip"))
    ]

    # Iterate over each name in the DataFrame
    for name in data_df["name"]:
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


def get_aop_tile_extents(dpid, site, year, token=None):
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
    if token == "":
        token = None

    # query the products endpoint for the product requested
    response = get_api("https://data.neonscience.org/api/v0/products/" + dpid, token)

    # exit function if response is None (eg. if no internet connection)
    if response is None:
        logging.info("No response from NEON API. Check internet connection")
        return

    # check that token was used
    if token and "x-ratelimit-limit" in response.headers:
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
            f"There are no NEON {dpid} data available at the site {site} in {year}. \nTo display available dates for a given data product and site, use the function list_available_dates()."
        )
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

    logging.info(f"Easting Bounds: {x_bounds}")
    logging.info(f"Northing Bounds: {y_bounds}")

    # return the sorted_coords list
    return sorted_coords


# %%


def by_file_aop(
    dpid,
    site,
    year,
    include_provisional=False,
    check_size=True,
    savepath=None,
    chunk_size=1024,
    token=None,
    verbose=False,
    skip_if_exists=False,
    overwrite="prompt",
):
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
        User-specific API token from data.neonscience.org user account. Defaults to None.
        See https://data.neonscience.org/data-api/rate-limiting/ for details about API
        rate limits and user tokens.

    verbose: bool, optional
        If set to True, the function will print more detailed information about the download process.

    skip_if_exists: bool, optional
        If set to True, the function will skip downloading files that already exist in the
        savepath and are valid (local checksums match the checksums of the published file).
        Defaults to False. If any local file checksums don't match those of files published
        on the NEON Data Portal, the user will be prompted to skip these files or overwrite
        the existing files with the new ones (see overwrite input).

    overwrite: str, optional
        Must be one of:
            'yes'    - overwrite mismatched files without prompting,
            'no'     - don't overwrite mismatched files (skip them, no prompt),
            'prompt' - prompt the user (y/n) to overwrite mismatched files after displaying them (default).
        If skip_if_exists is False, this parameter is ignored, and any existing files in
        the savepath will be overwritten according to the function's default behavior.

    Returns
    --------
    None; data are downloaded to the directory specified (savepath) or the current working directory.
    If data already exist in the expected path, they will be overwritten by default. To check for
    existing files before downloading, set skip_if_exists=True along with an overwrite option (y/n/prompt).

    Examples
    --------
    >>> by_file_aop(dpid="DP3.30015.001",
                    site="MCRA",
                    year="2021",
                    savepath="./test_download",
                    skip_if_exists=True)
    # This downloads the 2021 Canopy Height Model data from McRae Creek to the './test_download' directory.
    # If any files already exist in the savepath, they will be checked and skipped if they are valid.
    # The user will be prompted to ovewrite or skip downloading any existing files that do not match
    # the latest published data on the NEON Data Portal.

    Notes
    --------
    This function creates a folder named by the Data Product ID (DPID; e.g. DP3.30015.001) in the
    'savepath' directory, containing all AOP files meeting the query criteria. If 'savepath' is
    not provided, data are downloaded to the working directory, in a folder named by the DPID.
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

    # raise value error and print message if skip_if_exists input is not valid (boolean)
    validate_skip_if_exists(skip_if_exists)

    # raise value error and print message if validate_overwrite input is not valid (yes, no, prompt)
    validate_overwrite(overwrite)

    # warn if overwrite is set but skip_if_exists is False
    if not skip_if_exists and overwrite != "prompt":
        logging.info(
            "WARNING: overwrite option only applies if skip_if_exists=True. "
            "By default, any existing files will be overwritten unless you select skip_if_exists=True and overwrite='no' or 'prompt' (default)."
        )

    # if token is an empty string, set to None
    if token == "":
        token = None

    # query the products endpoint for the product requested
    response = get_api("https://data.neonscience.org/api/v0/products/" + dpid, token)

    # exit function if response is None (eg. if no internet connection)
    if response is None:
        logging.info("No response from NEON API. Check internet connection")
        return

    # check that token was used
    if token and "x-ratelimit-limit" in response.headers:
        check_token(response)

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
            f"There are no NEON {dpid} data available at the site {site} in {year}.\nTo display available dates for a given data product and site, use the function list_available_dates()."
        )
        # print("There are no data available at the selected site and year.")
        return

    # get file url dataframe for the available month urls
    file_url_df, releases = get_file_urls(site_year_urls, token=token)

    # get the number of files in the dataframe, if there are no files to download, return
    if len(file_url_df) == 0:
        logging.info("No NEON data files found.")
        return

    # if 'PROVISIONAL' in releases and not include_provisional:
    if include_provisional:
        # log provisional included message
        logging.info(
            "Provisional NEON data are included. To exclude provisional data, use input parameter include_provisional=False."
        )
    else:
        # log provisional not included message and filter to the released data
        file_url_df = file_url_df[file_url_df["release"] != "PROVISIONAL"]
        if len(file_url_df) == 0:
            logging.info(
                "Provisional NEON data are not included. To download provisional data, use input parameter include_provisional=True."
            )

    num_files = len(file_url_df)
    if num_files == 0:
        logging.info(
            "No NEON data files found. Available data may all be provisional. To download provisional data, use input parameter include_provisional=True."
        )
        return

    # get the total size of all the files found
    download_size_bytes = file_url_df["size"].sum()
    # print(f'download size, bytes: {download_size_bytes}')
    download_size = convert_byte_size(download_size_bytes)
    # print(f'download size: {download_size}')

    # report data download size and ask user if they want to proceed
    if check_size:
        if (
            input(
                f"Continuing will download {num_files} NEON data files totaling approximately {download_size}. Do you want to proceed? (y/n) "
            )
            .strip()
            .lower()  # lower or upper case 'y' will work
            != "y"
        ):
            print("Download halted.")
            return

    # create folder in working directory to put files in
    if savepath is not None:
        download_path = savepath + "/" + dpid
    else:
        download_path = os.getcwd() + "/" + dpid
    os.makedirs(download_path, exist_ok=True)

    # serially download all files, with progress bar
    files = list(file_url_df["url"])

    # different messages depending on whether skip_if_exists is True or False
    if skip_if_exists:
        logging.info(
            f"Found {num_files} NEON data files totaling approximately {download_size}.\n"
            "Files in savepath will be checked and skipped if they exist and match the latest version."
        )
    else:
        logging.info(
            f"Downloading {num_files} NEON data files totaling approximately {download_size}\n"
        )
        # verbose option to list all files being downloaded
        if verbose:
            logging.info("Downloading the following data and metadata files:")
            for file in files:
                logging.info(file.replace("https://storage.googleapis.com", f"{dpid}"))

    if skip_if_exists:
        # Existence and checksum check
        file_url_df[["exists_locally", "checksum_matches"]] = file_url_df.apply(
            lambda row: check_exists_and_checksum(row, download_path), axis=1
        )

        # Handle the various cases
        # 1. Skip files that exist locally and checksums match those on the GCS (exists_locally and checksum_matches are both True).
        # 2. Prompt the user to decide whether to download (overwrite) files that exist locally but checksums don't match (exists_locally is True and checksum_matches is False).
        # 3. Download files if they don't already exist locally (exists_locally is False).
        # 4. If there are extra files locally that don't exist on the GCS, display a warning message.

        # Skipped files
        files_to_skip = file_url_df[
            (file_url_df["exists_locally"]) & (file_url_df["checksum_matches"])
        ]

        # Files to prompt for overwrite
        mismatched_files = file_url_df[
            (file_url_df["exists_locally"]) & (~file_url_df["checksum_matches"])
        ]

        # Files to download (do not exist locally)
        files_to_download = file_url_df[~file_url_df["exists_locally"]]

        # Identify README files (case-insensitive, .txt)
        readme_files = file_url_df[
            file_url_df["name"].str.contains("readme", case=False, na=False)
        ]

        # If any files are missing or mismatched, download the README file
        if (
            not files_to_download.empty or not mismatched_files.empty
        ) and not readme_files.empty:
            logging.info("Downloading README file")
            for idx, row in tqdm(readme_files.iterrows(), total=len(readme_files)):
                download_file(
                    url=row["url"],
                    savepath=download_path,
                    chunk_size=chunk_size,
                    token=token,
                )

        # display a warning message if there are extra files locally that
        # are not published on the Portal as part of the Data Product

        # get all expected file paths (relative to download_path)
        expected_files = set()
        for _, row in file_url_df.iterrows():
            pathparts = row["url"].split("/")
            expected_files.add(
                os.path.normpath(os.path.join(download_path, *pathparts[3:]))
            )

        # get the set of all AOP bucket names from the URLs
        gcs_bucket = {
            url.split("/")[3]
            for url in file_url_df["url"]
            if len(url.split("/")) > 3
            and url.split("/")[3]
            in ["neon-aop-products", "neon-aop-provisional-products"]
        }

        # get the domain from the URLs
        domain = {
            val
            for url in file_url_df["url"]
            for val in [url.split("/")[6]]
            if re.fullmatch(r"D\d{2}", val)
        }

        # get the Year_Site_Visit from the URLs
        ysv = {
            val
            for url in file_url_df["url"]
            for val in [url.split("/")[7]]
            if re.fullmatch(r"\d{4}_[A-Z]{4}_\d+", val)
        }

        # recursively find all files in the expected path
        local_path = os.path.normpath(
            os.path.join(
                download_path,
                gcs_bucket.pop(),
                str(year),
                "FullSite",
                domain.pop(),
                ysv.pop(),
            )
        )

        # get the set of all local files (normalized paths)
        all_local_files = set(
            os.path.normpath(f)
            for f in glob.glob(os.path.join(local_path, "**"), recursive=True)
            if os.path.isfile(f)
        )

        # find extra files and display warning message if any exist
        extra_files = sorted(all_local_files - expected_files)
        if extra_files:
            logging.info(
                f"WARNING: Files found in the local folder that do not exist in the latest version of data for {dpid}. "
                f"\nYou will need to delete the extra files below if you want the download folder to match the latest available data contents,"
                f"\nor to be safe, delete the folder {local_path} and re-download."
                "\nExtra Files Found:"
            )
            for f in sorted(extra_files):
                logging.info(f"  {os.path.abspath(f)}")

        # files that do not exist locally
        if not files_to_download.empty:
            logging.info(
                "The following files will be downloaded (they do not already exist locally):"
            )
            for f in files_to_download["name"].sort_values():
                logging.info(f"  {f}")

            # Download files that do not exist locally
            for _, row in tqdm(
                files_to_download.iterrows(), total=len(files_to_download)
            ):
                download_file(
                    url=row["url"],
                    savepath=download_path,
                    chunk_size=chunk_size,
                    token=token,
                )

            if not files_to_skip.empty:
                logging.info(
                    "The remainder of the files in savepath will not be downloaded. "
                    "They already exist locally and match the latest available data."
                )

        # mismatched files (local checksums are not the same as those on the Data Portal)
        # filter out files where the name contains "readme" (case-insensitive)
        # readme files do not have a checksum so would always fail, these should be downloaded by default
        mismatched_files_no_readme = mismatched_files[
            ~mismatched_files["name"].str.contains("readme", case=False, na=False)
        ]

        # message if there is nothing to download - all files exist and they all match the checksums
        if files_to_download.empty and mismatched_files_no_readme.empty:
            logging.info(
                "All files already exist locally and match the latest available data. Skipping download."
            )

        if not mismatched_files_no_readme.empty:
            logging.info(
                "The following files exist locally but have a different checksum than the remote files:"
            )
            for f in sorted(mismatched_files_no_readme["name"]):
                logging.info(f"  {f}")
            # determine whether to overwrite mismatched files
            if overwrite == "yes":
                response = "y"
            elif overwrite == "no":
                response = "n"
                # logging.info("WARNING: Files where the checksum doesn't match the latest version will not be overwritten.")
            else:  # 'prompt' or any other value
                response = (
                    input(
                        "Do you want to overwrite these files with the latest version? (y/n) "
                    )
                    .strip()
                    .lower()
                )
            if response.lower() == "y":
                logging.info("Overwriting these files with the latest available data.")
                # download including the readme
                for _, row in tqdm(
                    mismatched_files.iterrows(), total=len(mismatched_files)
                ):
                    download_file(
                        url=row["url"],
                        savepath=download_path,
                        chunk_size=chunk_size,
                        token=token,
                    )
            else:
                logging.info("Skipped overwriting files with mismatched checksums.")

    else:
        for file in tqdm(files):
            download_file(
                url=file, savepath=download_path, chunk_size=chunk_size, token=token
            )

    # download issue log table
    ilog = get_issue_log(dpid=dpid, token=None)
    if ilog is not None:
        ilog.to_csv(f"{download_path}/issueLog_{dpid}.csv", index=False)

    # download citations
    if "PROVISIONAL" in releases:
        try:
            cit = get_citation(dpid=dpid, release="PROVISIONAL")
            with open(
                f"{download_path}/citation_{dpid}_PROVISIONAL.txt",
                mode="w+",
                encoding="utf-8",
            ) as f:
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
            with open(
                f"{download_path}/citation_{dpid}_{rel[0]}.txt",
                mode="w+",
                encoding="utf-8",
            ) as f:
                f.write(cit)
        except Exception:
            pass

    return


# %%


def by_tile_aop(
    dpid,
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
    verbose=False,
    skip_if_exists=False,
    overwrite="prompt",
):
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

    easting: float/int or list of float/int
        A number or list containing the easting UTM coordinate(s) of the locations to download.

    northing: float/int or list of float/int
        A number or list containing the northing UTM coordinate(s) of the locations to download.

    buffer: float/int, optional
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
        Files are downloaded to subdirectories starting with the dpid under the savepath.

    chunk_size: int, optional
        Size in bytes of chunk for chunked download. Defaults to 1024.

    token: str, optional
        User-specific API token from data.neonscience.org user account. Defaults to None.
        See https://data.neonscience.org/data-api/rate-limiting/ for details about
        API rate limits and user tokens.

    verbose: bool, optional
        If set to True, the function will print out a list of the tiles to be downloaded before downloading.
        Defaults to False.

    skip_if_exists: bool, optional
        If set to True, the function will skip downloading files that already exist in the
        savepath and are valid (local checksums match the checksums of the published file).
        Defaults to False. If any local file checksums don't match those of files published
        on the NEON Data Portal, the user will be prompted to skip these files or overwrite
        the existing files with the new ones (see overwrite input).

    overwrite: str, optional
        Must be one of:
            'yes'    - overwrite mismatched files without prompting,
            'no'     - don't overwrite mismatched files (skip them, no prompt),
            'prompt' - prompt the user (y/n) to overwrite mismatched files after displaying them (default).
        If skip_if_exists is False, this parameter is ignored, and any existing files in
        the savepath will be overwritten according to the function's default behavior.

    Returns
    --------
    None; data are downloaded to the directory specified (savepath) or the current working directory.
    If data already exist in the expected path, they will be overwritten by default. To check for
    existing files before downloading, set skip_if_exists=True along with an overwrite option (y/n/prompt).

    Example
    --------
    >>> by_tile_aop(dpid="DP3.30015.001",
                    site="MCRA",
                    easting=[566456, 566639],
                    northing=[4900783, 4901094],
                    year="2021",
                    savepath="./test_download",
                    skip_if_exists=True)
    # This downloads any tiles overlapping the specified UTM coordinates for
    # 2021 canopy height model data from McRae Creek to the './test_download' directory.
    # If any files already exist in the savepath, they will be checked and skipped if they are valid.
    # The user will be prompted to ovewrite or skip downloading any existing files that do not match
    # the latest published data on the NEON Data Portal.

    Notes
    --------
    This function creates a folder named by the Data Product ID (DPID; e.g. DP3.30015.001) in the
    'savepath' directory, containing all AOP files meeting the query criteria. If 'savepath' is
    not provided, data are downloaded to the working directory, in a folder named by the DPID.
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

    # raise value error and print message if skip_if_exists input is not valid (boolean)
    validate_skip_if_exists(skip_if_exists)

    # raise value error and print message if validate_overwrite input is not valid (yes, no, prompt)
    validate_overwrite(overwrite)

    # warn if overwrite is set to yes or no, but skip_if_exists is False
    if not skip_if_exists and overwrite != "prompt":
        logging.info(
            "Warning: overwrite option only applies if skip_if_exists=True. By default, "
            "any existing files in the expected directory will be overwritten unless "
            "you select skip_if_exists=True and overwrite='no' or 'prompt' (default)."
        )

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
            "The easting is invalid, this is required as a number or numeric list format, eg. 732000 or [732000, 733000]"
        )
        print(e)

    try:
        northing = [float(e) for e in northing]
    except ValueError as e:
        logging.info(
            "The northing is invalid, this is required as a number or numeric list format, eg. 4713000 or [4713000, 4714000]"
        )
        print(e)

    # link easting and northing coordinates

    # error message if easting and northing vector lengths don't match (also handles empty/NA cases)
    # there should not be any strings now that everything has been converted to a float
    easting = [e for e in easting if not np.isnan(e)]
    northing = [n for n in northing if not np.isnan(n)]

    if len(easting) != len(northing):
        logging.info(
            "Easting and northing list lengths do not match, and/or contain null values. Cannot identify paired coordinates."
        )
        return

    # if token is an empty string, set to None
    if token == "":
        token = None

    # query the products endpoint for the product requested
    response = get_api("https://data.neonscience.org/api/v0/products/" + dpid, token)

    # exit function if response is None (eg. if no internet connection)
    if response is None:
        logging.info("No response from NEON API. Check internet connection")
        return

    # check that token was used
    if token and "x-ratelimit-limit" in response.headers:
        check_token(response)

    # get the request response dictionary
    response_dict = response.json()

    # error message if dpid is not an AOP data product
    if response_dict["data"]["productScienceTeamAbbr"] != "AOP":
        logging.info(
            f"NEON {dpid} is not a remote sensing product. Use zips_by_product()"
        )
        return

    # replace collocated site with the site name it's published under
    site = get_shared_flights(site)

    # get the urls for months with data available, and subset to site & year
    site_year_urls = get_site_year_urls(response_dict, site, year)

    # error message if nothing is available
    if len(site_year_urls) == 0:
        logging.info(
            f"There are no NEON {dpid} data available at the site {site} in {year}.\nTo display available dates for a given data product and site, use the function list_available_dates()."
        )
        return

    # get file url dataframe for the available month url(s)
    file_url_df, releases = get_file_urls(site_year_urls, token=token)

    # get the number of files in the dataframe, if there are no files to download, return
    if len(file_url_df) == 0:
        logging.info("No NEON data files found.")
        return

    # if 'PROVISIONAL' in releases and not include_provisional:
    if include_provisional:
        # print provisional included message
        logging.info(
            "Provisional NEON data are included. To exclude provisional data, use input parameter include_provisional=False."
        )
    else:
        # print provisional not included message
        file_url_df = file_url_df[file_url_df["release"] != "PROVISIONAL"]
        logging.info(
            "Provisional NEON data are not included. To download provisional data, use input parameter include_provisional=True."
        )

        # get the number of files in the dataframe after filtering for provisional data, if there are no files to download, return
        num_files = len(file_url_df)
        if num_files == 0:
            logging.info(
                "No NEON data files found. Available data may all be provisional. To download provisional data, use input parameter include_provisional=True."
            )
            return

    # BLAN edge-case - contains plots in 18N and plots in 17N; flight data are all in 17N
    # convert easting & northing coordinates for Blandy (BLAN) to 17N

    if site == "BLAN" and any([e <= 250000.0 for e in easting]):
        # print('BLAN SITE - CONVERTING COORDINATES FROM 18N TO 17N')
        # check that pyproj is installed
        try:
            from pyproj import Proj, CRS
        except ImportError:
            logging.info(
                "Package pyproj is required for this function to work at the NEON BLAN site. Install and re-try."
            )
            return

        crs17 = CRS.from_epsg(32617)  # utm zone 17N
        crs18 = CRS.from_epsg(32618)  # utm zone 18N

        proj18to17 = Proj.from_crs(crs_from=crs18, crs_to=crs17)

        # link easting and northing coordinates so it's easier to parse the zone for each
        coord_tuples = [(easting[i], northing[i]) for i in range(0, len(easting))]

        coords17 = [(e, n) for (e, n) in coord_tuples if e > 250000.0]
        coords18 = [(e, n) for (e, n) in coord_tuples if e <= 250000.0]

        # apply the projection transformation from 18N to 17N for each coordinate tuple

        coords18_reprojected = [
            proj18to17.transform(coords18[i][0], coords18[i][1])
            for i in range(len(coords18))
        ]

        coords17.extend(coords18_reprojected)

        # re-set easting and northing
        easting = [c[0] for c in coords17]
        northing = [c[1] for c in coords17]

        logging.info(
            "Blandy (BLAN) plots include two UTM zones, flight data "
            "are all in 17N. Coordinates in UTM zone 18N have been "
            "converted to 17N to download the correct tiles. You "
            "will need to make the same conversion to connect "
            "airborne to ground data."
        )

    # function to round down to the nearest 1000, in order to determine
    # lower left coordinate of AOP tile to be downloaded
    def round_down1000(val):
        return int(np.floor(val / 1000) * 1000)

    # function to get the coordinates of the tiles including the buffer
    def get_buffer_coords(easting, northing, buffer):
        # apply the buffer to the easting and northings
        buffer_min_e = easting - buffer
        buffer_min_n = northing - buffer
        buffer_max_e = easting + buffer
        buffer_max_n = northing + buffer

        new_coords = [
            (buffer_min_e, buffer_min_n),
            (buffer_min_e, buffer_max_n),
            (buffer_max_e, buffer_min_n),
            (buffer_max_e, buffer_max_n),
        ]

        return new_coords

    # get the tiles corresponding to the new coordinates (mins and maxes)
    buffer_coords = []
    for e, n in zip(easting, northing):
        buffer_coords.extend(get_buffer_coords(e, n, buffer))

    buffer_coords_rounded = [
        (round_down1000(c[0]), round_down1000(c[1])) for c in buffer_coords
    ]
    # remove duplicate coordinates
    buffer_coords_set = list(set(buffer_coords_rounded))
    buffer_coords_set.sort()

    utm17_eastings_str = ", ".join([str(round(e, 2)) for e in easting])
    utm17_northings_str = ", ".join(str(round(n, 2)) for n in northing)

    if site == "BLAN" and verbose:
        logging.info(f"UTM 17N Easting(s): {utm17_eastings_str}")
        logging.info(f"UTM 17N Northing(s): {utm17_northings_str}")
        # logging.info('Buffer:', buffer)
    if verbose:
        logging.info("UTM (x, y) lower-left coordinates of tiles to be downloaded:")
        for coord in buffer_coords_set:
            logging.info(coord)

    # create the list of utm "easting_northing" strings that will be used to match to the tile names
    coord_strs = ["_".join([str(c[0]), str(c[1])]) for c in buffer_coords_set]

    # append the .txt file to include the README - IS THIS NEEDED?
    coord_strs.append(".txt")

    # subset the dataframe to include only the coordinate strings matching coord_strs
    file_url_df_subset = file_url_df[
        file_url_df["name"].str.contains("|".join(coord_strs))
    ]

    # remove .txt files (README) and create a copy to use for the checksum/existence checks
    file_url_df_subset2 = file_url_df_subset[
        ~file_url_df_subset["name"].str.endswith(".txt")
    ].copy()

    # file_url_df_subset2.reset_index(drop=True,inplace=True)
    file_url_df_subset2 = file_url_df_subset2.reset_index(drop=True)

    # print(list(file_url_df_subset2.head(1)))
    # print(file_url_df_subset2.head())

    # if any coordinates were not included in the data, print a warning message
    unique_coords_to_download = set(
        file_url_df_subset2["name"].str.extract(r"_(\d{6}_\d{7})_")[0]
    )

    coord_strs.remove(".txt")
    # print('coordinates:'); print(coord_strs)
    # print('unique coordinates to download:'); print(unique_coords_to_download)
    # find the coordinates that were not found in the data (difference between the two lists):
    coords_not_found = list(set(coord_strs).difference(list(unique_coords_to_download)))
    if len(coords_not_found) > 0:
        logging.info(
            "Warning: the following coordinates fall outside the bounds of the NEON site, so will not be downloaded:"
        )
        for coord in coords_not_found:
            print(",".join(coord.split("_")))

    # get the number of files in the dataframe, if there are no files to download, return
    num_files = len(file_url_df_subset)
    if num_files == 0:
        logging.info(f"Warning: No NEON {dpid} files found.")
        return

    # get the total size of all the files found
    download_size_bytes = file_url_df_subset["size"].sum()

    # convert to a human-readable format
    download_size = convert_byte_size(download_size_bytes)

    # ask whether to continue download, depending on size
    if check_size:
        if (
            input(
                f"Continuing will download {num_files} NEON data files totaling approximately {download_size}. Do you want to proceed? (y/n) "
            )
            .strip()
            .lower()
            != "y"
        ):
            print("Download halted")
            return

    # create folder in working directory to put files in
    if savepath is not None:
        download_path = savepath + "/" + dpid
    else:
        download_path = os.getcwd() + "/" + dpid
    # print('download path', download_path)
    os.makedirs(download_path, exist_ok=True)

    # Get the sorted list of the files to download
    # use the files from the subsetted dataframe
    files = list(file_url_df_subset["url"])
    files.sort()  # sort the files for consistent download order

    # different messages depending on whether skip_if_exists is True or False
    if skip_if_exists:
        logging.info(
            f"Found {num_files} NEON data files totaling approximately {download_size}.\n"
            "Files in savepath will be checked and skipped if they exist and match the latest version."
        )
    else:
        logging.info(
            f"Downloading {num_files} NEON data files totaling approximately {download_size}\n"
        )
        if verbose:
            logging.info("Downloading the following data and metadata files:")
            for file in files:
                logging.info(file.replace("https://storage.googleapis.com", f"{dpid}"))

    if skip_if_exists:
        # Check which files already exist locally and if their checksums match
        file_url_df_subset2[
            ["exists_locally", "checksum_matches"]
        ] = file_url_df_subset2.apply(
            lambda row: check_exists_and_checksum(row, download_path), axis=1
        )

        # Files to skip (already exist locally and checksums match)
        files_to_skip = file_url_df_subset2[
            (file_url_df_subset2["exists_locally"])
            & (file_url_df_subset2["checksum_matches"])
        ]

        # Files to prompt for overwrite (exists locally but checksums don't match)
        mismatched_files = file_url_df_subset2[
            (file_url_df_subset2["exists_locally"])
            & (~file_url_df_subset2["checksum_matches"])
        ]

        # Files to download (do not exist locally)
        files_to_download = file_url_df_subset2[~file_url_df_subset2["exists_locally"]]
        # print(file_url_df_subset2[["name","exists_locally","checksum_matches"]])

        # if verbose: # print these even in non-verbose mode
        if not files_to_download.empty:
            logging.info(
                "The following files will be downloaded (they do not already exist locally):"
            )
            for f in files_to_download["name"].sort_values():
                logging.info(f"  {f}")

            # Download files that do not exist locally
            for _, row in tqdm(
                files_to_download.iterrows(), total=len(files_to_download)
            ):
                download_file(
                    url=row["url"],
                    savepath=download_path,
                    chunk_size=chunk_size,
                    token=token,
                )

            if not files_to_skip.empty:
                logging.info(
                    "The remainder of the files in savepath will not be downloaded. "
                    "They already exist locally and match the latest available data."
                )

        # prompt for mismatched files (excluding readme.txt)
        mismatched_files_no_readme = mismatched_files[
            ~mismatched_files["name"].str.contains("readme", case=False, na=False)
        ]

        # identify README files (case-insensitive, usually .txt), there should only be one of these
        readme_files = mismatched_files[
            mismatched_files["name"].str.contains("readme", case=False, na=False)
        ]

        # download README files
        if not readme_files.empty:
            logging.info("Downloading README file")
            for _, row in tqdm(readme_files.iterrows(), total=len(readme_files)):
                download_file(
                    url=row["url"],
                    savepath=download_path,
                    chunk_size=chunk_size,
                    token=token,
                )

        # message if there is nothing to download - all files exist and they all match the checksums
        if files_to_download.empty and mismatched_files_no_readme.empty:
            logging.info(
                "All files already exist locally and match the latest available data. Skipping download."
            )

        if not mismatched_files_no_readme.empty:
            logging.info(
                "The following files exist locally but have a different checksum than the remote files:"
            )
            for f in sorted(mismatched_files_no_readme["name"]):
                logging.info(f"  {f}")
            # determine whether to overwrite mismatched files
            if overwrite == "yes":
                response = "y"
            elif overwrite == "no":
                response = "n"
            else:  # 'prompt' or any other value
                response = (
                    input(
                        "Do you want to overwrite these files with the latest version? (y/n) "
                    )
                    .strip()
                    .lower()
                )

            if response.lower() == "y":
                logging.info("Overwriting these files with the latest available data.")
                for idx, row in tqdm(
                    mismatched_files.iterrows(), total=len(mismatched_files)
                ):
                    download_file(
                        url=row["url"],
                        savepath=download_path,
                        chunk_size=chunk_size,
                        token=token,
                    )
            else:
                logging.info("Skipped overwriting files with mismatched checksums.")
    else:  # if skip_if_exists=False (default behavior)
        for file in tqdm(files):
            download_file(
                url=file, savepath=download_path, chunk_size=chunk_size, token=token
            )

    # download issue log table
    ilog = get_issue_log(dpid=dpid, token=None)
    if ilog is not None:
        ilog.to_csv(f"{download_path}/issueLog_{dpid}.csv", index=False)

    # download citations
    if "PROVISIONAL" in releases:
        try:
            cit = get_citation(dpid=dpid, release="PROVISIONAL")
            with open(
                f"{download_path}/citation_{dpid}_PROVISIONAL.txt",
                mode="w+",
                encoding="utf-8",
            ) as f:
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
            with open(
                f"{download_path}/citation_{dpid}_{rel[0]}.txt",
                mode="w+",
                encoding="utf-8",
            ) as f:
                f.write(cit)
        except Exception:
            pass

    return
