#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import re
import os
import time
import platform
import importlib.metadata
import logging
import warnings
import pandas as pd
from tqdm import tqdm
from .metadata_helpers import get_recent

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Set global user agent
vers = importlib.metadata.version("neonutilities")
plat = platform.python_version()
osplat = platform.platform()

usera = f"neonutilities/{vers} Python/{plat} {osplat}"


def get_api(api_url, token=None):
    """

    Accesses the API with options to use the user-specific API token generated within neon.datascience user accounts.

    Parameters
    --------
    api_url: The API endpoint URL.
    token: User specific API token (generated within neon.datascience user accounts). Optional.

    Return
    --------
    API GET response containing status code and data that can be parsed into a json file

    Example
    --------
    Get the sample class for a surface water particulate stable isotope sample

    >>> sample_get = get_api(api_url='https://data.neonscience.org/api/v0/samples/classes?sampleTag=MCRA.SS.20230425.POM.1',token=None)

    Created on Fri Aug 30 2023

    @author: Zachary Nickerson
    """

    def get_status_code_meaning(status_code):
        return requests.status_codes._codes[status_code][0]

    # Check internet connection
    try:
        check_connection = requests.get(
            "https://data.neonscience.org/", headers={"User-Agent": usera}
        )
        if check_connection.status_code != 200:
            status_code = check_connection.status_code
            status_code_meaning = get_status_code_meaning(status_code)
            raise ConnectionError(
                f"Request failed with status code {status_code}, indicating '{status_code_meaning}'\n"
            )
    except Exception:  # ConnectionError as e
        raise ConnectionError("Connection error. Cannot access NEON API.\n")

    # Make 5 request attempts. If the rate limit is reached, pause for the
    # burst reset time to try again.
    j = 1

    while j <= 5:
        # Try making the request
        try:
            # Construct URL either with or without token
            if token is None:
                response = requests.get(
                    api_url, headers={"accept": "application/json", "User-Agent": usera}
                )
            else:
                response = requests.get(
                    api_url,
                    headers={
                        "X-API-TOKEN": token,
                        "accept": "application/json",
                        "User-Agent": usera,
                    },
                )

            # Check for successful response
            if response.status_code == 200:
                if "x-ratelimit-limit" in response.headers:
                    # Retry get request if rate limit is reached
                    limit_remain = response.headers.get("x-ratelimit-remaining")

                    if int(limit_remain) < 1:
                        # Wait for the reset time
                        time_reset = response.headers.get("x-ratelimit-reset")
                        logging.info(
                            f"Rate limit reached. Pausing for {time_reset} seconds to reset.\n"
                        )
                        time.sleep(int(time_reset))
                        # Increment loop to retry request attempt
                        j += 1

                    else:
                        # If rate limit is not reached, exit out of loop
                        j += 5

                else:
                    # If x-ratelimit-limit not found in headers, exit out of
                    # loop (don't need to retry because of rate-limit)
                    j += 5

            else:
                # Return nothing if request failed (status code is not 200)
                # Print the status code and it's meaning
                status_code_meaning = get_status_code_meaning(response.status_code)
                raise ConnectionError(
                    f"Request failed with status code {response.status_code}, indicating '{status_code_meaning}'\n"
                )

            return response

        except Exception as error:
            print(error)
            return None


def get_api_headers(api_url, token=None):
    """

    Accesses the API with options to use the user-specific API token generated within neon.datascience user accounts.

    Parameters
    --------
    api_url: The API endpoint URL.
    token: User specific API token (generated within neon.datascience user accounts). Optional.

    Return
    --------
    The header only from an API GET response

    Created on Feb 26 2024

    Adapted from get_api
    @author: Zachary Nickerson
    @author: Claire Lunch
    """

    def get_status_code_meaning(status_code):
        return requests.status_codes._codes[status_code][0]

    # Check internet connection
    try:
        check_connection = requests.head(
            "https://data.neonscience.org/", headers={"User-Agent": usera}
        )
        if check_connection.status_code != 200:
            status_code = check_connection.status_code
            status_code_meaning = get_status_code_meaning(status_code)
            raise ConnectionError(
                f"Request failed with status code {status_code}, indicating '{status_code_meaning}'\n"
            )
    except Exception:  # ConnectionError as e
        raise ConnectionError(
            "No internet connection detected. Cannot access NEON API.\n"
        )

    # Make 5 request attempts. If the rate limit is reached, pause for the
    # burst reset time to try again.
    j = 1

    while j <= 5:
        # Try making the request
        try:
            # Construct URL either with or without token
            if token is None:
                response = requests.head(
                    api_url, headers={"accept": "application/json", "User-Agent": usera}
                )
            else:
                response = requests.head(
                    api_url,
                    headers={
                        "X-API-TOKEN": token,
                        "accept": "application/json",
                        "User-Agent": usera,
                    },
                )

            # Check for successful response
            if response.status_code == 200:
                if "x-ratelimit-limit" in response.headers:
                    # Retry get request if rate limit is reached
                    limit_remain = response.headers.get("x-ratelimit-remaining")

                    if int(limit_remain) < 1:
                        # Wait for the reset time
                        time_reset = response.headers.get("x-ratelimit-reset")
                        logging.info(
                            f"Rate limit reached. Pausing for {time_reset} seconds to reset.\n"
                        )
                        time.sleep(int(time_reset))
                        # Increment loop to retry request attempt
                        j += 1

                    else:
                        # If rate limit is not reached, exit out of loop
                        j += 5

                else:
                    # If x-ratelimit-limit not found in headers, exit out of
                    # loop (don't need to retry because of rate-limit)
                    j += 5

            else:
                # Return nothing if request failed (status code is not 200)
                # Print the status code and it's meaning
                status_code_meaning = get_status_code_meaning(response.status_code)
                raise ConnectionError(
                    f"Request failed with status code {response.status_code}, indicating '{status_code_meaning}'\n"
                )

            return response

        except Exception:
            raise ConnectionError(
                "No response. NEON API may be unavailable, check NEON data portal for outage alerts. If the problem persists and can't be traced to an outage alert, check your computer for firewall or other security settings preventing Python from accessing the internet."
            )


def get_zip_urls(
    url_set, package, release, include_provisional, token=None, progress=True
):
    """

    Given a set of urls to the data endpoint of the NEON API, returns the set of zip file urls for each site-month package. Internal function, called by zips_by_product().

    Parameters
    --------
    url_set: A list of urls pointing to the data endpoint of the NEON API
    package: Data download package, basic or expanded.
    release: Data release to download.
    include_provisional: Should Provisional data be returned in the download?
    token: User specific API token (generated within neon.datascience user accounts). Optional.
    progress: Should the progress bar be displayed?

    Return
    --------
    List of urls pointing to zip files for each product-site-month.

    Created on Feb 23 2024

    @author: Claire Lunch
    """

    flnm = []
    z = []
    sz = []
    rel = []
    provflag = False
    if progress:
        logging.info("Finding available files")

    for i in tqdm(range(0, len(url_set)), disable=not progress):
        # get list of files from data endpoint
        m_res = get_api(api_url=url_set[i], token=token)
        if m_res is None:
            logging.info(
                "Connection error for a subset of urls. Check outputs for missing data."
            )
            return None
        m_di = m_res.json()

        # only keep queried release
        if release != "current":
            if release != m_di["data"]["release"]:
                continue

        # if include_provisional=F, exclude provisional
        if not include_provisional:
            if m_di["data"]["release"] == "PROVISIONAL":
                provflag = True
                continue

        # check for no files
        if "packages" not in list(m_di["data"]):
            logging.info(
                f"No files found for site {m_di['data']['siteCode']} and month {m_di['data']['month']}"
            )
            continue

        if len(m_di["data"]["packages"]) == 0:
            logging.info(
                f"No files found for site {m_di['data']['siteCode']} and month {m_di['data']['month']}"
            )
            continue

        # if package=expanded, check for expanded. reassign to basic if not found.
        if package == "expanded":
            if package not in [p["type"] for p in m_di["data"]["packages"]]:
                logging.info(
                    f"No expanded package found for site {m_di['data']['siteCode']} and month {m_di['data']['month']}. Basic package downloaded instead."
                )
                package = "basic"

        # get zip file url and file name
        zi = [u["url"] for u in m_di["data"]["packages"] if u["type"] == package]
        h = get_api_headers(api_url=zi[0], token=token)
        fltp = re.sub(pattern='"', repl="", string=h.headers["content-disposition"])
        flnmi = re.sub(pattern="inline; filename=", repl="", string=fltp)

        # get file sizes
        szr = re.compile(package)
        flszs = [siz["size"] for siz in m_di["data"]["files"] if szr.search(siz["url"])]
        flszi = sum(flszs)

        # return url, file name, file size, and release
        flnm.append(flnmi)
        z.append(zi)
        sz.append(flszi)
        rel.append(m_di["data"]["release"])

    z = sum(z, [])
    zpfiles = dict(flnm=flnm, z=z, sz=sz, rel=rel)

    # provisional message
    if provflag:
        logging.info(
            "Provisional data were excluded from available files list. To download provisional data, use input parameter include_provisional=True."
        )

    return zpfiles


def get_tab_urls(
    url_set,
    package,
    release,
    include_provisional,
    timeindex,
    tabl,
    token=None,
    progress=True,
):
    """

    Given a set of urls to the data endpoint of the NEON API, and averaging interval or table name criteria, returns the set of urls to individual files for each site-month package. Internal function, called by zips_by_product().

    Parameters
    --------
    url_set: A list of urls pointing to the data endpoint of the NEON API
    package: Data download package, basic or expanded.
    release: Data release to download.
    include_provisional: Should Provisional data be returned in the download?
    timeindex: Averaging interval of data to download.
    tabl: Table name of data to download.
    token: User specific API token (generated within neon.datascience user accounts). Optional.
    progress: Should the progress bar be displayed?

    Return
    --------
    List of urls pointing to files for each product-site-month and subset.

    Created on Mar 23 2024

    @author: Claire Lunch
    """

    # initiate file lists
    flnm = []
    flpth = []
    z = []
    sz = []
    rel = []
    varf = []
    rdme = []
    sp = []

    # create regular expressions for file finding
    vr = re.compile("variables")
    rdr = re.compile("readme")
    spr = re.compile("sensor_positions")

    if timeindex != "all":
        tt = re.compile(
            str(timeindex) + "min|" + str(timeindex) + "_min|science_review_flags"
        )

    if tabl != "all":
        tb = re.compile("[.]" + tabl + "[.]")

    provflag = False
    if progress:
        logging.info("Finding available files")

    for i in tqdm(range(0, len(url_set)), disable=not progress):
        # get list of files from data endpoint
        m_res = get_api(api_url=url_set[i], token=token)
        if m_res is None:
            logging.info(
                "Connection error for a subset of urls. Check outputs for missing data."
            )
            return None
        m_di = m_res.json()

        # only keep queried release
        if release != "current":
            if release != m_di["data"]["release"]:
                continue

        # if include_provisional=F, exclude provisional
        if not include_provisional:
            if m_di["data"]["release"] == "PROVISIONAL":
                provflag = True
                continue

        # subset to package. switch to basic if expanded not available
        # package name isn't always in file name (lab files, SRFs) but is always in url
        pr = re.compile(package)
        flsp = [f for f in m_di["data"]["files"] if pr.search(f["url"])]
        if package == "expanded" and len(flsp) == 0:
            pr = re.compile("basic")
            flsp = [f for f in m_di["data"]["files"] if pr.search(f["url"])]

        # check for no files
        if len(flsp) == 0:
            logging.info(
                f"No files found for site {m_di['data']['siteCode']} and month {m_di['data']['month']}"
            )
            continue

        # get zip file url and file name
        zi = [u["url"] for u in m_di["data"]["packages"] if u["type"] == package]
        h = get_api_headers(api_url=zi[0], token=token)
        fltp = re.sub(pattern='"', repl="", string=h.headers["content-disposition"])
        flpthit = re.sub(pattern="inline; filename=", repl="", string=fltp)
        flpthi = re.sub(pattern=".zip", repl="/", string=flpthit)

        # make separate lists of variables, readme and sensor positions
        varfi = [f for f in m_di["data"]["files"] if vr.search(f["name"])]
        rdmei = [f for f in m_di["data"]["files"] if rdr.search(f["name"])]
        spi = [f for f in m_di["data"]["files"] if spr.search(f["name"])]
        for f in varfi:
            f["name"] = flpthi + f["name"]
        for f in rdmei:
            f["name"] = flpthi + f["name"]

        varf.append(varfi)
        rdme.append(rdmei)
        if len(spi) > 0:
            for f in spi:
                f["name"] = flpthi + f["name"]
            sp.append(spi)

        # subset by averaging interval
        if timeindex != "all":
            flnmi = [flpthi + fl["name"] for fl in flsp if tt.search(fl["name"])]
            flszi = [fl["size"] for fl in flsp if tt.search(fl["name"])]
            zi = [fl["url"] for fl in flsp if tt.search(fl["name"])]

            # check for no files
            if len(flnmi) == 0:
                logging.info(
                    f"No files found for site {m_di['data']['siteCode']}, month {m_di['data']['month']}, and averaging interval (time index) {timeindex}"
                )
                continue

        # subset by table
        if tabl != "all":
            flnmi = [flpthi + fl["name"] for fl in flsp if tb.search(fl["name"])]
            flszi = [fl["size"] for fl in flsp if tb.search(fl["name"])]
            zi = [fl["url"] for fl in flsp if tb.search(fl["name"])]

            # check for no files
            if len(flnmi) == 0:
                logging.info(
                    f"No files found for site {m_di['data']['siteCode']}, month {m_di['data']['month']}, and table {tabl}"
                )
                continue

        # return url, file name, file size, and release
        flnm.append(flnmi)
        flpth.append(flpthi)
        z.append(zi)
        sz.append(flszi)
        rel.append(m_di["data"]["release"])

    # get most recent metadata files from lists
    try:
        varf = sum(varf, [])
        varfl = get_recent(varf, "variables")
        flnm.append([fl["name"] for fl in varfl])
        z.append([fl["url"] for fl in varfl])
        sz.append([fl["size"] for fl in varfl])
    except Exception:
        pass

    try:
        rdme = sum(rdme, [])
        rdfl = get_recent(rdme, "readme")
        flnm.append([fl["name"] for fl in rdfl])
        z.append([fl["url"] for fl in rdfl])
        sz.append([fl["size"] for fl in rdfl])
    except Exception:
        pass

    # get most recent sensor positions file for each site
    if len(sp) > 0:
        sp = sum(sp, [])
        sr = re.compile("[/][A-Z]{4}[/]")
        sites = [sr.search(f["url"]).group(0) for f in sp]
        sites = list(set(sites))
        sites = [re.sub(pattern="/", repl="", string=s) for s in sites]
        try:
            for s in sites:
                spfl = get_recent(sp, s)
                flnm.append([fl["name"] for fl in spfl])
                z.append([fl["url"] for fl in spfl])
                sz.append([fl["size"] for fl in spfl])
        except Exception:
            pass

    z = sum(z, [])
    flnm = sum(flnm, [])
    sz = sum(sz, [])
    tbfiles = dict(flnm=flnm, flpth=flpth, z=z, sz=sz, rel=rel)

    # provisional message
    if provflag:
        logging.info(
            "Provisional data were excluded from available files list. To download provisional data, use input parameter include_provisional=True."
        )

    return tbfiles


def download_urls(url_set, outpath, token=None, progress=True):
    """

    Given a set of urls to NEON data packages or files, downloads the contents of each. Internal function, called by zips_by_product().

    Parameters
    --------
    url_set: A list of urls pointing to zipped data packages
    outpath: Filepath of the folder to download to
    token: User specific API token (generated within neon.datascience user accounts). Optional.
    progress: Should the progress bar be displayed?

    Return
    --------
    Files in the designated folder

    Created on Feb 28 2024

    @author: Claire Lunch
    """

    if progress:
        logging.info("Downloading files")

    for i in tqdm(range(0, len(url_set["z"])), disable=not progress):
        if len(outpath + url_set["flnm"][i]) > 260 and platform.system() == "Windows":
            raise OSError(
                f'Filepath is {len(outpath+url_set["flnm"][i])} characters long. Filepaths on Windows are limited to 260 characters. Move your working directory closer to the root directory or enable long path support in Windows through the Registry Editor.'
            )
            return

        else:
            try:
                if token is None:
                    j = 0
                    while j < 3:
                        try:
                            with open(outpath + url_set["flnm"][i], "wb") as out_file:
                                content = requests.get(
                                    url_set["z"][i],
                                    stream=True,
                                    headers={
                                        "accept": "application/json",
                                        "User-Agent": usera,
                                    },
                                    timeout=(10, 120),
                                ).content
                                out_file.write(content)
                            j = j + 5
                        except Exception as e:
                            logging.info(
                                f"File {url_set['flnm'][i]} could not be downloaded. Re-attempting."
                            )
                            print(e)
                            j = j + 1
                            time.sleep(5)
                else:
                    j = 0
                    while j < 3:
                        try:
                            with open(outpath + url_set["flnm"][i], "wb") as out_file:
                                content = requests.get(
                                    url_set["z"][i],
                                    stream=True,
                                    headers={
                                        "X-API-TOKEN": token,
                                        "accept": "application/json",
                                        "User-Agent": usera,
                                    },
                                    timeout=(10, 120),
                                ).content
                                out_file.write(content)
                            j = j + 5
                        except Exception as e:
                            logging.info(
                                f"File {url_set['flnm'][i]} could not be downloaded. Re-attempting."
                            )
                            print(e)
                            j = j + 1
                            time.sleep(5)

            except Exception:
                logging.info(
                    f"File {url_set['flnm'][i]} could not be downloaded and was skipped. If this issue persists, check your network connection and check the NEON Data Portal for outage alerts."
                )
                pass

    return None


def download_file(url, savepath, chunk_size=1024, token=None):
    """
    This function downloads a single file from a Google Cloud Storage URL to a user-specified directory.

    Parameters
    --------
    url: str
        The Google Cloud Storage URL where the file is stored.

    savepath: str
        The file location (path) where the file will be downloaded.

    chunk_size:
        Size in bytes of chunks for chunked download

    token: str, optional
        User-specific API token generated within neon.datascience user accounts. If provided, it will be used for authentication.

    Returns
    --------
    None

    Raises
    --------
    None

    Examples
    --------
    >>> download_file('https://storage.googleapis.com/neon-aop-products/2023/FullSite/D02/2023_SCBI_6/L3/DiscreteLidar/CanopyHeightModelGtif/NEON_D02_SCBI_DP3_741000_4301000_CHM.tif', '/path/to/save', 'my-api-token')
    # This will download 'NEON_D02_SCBI_DP3_741000_4301000_CHM.tif' from the specified URL to '/path/to/save' directory using 'my-api-token' for authentication.

    Notes
    --------
    The function creates the directory specified by 'savepath' if it does not exist.
    It also downloads the readme.txt file which contains detailed information about the data package, issue logs, etc.
    https://storage.googleapis.com/neon-publication/NEON.DOM.SITE.DP3.30015.001/SCBI/20230601T000000--20230701T000000/basic/NEON.D02.SCBI.DP3.30015.001.readme.20240206T001418Z.txt

    The function issues a warning on Windows systems if the full download file path exceeds 260 characters, as the file may not be downloaded due to path length limitations.

    @author: Bridget Hass

    """

    pathparts = url.split("/")
    file_path = "/".join(pathparts[3 : len(pathparts)])

    file_fullpath = savepath + "/" + file_path
    file_fullpath_abs = os.path.abspath(file_fullpath)  # get the absolute path

    if len(file_fullpath_abs) > 260 and platform.system() == "Windows":
        warnings.warn(
            f"Filepaths on Windows are limited to 260 characters. Attempting to download a filepath that is {len(file_fullpath_abs)} characters long. Set the working or savepath directory to be closer to the root directory or enable long path support in Windows."
        )
        # return

    else:
        os.makedirs(os.path.dirname(file_fullpath), exist_ok=True)

        try:
            if token is None:
                j = 0
                while j < 3:
                    try:
                        r = requests.get(
                            url,
                            stream=True,
                            headers={"accept": "application/json", "User-Agent": usera},
                            timeout=(10, 120),
                        )
                        j = j + 5
                    except Exception:
                        logging.info(
                            f"File {os.path.basename(url)} could not be downloaded. Re-attempting."
                        )
                        j = j + 1
                        time.sleep(5)
            else:
                j = 0
                while j < 3:
                    try:
                        r = requests.get(
                            url,
                            stream=True,
                            headers={
                                "X-API-TOKEN": token,
                                "accept": "application/json",
                                "User-Agent": usera,
                            },
                            timeout=(10, 120),
                        )
                        j = j + 5
                    except Exception:
                        logging.info(
                            f"File {os.path.basename(url)} could not be downloaded. Re-attempting."
                        )
                        j = j + 1
                        time.sleep(5)

            with open(file_fullpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            r.close()

        except Exception:
            logging.info(
                f"File {os.path.basename(url)} could not be downloaded and was skipped or partially downloaded. If this issue persists, check your network connection and check the NEON Data Portal for outage alerts."
            )
            # print(e)
            pass

        return


def readme_url(readmepath):
    """
    This function accesses a readme file via a url and converts to a usable format.
    Called by the stacking functions when run in cloud mode.
    A token is not used in the API call since tokens are not used in the stacking functions.

    Parameters
    --------
    readmepath: str
        URL path to the readme file.

    Returns
    --------
    DataFrame
        A pandas data frame of the readme content.
    """

    rdres = requests.get(
        readmepath, headers={"accept": "application/json", "User-Agent": usera}
    )
    rdtxt = rdres.text
    rdlst = rdtxt.split("\n")
    rdfrm = pd.DataFrame(rdlst)

    return rdfrm
