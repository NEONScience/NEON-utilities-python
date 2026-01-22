#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import importlib_resources
import pandas as pd
import logging
from .helper_mods.api_helpers import get_api
from .helper_mods.api_helpers import token_check
from .helper_mods.api_helpers import get_zip_urls
from .helper_mods.api_helpers import get_tab_urls
from .helper_mods.api_helpers import download_urls
from .helper_mods.metadata_helpers import convert_byte_size
from . import __resources__

logging.basicConfig(level=logging.INFO, format="%(message)s")


def query_files(
    lst,
    dpid,
    site="all",
    startdate=None,
    enddate=None,
    package="basic",
    release="current",
    timeindex="all",
    tabl="all",
    include_provisional=False,
    token=None,
):
    """
    Use the query endpoint to get a list of data package files from NEON.

    Parameters
    --------
    lst: Availability info from a call to the products endpoint of the NEON API.
    dpid: Data product identifier in the form DP#.#####.###
    site: Either the string 'all', or one or more 4-letter NEON site codes. Defaults to 'all'.
    startdate: Earliest date of data to download, in the form YYYY-MM
    enddate: Latest date of data to download, in the form YYYY-MM
    package: Download package to access, either basic or expanded
    release: Data release to download. Defaults to the most recent release.
    timeindex: Either the string 'all', or the time index of data to download, in minutes. Only applicable to sensor (IS) data. Defaults to 'all'.
    tabl: Either the string 'all', or the name of a single data table to download. Defaults to 'all'.
    include_provisional: Should Provisional data be returned in the download? Defaults to False.
    token: User specific API token (generated within neon.datascience user accounts). If omitted, download uses the public rate limit.

    Return
    --------
    A list of paths to NEON data package files.

    Created on 10 Jul 2024

    @author: Claire Lunch
    """

    adict = lst["data"]["siteCodes"]
    releasedict = {}
    
    # check for expired token
    if token is not None:
        token = token_check(token)

    # check expanded package status
    if package == "expanded":
        if not lst["data"]["productHasExpanded"]:
            logging.info(
                "No expanded package found for NEON "
                + dpid
                + ". Basic package downloaded instead."
            )
            package = "basic"

    # if sites are not specified, get list of sites with data
    if site == ["all"]:
        siteset = []
        for i in range(0, len(adict)):
            siteset.append(adict[i].get("siteCode"))
    else:
        if isinstance(site, list):
            siteset = site
        else:
            siteset = [site]

    # set up site query
    sitesurllist = ["&siteCode=" + s for s in siteset]
    sitesurl = "".join(sitesurllist)

    # if dates are not specified, get data date range
    if startdate is None or enddate is None:
        dateset = []
        for i in range(0, len(adict)):
            dateset.append(adict[i].get("availableMonths"))
        dateset = sum(dateset, [])
        if startdate is None:
            startdate = min(dateset)
        if enddate is None:
            enddate = max(dateset)

    # set up date query
    dateurl = "&startDateMonth=" + startdate + "&endDateMonth=" + enddate

    # string for true/false include provisional
    if include_provisional is True:
        ipurl = "&includeProvisional=true"
    else:
        ipurl = "&includeProvisional=false"

    if release == "current":
        relurl = ""
    else:
        relurl = "&release=" + release

    # construct full query url and run query
    qurl = (
        "https://data.neonscience.org/api/v0/data/query?productCode="
        + dpid
        + sitesurl
        + dateurl
        + ipurl
        + "&package="
        + package
        + relurl
    )
    qreq = get_api(api_url=qurl, token=token)
    if qreq is None:
        logging.info("No NEON Portal API response for selected query. Check inputs.")
        return None
    qdict = qreq.json()

    # get file list from dictionary response
    reldict = qdict.get("data")
    pdict = reldict.get("releases")
    flurl = []
    for i in range(0, len(pdict)):
        packdict = pdict[i].get("packages")
        rdict = pdict[i].get("release")
        for j in range(0, len(packdict)):
            fdict = packdict[j].get("files")
            for k in range(0, len(fdict)):
                flurl.append(fdict[k].get("url"))
                releasedict[
                    re.sub(
                        pattern="https://storage.googleapis.com/",
                        repl="",
                        string=fdict[k].get("url"),
                    )
                ] = rdict

    # if timeindex or tabl are set, subset the list of files
    if timeindex == "all" and tabl == "all":
        return [flurl, releasedict]
    else:
        if timeindex != "all" and tabl != "all":
            raise ValueError(
                "Only one of timeindex or tabl can be specified, not both."
            )
        else:
            if timeindex != "all":
                tt = re.compile(
                    str(timeindex)
                    + "min|"
                    + str(timeindex)
                    + "_min|science_review_flags|variables|readme|sensor_positions|categoricalCodes"
                )

            if tabl != "all":
                tt = re.compile(
                    "[.]"
                    + tabl
                    + "[.]|variables|readme|sensor_positions|categoricalCodes"
                )

            flurlsub = [f for f in flurl if tt.search(f)]
            releasedictsub = {r: v for r, v in releasedict.items() if tt.search(r)}
            return [flurlsub, releasedictsub]


def zips_by_product(
    dpid,
    site="all",
    startdate=None,
    enddate=None,
    package="basic",
    release="current",
    timeindex="all",
    tabl="all",
    check_size=True,
    include_provisional=False,
    cloud_mode=False,
    progress=True,
    token=None,
    savepath=None,
):
    """
    This function queries the NEON API for data by data product, site(s), and
    month(s), and downloads the corresponding data packages. Use this function
    to download NEON observational (OS) and instrument (IS) data; for remote
    sensing data, use the by_file_aop() and by_tile_aop() functions.

    Parameters
    ------------------
    dpid: str
        Data product identifier in the form DP#.#####.###

    site: str
        Either the string 'all', or one or more 4-letter NEON site codes. Defaults to 'all'.

    startdate: str, optional
        Earliest date of data to download, in the form YYYY-MM

    enddate: str, optional
        Latest date of data to download, in the form YYYY-MM

    package: str, optional
        Download package to access, either basic or expanded. Defaults to 'basic'.

    release: str, optional
        Data release to download. Defaults to the most recent release.

    timeindex: str, optional
        Either 'all', or the time index of data to download, in minutes. Only applicable to sensor (IS) data. Defaults to 'all'.

    tabl: str, optional
        Either the string 'all', or the name of a single data table to download. Only applicable to observational (OS) data. Defaults to 'all'.

    check_size: bool, optional
        True or False, should the user approve the total file size before downloading? Defaults to True. When working in batch mode, or other non-interactive workflow, use check_size=False.

    include_provisional: bool, optional
        Should Provisional data be returned in the download? Defaults to False. See
        https://www.neonscience.org/data-samples/data-management/data-revisions-releases
        for details on the difference between provisional and released data.

    cloud_mode: bool, optional
        Use cloud mode to transfer files cloud-to-cloud? If used, zips_by_product() returns a
        list of files rather than downloading them. Defaults to False; in general this
        option should be used via load_by_product(), in which zips_by_product() is a
        helper function.

    progress: bool, optional
        Should the function display progress bars as it runs? Defaults to True.

    token: str, optional
        User-specific API token from data.neonscience.org user account. See
        https://data.neonscience.org/data-api/rate-limiting/ for details about
        API rate limits and user tokens. If omitted, download uses the public rate limit.

    savepath: str, optional
        File path of location to save data.

    Return
    -----------------
    A folder on the local drive containing the set of data package files meeting the input criteria or,
    in cloud mode, a list of file paths meeting the input criteria.

    Example
    -----------------
    Download water quality data from COMO (Como Creek) in 2018

    >>> zips_by_product(dpid="DP1.20288.001",site="COMO",
                        startdate="2018-01", enddate="2018-12",
                        token=None, savepath="/mypath/Downloads")

    Created on Wed Jan 31 14:36:22 2024

    @author: Claire Lunch
    """

    # error message if dpid is not formatted correctly
    if not re.search(pattern="DP[1-4]{1}.[0-9]{5}.00[0-9]{1}", string=dpid):
        raise ValueError(
            f"{dpid} is not a properly formatted NEON data product ID. The correct format is DP#.#####.00#"
        )

    # error message if package is not basic or expanded
    if not package in ["basic", "expanded"]:
        raise ValueError(
            f"{package} is not a valid NEON download package name. Package must be basic or expanded"
        )

    # error messages for products that can't be downloaded by zips_by_product()
    # AOP products
    if dpid[4:5:1] == 3 and dpid != "DP1.30012.001":
        raise ValueError(
            f"{dpid} is a NEON remote sensing data product. Use the by_file_aop() or by_tile_aop() function."
        )

    # Phenocam products
    if dpid == "DP1.00033.001" or dpid == "DP1.00042.001":
        raise ValueError(
            f"{dpid} is a NEON phenological image product, data are hosted by Phenocam."
        )

    # Aeronet product
    if dpid == "DP1.00043.001":
        raise ValueError(
            f"NEON Spectral sun photometer ({dpid}) data are hosted by Aeronet."
        )

    # DHP expanded package
    if dpid == "DP1.10017.001" and package == "expanded":
        raise ValueError(
            "NEON Digital hemispherical images expanded file packages exceed programmatic download limits. Either download from the data portal, or download the basic package and use the URLs in the data to download the images themselves. Follow instructions in the Data Product User Guide for image file naming."
        )

    # individual SAE products
    if dpid in [
        "DP1.00007.001",
        "DP1.00010.001",
        "DP1.00034.001",
        "DP1.00035.001",
        "DP1.00036.001",
        "DP1.00037.001",
        "DP1.00099.001",
        "DP1.00100.001",
        "DP2.00008.001",
        "DP2.00009.001",
        "DP2.00024.001",
        "DP3.00008.001",
        "DP3.00009.001",
        "DP3.00010.001",
        "DP4.00002.001",
        "DP4.00007.001",
        "DP4.00067.001",
        "DP4.00137.001",
        "DP4.00201.001",
        "DP1.00030.001",
    ]:
        raise ValueError(
            f"{dpid} is only available in the bundled NEON eddy covariance data product. Download DP4.00200.001 to access these data."
        )

    # check for incompatible values of release= and include_provisional=
    if release == "PROVISIONAL" and not include_provisional:
        raise ValueError(
            "NEON Download request is for release=PROVISIONAL. To download PROVISIONAL data, enter input parameter include_provisional=True."
        )
    if re.search(pattern="RELEASE", string=release) is not None and include_provisional:
        logging.info(
            f"NEON Download request is for release={release} but include_provisional=True. Only data in {release} will be downloaded."
        )

    # error message if dates aren't formatted correctly
    # separate logic for each, to easily allow only one to be NA
    if startdate is not None:
        if re.search(pattern="[0-9]{4}-[0-9]{2}", string=startdate) is None:
            raise ValueError(
                "startdate and enddate must be either None or valid dates in the form YYYY-MM"
            )

    if enddate is not None:
        if re.search(pattern="[0-9]{4}-[0-9]{2}", string=enddate) is None:
            raise ValueError(
                "startdate and enddate must be either None or valid dates in the form YYYY-MM"
            )

    # can only specify timeindex xor tabl
    if timeindex != "all" and tabl != "all":
        raise ValueError("Only one of timeindex or tabl can be specified, not both.")
    # consider adding warning message about using tabl=

    # allow for single sites
    if not isinstance(site, list):
        site = [site]

    # redirect for aqu met products and bundles
    shared_aquatic_file = (
        importlib_resources.files(__resources__) / "shared_aquatic.csv"
    )
    shared_aquatic_df = pd.read_csv(shared_aquatic_file, index_col="site")

    if site != ["all"]:
        siter = []
        indx = 0
        for s in site:
            if s in shared_aquatic_df.index:
                if s in ["BLWA"] and release not in ["RELEASE-2021","RELEASE-2022","RELEASE-2023","RELEASE-2024","RELEASE-2025"]:
                    indx = indx + 1
                    if "DELA" not in site:
                        siter.append(["DELA"])
                    if indx == 1:
                        logging.info(
                            f"Some NEON sites in your download request are aquatic sites where {dpid} is collected at a nearby terrestrial site. The sites you requested, and the sites that will be accessed instead, are listed below."
                            )
                    logging.info("Until the fall of 2025, meteorological data for BLWA were collected at DELA. Data collection at DELA ended in late 2025 and the meteorological station was relocated to BLWA. If your download request crosses this time period, data will be downloaded from each site for the time period when they are available.")
                else:
                    ss = shared_aquatic_df.loc[s]
                    if dpid in list(ss["product"]):
                        indx = indx + 1
                        sx = list(ss["towerSite"][ss["product"] == dpid])
                        siter.append(sx)
                        if indx == 1:
                            logging.info(
                                f"Some NEON sites in your download request are aquatic sites where {dpid} is collected at a nearby terrestrial site. The sites you requested, and the sites that will be accessed instead, are listed below."
                                )
                        logging.info(f"{s} -> {''.join(sx)}")
                    else:
                        siter.append([s])
            else:
                siter.append([s])
        siter = sum(siter, [])
    else:
        siter = site

    # redirect for chemistry bundles
    chem_bundles_file = importlib_resources.files(__resources__) / "chem_bundles.csv"
    chem_bundles_df = pd.read_csv(chem_bundles_file)
    if dpid in list(chem_bundles_df["product"]):
        newDPID = list(
            chem_bundles_df["homeProduct"][chem_bundles_df["product"] == dpid]
        )
        if newDPID == ["depends"]:
            raise ValueError(
                "NEON Root chemistry and isotopes have been bundled with the root biomass data. For root chemistry from Megapits, download DP1.10066.001. For root chemistry from periodic sampling, download DP1.10067.001."
            )
        else:
            raise ValueError(
                f"{''.join(dpid)} has been bundled with {''.join(newDPID)} and is not available independently. Please download {''.join(newDPID)}."
            )

    # redirect for veg structure and sediment data product bundles
    other_bundles_file = importlib_resources.files(__resources__) / "other_bundles.csv"
    other_bundles_df = pd.read_csv(other_bundles_file)
    if dpid in list(other_bundles_df["product"]):
        bundle_release = other_bundles_df["lastRelease"][
            other_bundles_df["product"] == dpid
        ].values[0]
        if release > bundle_release:
            newDPID = list(
                other_bundles_df["homeProduct"][other_bundles_df["product"] == dpid]
            )
            raise ValueError(
                f"In all NEON releases after {bundle_release}, {''.join(dpid)} has been bundled with {''.join(newDPID)} and is not available independently. Please download {''.join(newDPID)}."
            )

    # check for expired token
    if token is not None:
        token = token_check(token)

    # end of error and exception handling, start the work
    # query the /products endpoint for the product requested
    if release == "current" or release == "PROVISIONAL":
        prodreq = get_api(
            api_url="https://data.neonscience.org/api/v0/products/" + dpid, token=token
        )
    else:
        prodreq = get_api(
            api_url="https://data.neonscience.org/api/v0/products/"
            + dpid
            + "?release="
            + release,
            token=token,
        )

    if prodreq is None:
        if release == "LATEST":
            logging.info(
                f"No data found for NEON data product {dpid}. LATEST data requested; check that token is valid for LATEST access."
            )
            return
        else:
            if release != "current" and release != "PROVISIONAL":
                rels = get_api(
                    api_url="https://data.neonscience.org/api/v0/releases/", token=token
                )
                if rels is None:
                    raise ConnectionError(
                        f"NEON Data product {dpid} was not found or API was unreachable."
                    )
                relj = rels.json()
                reld = relj["data"]
                rellist = []
                for i in range(0, len(reld)):
                    rellist.append(reld[i]["release"])
                if release not in rellist:
                    raise ValueError(f"Release not found. Valid NEON data releases are {rellist}")
                else:
                    raise ConnectionError(
                        f"NEON Data product {dpid} was not found or API was unreachable."
                    )
            else:
                raise ConnectionError(
                    f"NEON Data product {dpid} was not found or API was unreachable."
                )

    avail = prodreq.json()

    # error message if product or data not found
    # I think this would never be called due to the way get_api() is set up
    try:
        avail["error"]["status"]
        logging.info(f"No NEON data found for data product {dpid}")
        return
    except Exception:
        pass

    # check that token was used
    if "x-ratelimit-limit" in prodreq.headers and token is not None:
        if prodreq.headers.get("x-ratelimit-limit") == 200:
            logging.info("NEON Portal API token was not recognized. Public rate limit applied.")

    # use query endpoint if cloud mode selected
    if cloud_mode:
        fls = query_files(
            lst=avail,
            dpid=dpid,
            site=siter,
            startdate=startdate,
            enddate=enddate,
            package=package,
            release=release,
            timeindex=timeindex,
            tabl=tabl,
            include_provisional=include_provisional,
            token=token,
        )
        return fls

    else:
        # get data urls
        month_urls = []
        for i in range(0, len(avail["data"]["siteCodes"])):
            month_urls.append(avail["data"]["siteCodes"][i]["availableDataUrls"])

        # check for no results
        if len(month_urls) == 0:
            logging.info(f"There are no NEON {dpid} data matching the search criteria.")
            return

        # un-nest list
        month_urls = sum(month_urls, [])

        # subset by site
        if siter != ["all"]:
            site_urls = []
            for si in siter:
                se = re.compile(si)
                month_sub = [s for s in month_urls if se.search(s)]
                site_urls.append(month_sub)
            site_urls = sum(site_urls, [])
        else:
            site_urls = month_urls

        # check for no results
        if len(site_urls) == 0:
            logging.info(f"There are no NEON {dpid} data at the selected sites.")
            return

        # subset by start date
        if startdate is not None:
            ste = re.compile("20[0-9]{2}-[0-9]{2}")
            start_urls = [
                st for st in site_urls if ste.search(st).group(0) >= startdate
            ]
        else:
            start_urls = site_urls

        # check for no results
        if len(start_urls) == 0:
            logging.info(f"There are no NEON {dpid} data at the selected date(s).")
            return

        # subset by end date
        if enddate is not None:
            ete = re.compile("20[0-9]{2}-[0-9]{2}")
            end_urls = [et for et in start_urls if ete.search(et).group(0) <= enddate]
        else:
            end_urls = start_urls

        # check for no results
        if len(end_urls) == 0:
            logging.info(f"There are no NEON {dpid} data at the selected date(s).")
            return

        # if downloading entire site-months, pass to get_zip_urls to query each month for url
        if timeindex == "all" and tabl == "all":
            durls = get_zip_urls(
                url_set=end_urls,
                package=package,
                release=release,
                include_provisional=include_provisional,
                token=token,
                progress=progress,
            )
        else:
            # if downloading by table or averaging interval, pass to get_tab_urls
            durls = get_tab_urls(
                url_set=end_urls,
                package=package,
                release=release,
                include_provisional=include_provisional,
                timeindex=timeindex,
                tabl=tabl,
                token=token,
                progress=progress,
            )

        # check download size
        download_size = convert_byte_size(sum(durls["sz"]))
        if check_size:
            if (
                input(
                    f"Continuing will download {len(durls['z'])} NEON {dpid} files totaling approximately {download_size}. Do you want to proceed? (y/n) "
                )
                != "y"
            ):
                logging.info("Download halted.")
                return None
        else:
            logging.info(
                f"Downloading {len(durls['z'])} NEON {dpid} files totaling approximately {download_size}."
            )

        # set up folder to save to
        if savepath is None:
            savepath = os.getcwd()
        outpath = savepath + "/filesToStack" + dpid[4:9] + "/"

        if not os.path.exists(outpath):
            os.makedirs(outpath)
        else:
            logging.info(
                "Warning: Download folder already exists. Check carefully for duplicate files."
            )

        if timeindex != "all" or tabl != "all":
            for f in durls["flpth"]:
                if not os.path.exists(outpath + f):
                    os.makedirs(outpath + f)

        # download data from each url
        download_urls(url_set=durls, outpath=outpath, token=token, progress=progress)

        return None
