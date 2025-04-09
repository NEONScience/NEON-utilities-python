# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:32:58 2024

@author: bhass

Get the issue log for a specific data product.

Adapted from R neonUtilities getIssueLog.R
https://github.com/NEONScience/NEON-utilities/blob/main/neonUtilities/R/getIssueLog.R
written by:
@author: Claire Lunch (clunch@battelleecology.org)

"""

import re
import logging
import pandas as pd
from .helper_mods.api_helpers import get_api

logging.basicConfig(level=logging.INFO, format="%(message)s")

# %% functions to validate inputs (should pull these out into another helper module??)


def validate_dpid(dpid):
    """
    Validates the format of a NEON data product ID (dpid).

    Args:
        dpid (str): The NEON data product ID to validate.

    Raises:
        ValueError: If the DPID is not in the correct format.
    """
    dpid_pattern = "DP[1-4]{1}.[0-9]{5}.00[1-2]{1}"
    if not re.fullmatch(dpid_pattern, dpid):
        raise ValueError(
            f"{dpid} is not a properly formatted data product ID. The correct format is DP#.#####.00#"
        )


# %% functions to get the change / issue logs


def get_change_log_df(dpid, token=None):
    """
    Retrieves the change log for a NEON data product.

    Args:
        dpid (str): The NEON data product ID.
        token (str, optional): The NEON API token. Defaults to None.

    Returns:
        change_log_df: A DataFrame containing the changeLogs for the provided dpid.
        columns of the dataframe are: 'id', 'parentIssueID', 'issueDate',
        'resolvedDate', 'dateRangeStart', 'dateRangeEnd', 'locationAffected',
        'issue', 'resolution'
    """
    req = get_api(
        api_url=f"https://data.neonscience.org/api/v0/products/{dpid}", token=token
    )
    if req is None:
        logging.info(f"Error in metadata retrieval for {dpid}. Issue log not found.")
        return None
    all_product_info = pd.json_normalize(req.json()["data"])
    change_log_df = pd.DataFrame(all_product_info["changeLogs"][0])

    return change_log_df


def get_eddy_issue_log(dpid, token=None):
    """
    Retrieves the issue log for bundled eddy covariance data products.

    Args:
        dpid (str): The NEON data product ID.
        token (str, optional): The NEON API token. Defaults to None.

    Returns:
        eddy_issue_log_df: A DataFrame containing the bundled eddy covariance issue logs, including the data product id of the sub-data products.
        columns of the dataframe are: 'dpid', 'id', 'parentIssueID', 'issueDate', 'resolvedDate', 'dateRangeStart',
               'dateRangeEnd', 'locationAffected', 'issue', 'resolution'
    """

    bundle_dps = [
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
        "DP4.00200.001",
    ]

    eddy_issue_log_list = []

    for dpid in bundle_dps:
        change_log_df = get_change_log_df(dpid, token=token)
        if change_log_df is not None and not change_log_df.empty:
            change_log_df["dpid"] = dpid
            eddy_issue_log_list.append(change_log_df)

    eddy_issue_log_df = pd.concat(eddy_issue_log_list, ignore_index=True)

    return eddy_issue_log_df


def get_issue_log(dpid, token=None):
    """
    Retrieves the issue log for any NEON data products. Bundled eddy covariance data products have an additional column of the sub-data product id.

    Args:
        dpid: str
            The NEON data product ID.

        token: str
            User-specific API token from data.neonscience.org user account. See
            https://data.neonscience.org/data-api/rate-limiting/ for details about
            API rate limits and user tokens. If omitted, download uses the public rate limit.

    Returns:
        issue_log_df: A pandas DataFrame containing the changeLogs for the provided dpid.
        columns of the bundled eddy data frame are: 'dpid', 'id',
        'parentIssueID', 'issueDate', 'resolvedDate', 'dateRangeStart',
        'dateRangeEnd', 'locationAffected', 'issue', 'resolution';
        all other data products have the same columns minus 'dpid'

    Example
    -------
    Get the issue log for Breeding landbird point counts (DP1.10003.001)

    >>> birdiss = get_issue_log(dpid="DP1.10003.001")

    """
    # raise value error and print message if dpid isn't formatted as expected
    validate_dpid(dpid)

    if dpid == "DP4.00200.001":
        issue_log_df = get_eddy_issue_log(dpid, token)
    else:
        issue_log_df = get_change_log_df(dpid, token)

    return issue_log_df
