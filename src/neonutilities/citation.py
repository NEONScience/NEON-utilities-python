#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime
import requests


def get_citation(dpid, release):
    """
    Use the DOI Foundation API to get BibTex-formatted citations for NEON data,
    or use a template to generate a BibTex citation for provisional data.
    Helper function to download and stacking functions.

    Parameters
    ----------
    dpid: str
        The data product ID of the data to be cited

    release: str
        The data release to be cited. Can be provisional.

    Return
    ------
    A character string containing the citation in BibTex format.

    Example
    -------
    Get the citation for Breeding landbird point counts (DP1.10003.001),
    RELEASE-2023

    >>> cit = get_citation(dpid="DP1.10003.001", release="RELEASE-2023")

    Created on Fri Aug 25 10:26:23 2023

    @author: Claire Lunch
    """

    if release == "PROVISIONAL":
        # construct citation from template
        citI = "@misc{DPID/provisional,\n  doi = {},\n  url = {https://data.neonscience.org/data-products/DPID},\n  author = {{National Ecological Observatory Network (NEON)}},\n  language = {en},\n  title = {NAME (DPID)},\n  publisher = {National Ecological Observatory Network (NEON)},\n  year = {YEAR}\n}"
        citDP = citI.replace("DPID", dpid)
        citY = citDP.replace("YEAR", str(datetime.now().year))

        nm_req = requests.get("https://data.neonscience.org/api/v0/products/" + dpid)
        nm_str = nm_req.json()
        nm = nm_str["data"]["productName"]

        cit = citY.replace("NAME", nm)
        return cit

    else:
        # get DOI from NEON API, then citation from DOI API
        pr_req = requests.get("https://data.neonscience.org/api/v0/products/" + dpid)
        pr_str = pr_req.json()
        rels = pr_str["data"]["releases"]
        relinfo = next((i for i in rels if i["release"] == release), None)

        if relinfo is None:
            print("There are no data with dpid=" + dpid + " and release=" + release)
            return relinfo

        else:
            doi = relinfo["productDoi"]["url"]
            doi_req = requests.get(doi, headers={"accept": "application/x-bibtex"})
            return doi_req.text
