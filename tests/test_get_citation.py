# -*- coding: utf-8 -*-
"""
Created on 14 Mar 2024

@author: Claire Lunch

Unit tests for get_citation()

Mocking is not used here, tests access API resources.

"""

# import required packages
import os
from datetime import date
from src.neonutilities.citation import get_citation

# read in token from os.environ
token = os.environ.get("NEON_TOKEN")

# Test a provisional and a release citation


def test_get_citation_provisional():
    """
    Test that the get_citation() function returns the expected citation for provisional data
    """
    cit = get_citation(dpid="DP1.10003.001", release="PROVISIONAL")
    dt = date.today()
    year = dt.year
    citexp = "@misc{DP1.10003.001/provisional,\n  doi = {},\n  url = {https://data.neonscience.org/data-products/DP1.10003.001},\n  author = {{National Ecological Observatory Network (NEON)}},\n  language = {en},\n  title = {Breeding landbird point counts (DP1.10003.001)},\n  publisher = {National Ecological Observatory Network (NEON)},\n  year = {" + str(year) + "}\n}"
    assert cit == citexp


def test_get_citation_release():
    """
    Test that the get_citation() function returns the expected citation for a Release
    """
    cit = get_citation(dpid="DP1.10098.001", release="RELEASE-2023")
    citexp = "@misc{https://doi.org/10.48443/73zn-k414,\n  doi = {10.48443/73ZN-K414},\n  url = {https://data.neonscience.org/data-products/DP1.10098.001/RELEASE-2023},\n  author = {{National Ecological Observatory Network (NEON)}},\n  keywords = {plant productivity, production, carbon cycle, biomass, vegetation, productivity, plants, trees, shrubs, lianas, saplings, net primary productivity (NPP), annual net primary productivity (ANPP), woody plants, vegetation structure, tree height, canopy height, vst},\n  language = {en},\n  title = {Vegetation structure (DP1.10098.001)},\n  publisher = {National Ecological Observatory Network (NEON)},\n  year = {2023}\n}\n"
    assert cit == citexp
