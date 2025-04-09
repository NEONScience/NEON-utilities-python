# -*- coding: utf-8 -*-
"""
Created on 14 Mar 2024

@author: Claire Lunch

Unit tests for zips_by_product()

Mocking is not used here, tests access API resources.

"""

# import required packages
from src.neonutilities.tabular_download import zips_by_product
import pytest
import logging


def test_zips_by_product_dpid():
    """
    Test that the zips_by_product() function errors correctly for an invalid DPID
    """
    with pytest.raises(ValueError) as exc_info:
        zips_by_product(
            dpid="DP1.444.001", site="NIWO", startdate="2012-01", enddate="2022-12"
        )
    assert (
        str(exc_info.value)
        == "DP1.444.001 is not a properly formatted data product ID. The correct format is DP#.#####.00#"
    )


def test_zips_by_product_site(caplog):
    """
    Test that the zips_by_product() function returns correct message for no data at particular site
    """
    caplog.set_level(logging.INFO)

    zips_by_product(
        dpid="DP1.10003.001",
        site=["OKSR", "ARIK"],
        startdate="2012-01",
        enddate="2022-12",
    )

    assert any(
        "There are no data at the selected sites." in record.message
        for record in caplog.records
    )


def test_zips_by_product_cloud():
    """
    Test that running in cloud mode returns the correct list of files
    """
    murls = zips_by_product(
        dpid="DP1.10003.001",
        site=["NIWO", "PUUM"],
        startdate="2019-01",
        enddate="2019-12",
        release="RELEASE-2024",
        check_size=False,
        cloud_mode=True,
    )
    lst = [
        "https://storage.googleapis.com/neon-publication/NEON.DOM.SITE.DP1.10003.001/NIWO/20190701T000000--20190801T000000/basic/NEON.D13.NIWO.DP1.10003.001.brd_perpoint.2019-07.basic.20231227T192510Z.csv",
        "https://storage.googleapis.com/neon-publication/release/tag/RELEASE-2024/NEON.DOM.SITE.DP1.10003.001/NIWO/20190701T000000--20190801T000000/basic/NEON.D13.NIWO.DP1.10003.001.EML.20190703-20190713.20240127T000425Z.xml",
        "https://storage.googleapis.com/neon-publication/NEON.DOM.SITE.DP1.10003.001/NIWO/20190701T000000--20190801T000000/basic/NEON.D13.NIWO.DP1.10003.001.variables.20231227T192510Z.csv",
        "https://storage.googleapis.com/neon-publication/NEON.DOM.SITE.DP1.10003.001/NIWO/20190701T000000--20190801T000000/basic/NEON.D13.NIWO.DP0.10003.001.validation.20231227T192510Z.csv",
        "https://storage.googleapis.com/neon-publication/NEON.DOM.SITE.DP1.10003.001/NIWO/20190701T000000--20190801T000000/basic/NEON.D13.NIWO.DP1.10003.001.brd_countdata.2019-07.basic.20231227T192510Z.csv",
        "https://storage.googleapis.com/neon-publication/release/tag/RELEASE-2024/NEON.DOM.SITE.DP1.10003.001/NIWO/20190701T000000--20190801T000000/basic/NEON.D13.NIWO.DP1.10003.001.readme.20240127T000425Z.txt",
        "https://storage.googleapis.com/neon-publication/NEON.DOM.SITE.DP1.10003.001/NIWO/20190701T000000--20190801T000000/basic/NEON.D13.NIWO.DP0.10003.001.categoricalCodes.20231227T192510Z.csv",
    ]
    assert murls[0] == lst


def test_zips_by_product_avg():
    """
    Test that download by averaging interval returns the correct list of files
    """
    murls = zips_by_product(
        dpid="DP1.00005.001",
        site=["NIWO", "PUUM"],
        startdate="2022-06",
        enddate="2022-07",
        timeindex=30,
        check_size=False,
        progress=False,
        release="RELEASE-2024",
        cloud_mode=True,
    )
    assert len(murls) == 2
    ml = murls[0]
    assert len(ml) == 29
    assert (
        "https://storage.googleapis.com/neon-publication/release/tag/RELEASE-2024/NEON.DOM.SITE.DP1.00005.001/NIWO/20220701T000000--20220801T000000/basic/NEON.D13.NIWO.DP1.00005.001.readme.20240127T000425Z.txt"
        in ml
    )
    assert list(murls[1].values())[0] == "RELEASE-2024"
