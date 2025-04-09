# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:00:19 2024

@author: bhass

Installation dependencies:
    conda install pytest
    conda install parameterized

Unit tests for by_file_aop and by_tile_aop.

These mainly test the functions' output messages for invalid inputs,
collocated sites, data not found, and provisional availability scenarios

More complete integration tests are included in the NEONScience/nu_python_testing repository, which is currently private.

Notes:
- These tests require an internet connection to run, as they are actually making API calls
- Paramaterization with the @parameterized.expand decorator allows for testing different sets of inputs with the same test
    This requires the parameterized package (pip install parameterized)
- The @patch('builtins.input', return_value='n') decorator mocks the user input to say "n" (no) to avoid downloading data


"""

# import required packages
import os
import pytest
import unittest
from unittest.mock import patch
from parameterized import parameterized

# import neon utilities functions that are being tested
from src.neonutilities.aop_download import (
    by_file_aop,
    by_tile_aop,
)

# read in token from os.environ (requires the token to be set)
token = os.environ.get("NEON_TOKEN")

# Test invalid inputs and make sure informational messages display correctly
# This contains similar tests for by_file_aop and by_tile_aop,
# with some additional tests for extra functionality in by_tile_aop


class TestByFileAOP(unittest.TestCase):
    def setUp(self):
        """Set up before each test"""
        # these are valid inputs, invalid inputs for testing logs / messages are supplied within each test
        self.dpid = "DP3.30015.001"
        self.site = "MCRA"
        self.year = 2021

    def test_invalid_dpid_format(self):
        """
        Test that invalid dpid format raises ValueError and message displays correctly.
        """
        invalid_dpid = "DP1.30001"
        with self.assertRaises(
            ValueError,
            msg=f"{invalid_dpid} is not a properly formatted data product ID. The correct format is DP#.#####.00#",
        ):
            by_file_aop(dpid=invalid_dpid, site=self.site, year=self.year)

    def test_invalid_aop_dpid_pattern(self):
        """
        Test that invalid AOP dpid pattern raises a ValueError and message displays correctly.
        """
        invalid_aop_dpid = "DP1.20001.001"
        with self.assertRaises(
            ValueError,
            msg=f"{invalid_aop_dpid} is not a valid AOP data product ID. AOP products follow the format DP#.300##.00#",
        ):
            by_file_aop(dpid=invalid_aop_dpid, site=self.site, year=self.year)

    def test_invalid_aop_dpid_suspended(self):
        """
        Test that entering a suspended AOP data product raises a ValueError and message displays correctly.
        """
        suspended_aop_dpid = "DP2.30016.001"
        # ' Valid AOP DPIDs are '):
        with self.assertRaises(
            ValueError,
            msg=f"{suspended_aop_dpid} has been suspended and is not currently available, see https://www.neonscience.org/data-products/{suspended_aop_dpid} for more details.",
        ):
            by_file_aop(dpid=suspended_aop_dpid, site=self.site, year=self.year)

    def test_check_field_spectra_dpid(self):
        """
        Test that providing field spectra dpid raises ValueError and message displays correctly.
        """
        field_spectra_dpid = "DP1.30012.001"
        with self.assertRaises(
            ValueError,
            msg=f"{field_spectra_dpid} is the Field spectral data product, which is published as tabular data. Use zipsByProduct() or loadByProduct() to download these data.",
        ):
            by_file_aop(dpid=field_spectra_dpid, site=self.site, year=self.year)

    def test_invalid_site_format(self):
        """
        Test that invalid site format raises ValueError and message displays correctly.
        """
        invalid_site = "McRae"
        with self.assertRaises(
            ValueError,
            msg=f"{invalid_site} is an invalid site format. A four-letter NEON site code is required. NEON sites codes can be found here: https://www.neonscience.org/field-sites/explore-field-sites",
        ):
            by_file_aop(dpid=self.dpid, site=invalid_site, year=self.year)

    def test_invalid_neon_site(self):
        """
        Test that an invalid NEON site code raises a ValueError and the message displays correctly.
        """
        invalid_site = "HOPD"
        with self.assertRaises(
            ValueError,
            msg=f"{invalid_site} is not a valid NEON site code. A complete list of NEON site codes can be found here: https://www.neonscience.org/field-sites/explore-field-sites",
        ):
            by_file_aop(dpid=self.dpid, site=invalid_site, year=self.year)

    @parameterized.expand(
        [
            ("21",),
            ("2021-05",),
        ]
    )
    def test_invalid_year_format(self, year):
        """
        Test that invalid year format raises ValueError and message displays correctly.
        """
        with pytest.raises(
            ValueError,
            match=f'{year} is an invalid year. Year is required in the format "2017" or 2017, eg. AOP data are available from 2013 to present.',
        ):
            by_file_aop(dpid=self.dpid, site=self.site, year=year)

    @parameterized.expand(
        [
            ("2022", "CHEQ", "STEI"),
            ("2022", "TREE", "STEI"),
            ("2021", "DCFS", "WOOD"),
            ("2020", "KONA", "KONZ"),
        ],
        name_func=lambda f, n, p: f"{f.__name__}_{p.args[1]}_{p.args[0]}",
    )
    # the name_func displays a more descriptive test name when running the tests
    @patch("builtins.input", return_value="n")
    def test_collocated_terrestrial_site_message(
        self, year, site, flightSite, input_mock
    ):
        """
        Test application of the terrestrial collocated site lookup, and expected message display.
        """
        with self.assertLogs(level="INFO") as cm:
            by_file_aop(dpid="DP3.30015.001", site=site, year=year, token=token)
            self.assertIn(
                f"INFO:root:{site} is part of the flight box for {flightSite}. Downloading data from {flightSite}.",
                cm.output,
            )

    @parameterized.expand(
        [
            ("2018", "BARC", "OSBS"),
            ("2020", "COMO", "NIWO"),
            # ("2021", "BLDE", "YELL"),
            # ("2020", "KING", "KONZ"),
        ],
        name_func=lambda f, n, p: f"{f.__name__}_{p.args[1]}_{p.args[0]}",
    )
    @patch("builtins.input", return_value="n")
    def test_collocated_aquatic_site_message(self, year, site, flightSite, input_mock):
        """
        Test application of the aquatic collocated site lookup, and expected message display.
        """
        with self.assertLogs(level="INFO") as cm:
            by_file_aop(dpid=self.dpid, site=site, year=year, token=token)
            self.assertIn(
                f"INFO:root:{site} is an aquatic site and is sometimes included in the flight box for {flightSite}. Aquatic sites are not always included in the flight coverage every year.\nDownloading data from {flightSite}. Check data to confirm coverage of {site}.",
                cm.output,
            )

    def test_no_data_available_message(self):
        """
        Test that the by_file_aop() function returns the expected error log when no data is available for a selected site and year.
        """
        with self.assertLogs(level="INFO") as cm:
            by_file_aop(dpid="DP3.30015.001", site=self.site, year=2020)
            self.assertIn(
                f"INFO:root:There are no {self.dpid} data available at the site {self.site} in 2020.\nTo display available dates for a given data product and site, use the function list_available_dates().",
                cm.output,
            )

    @patch("builtins.input", return_value="n")
    def test_check_download_size_message(self, input_mock):
        """
        Test that download check_size message displays correctly.
        """
        result = by_file_aop(dpid=self.dpid, site=self.site, year=self.year)
        # Check that the function asked for confirmation to download and prints expected message.
        input_mock.assert_called_once_with(
            "Continuing will download 128 files totaling approximately 93.1 MB. Do you want to proceed? (y/n) "
        )
        # Check that the function halted the download
        self.assertEqual(result, None)

    # Provisional scenarios

    # provisional not included, and no data available
    def test_all_provisional_no_data_available_message(self):
        """
        Test that the by_file_aop() function returns the expected message when include_provisional is set to False (default) but no data are available.
        This has already run through the check that any data is available(eg. there is data at that site for the year provided)
        """
        with self.assertLogs(level="INFO") as cm:
            by_file_aop(dpid="DP3.30015.001", site="WLOU", year=2024)
            self.assertIn(
                "INFO:root:No data files found. Available data may all be provisional. To download provisional data, use input parameter include_provisional=True.",
                cm.output,
            )

    # provisional included, and no data available
    @patch("builtins.input", return_value="n")
    def test_provisional_included_and_data_available_message(self, input_mock):
        """
        Test that the by_file_aop() function returns the expected message when include_provisional is set to False (default) but no data are available.
        This has already run through the check that any data is available(eg. there is data at that site for the year provided)
        """
        with self.assertLogs(level="INFO") as cm:
            by_file_aop(
                dpid="DP3.30015.001", site="WLOU", year=2024, include_provisional=True
            )
            self.assertIn(
                "INFO:root:Provisional data are included. To exclude provisional data, use input parameter include_provisional=False.",
                cm.output,
            )

    # other scenarios- check messages but don't download the data ?
    # provisional not included, and data available
    # provisional included, and data available


class TestByTileAop(unittest.TestCase):
    def setUp(self):
        """Set up before each test"""
        # these are valid inputs, invalid inputs for each tests are supplied within each test
        self.dpid = "DP3.30015.001"
        self.easting = 566000
        self.northing = 4900000
        self.site = "MCRA"
        self.year = 2021

    def test_invalid_dpid_format(self):
        invalid_dpid = "DP1.30001"
        with self.assertRaises(
            ValueError,
            msg=f"{invalid_dpid} is not a properly formatted data product ID. The correct format is DP#.#####.00#",
        ):
            by_tile_aop(
                dpid=invalid_dpid,
                site=self.site,
                year=self.year,
                easting=self.easting,
                northing=self.northing,
            )

    def test_invalid_aop_l3_dpid(self):
        """
        Test that invalid AOP dpid raises a ValueError and message displays correctly.
        """
        invalid_aop_dpid = "DP1.30001.001"
        with self.assertRaises(
            ValueError,
            msg=f"{invalid_aop_dpid} is not a valid Level 3 AOP data product ID. Level 3 AOP products follow the format DP3.300##.00#",
        ):
            by_tile_aop(
                dpid=invalid_aop_dpid,
                site=self.site,
                year=self.year,
                easting=self.easting,
                northing=self.northing,
            )

    def test_check_field_spectra_dpid(self):
        field_spectra_dpid = "DP1.30012.001"
        with self.assertRaises(
            ValueError,
            msg=f"{field_spectra_dpid} is the Field spectral data product, which is published as tabular data. Use zipsByProduct() or loadByProduct() to download these data.",
        ):
            by_tile_aop(
                dpid=field_spectra_dpid,
                site=self.site,
                year=self.year,
                easting=self.easting,
                northing=self.northing,
            )

    def test_invalid_site_format(self):
        invalid_site = "McRae"
        with self.assertRaises(
            ValueError,
            msg=f"{invalid_site} is an invalid site format. A four-letter NEON site code is required. NEON sites codes can be found here: https://www.neonscience.org/field-sites/explore-field-sites",
        ):
            by_tile_aop(
                dpid=self.dpid,
                site=invalid_site,
                year=self.year,
                easting=self.easting,
                northing=self.northing,
            )

    def test_invalid_neon_site(self):
        invalid_site = "ABBA"
        with self.assertRaises(
            ValueError,
            msg=f"{invalid_site} is not a valid NEON site code. A complete list of NEON site codes can be found here: https://www.neonscience.org/field-sites/explore-field-sites",
        ):
            by_tile_aop(
                dpid=self.dpid,
                site=invalid_site,
                year=self.year,
                easting=self.easting,
                northing=self.northing,
            )

    @parameterized.expand(
        [
            ("21",),
            ("2021-05",),
        ]
    )
    def test_invalid_year_format(self, year):
        """
        Test that invalid year format raises ValueError and message displays correctly.
        """
        with pytest.raises(
            ValueError,
            match=f'{year} is an invalid year. Year is required in the format "2017" or 2017, eg. AOP data are available from 2013 to present.',
        ):
            by_tile_aop(
                dpid=self.dpid,
                site=self.site,
                year=year,
                easting=self.easting,
                northing=self.northing,
            )

    @parameterized.expand(
        [
            ("2022", "CHEQ", "STEI"),
            ("2022", "TREE", "STEI"),
            ("2021", "DCFS", "WOOD"),
            ("2020", "KONA", "KONZ"),
        ],
        name_func=lambda f, n, p: f"{f.__name__}_{p.args[1]}_{p.args[0]}",
    )
    # the name_func displays a more descriptive test name when running the tests
    @patch("builtins.input", return_value="n")
    def test_collocated_terrestrial_site_message(
        self, year, site, flightSite, input_mock
    ):
        """
        Test application of the terrestrial collocated site lookup, and expected message display.
        """
        with self.assertLogs(level="INFO") as cm:
            by_tile_aop(dpid=self.dpid, site=site, year=year, easting=[], northing=[])
            self.assertIn(
                f"INFO:root:{site} is part of the flight box for {flightSite}. Downloading data from {flightSite}.",
                cm.output,
            )

    @parameterized.expand(
        [
            ("2018", "BARC", "OSBS"),
            ("2020", "COMO", "NIWO"),
            # ("2021", "BLDE", "YELL"),
            # ("2020", "KING", "KONZ"),
        ],
        name_func=lambda f, n, p: f"{f.__name__}_{p.args[1]}_{p.args[0]}",
    )
    @patch("builtins.input", return_value="n")
    def test_collocated_aquatic_site_message(self, year, site, flightSite, input_mock):
        """
        Test application of the aquatic collocated site lookup, and expected message display.
        """
        with self.assertLogs(level="INFO") as cm:
            by_tile_aop(dpid=self.dpid, site=site, year=year, easting=[], northing=[])
            self.assertIn(
                f"INFO:root:{site} is an aquatic site and is sometimes included in the flight box for {flightSite}. Aquatic sites are not always included in the flight coverage every year.\nDownloading data from {flightSite}. Check data to confirm coverage of {site}.",
                cm.output,
            )

    def test_no_data_available_message(self):
        """
        Test that the by_tile_aop() function returns the expected error log when no data is available for a selected site and year.
        """
        with self.assertLogs(level="INFO") as cm:
            by_tile_aop(
                dpid="DP3.30015.001",
                site=self.site,
                year=2020,
                easting=self.easting,
                northing=self.northing,
            )
            self.assertIn(
                f"INFO:root:There are no DP3.30015.001 data available at the site {self.site} in 2020.\nTo display available dates for a given data product and site, use the function list_available_dates().",
                cm.output,
            )
            # 'INFO:root:There are no data available at the selected site and year.', cm.output)

    def test_no_data_files_found_message(self):
        """
        Test that the by_tile_aop() function returns the expected error log when no data files are found.
        """
        with self.assertLogs(level="INFO") as cm:
            by_tile_aop(
                dpid="DP3.30015.001",
                site=self.site,
                year=2020,
                easting=564000,
                northing=4900000,
            )
            self.assertIn(
                f"INFO:root:There are no DP3.30015.001 data available at the site {self.site} in 2020.\nTo display available dates for a given data product and site, use the function list_available_dates().",
                cm.output,
            )

    @patch("builtins.input", return_value="n")
    def test_check_download_size_message(self, input_mock):
        """
        Test that download check_size message displays correctly.
        """
        result = by_tile_aop(
            dpid=self.dpid,
            site=self.site,
            year=self.year,
            easting=self.easting,
            northing=self.northing,
        )
        # Check that the function asked for confirmation to download and prints expected message.
        input_mock.assert_called_once_with(
            "Continuing will download 7 files totaling approximately 3.9 MB. Do you want to proceed? (y/n) "
        )
        # Check that the function halted the download
        self.assertEqual(result, None)

    # Provisional scenarios

    # provisional not included, and no data available
    # these tests should be mocked, as provisional data changes with each NEON Data Release
    def test_all_provisional_no_data_available_message(self):
        """
        Test that the by_tile_aop() function returns the expected message when include_provisional is set to False (default) but no data are available.
        This has already run through the check that any data is available(eg. there is data at that site for the year provided)
        """
        with self.assertLogs(level="INFO") as cm:
            by_tile_aop(
                dpid=self.dpid,
                site="WLOU",
                year=2024,
                easting=self.easting,
                northing=self.northing,
            )
            self.assertIn(
                "INFO:root:No data files found. Available data may all be provisional. To download provisional data, use input parameter include_provisional=True.",
                cm.output,
            )

    # provisional included, and no data available
    @patch("builtins.input", return_value="n")
    def test_provisional_included_and_data_available_message(self, input_mock):
        """
        Test that the by_file_aop() function returns the expected message when include_provisional is set to False (default) but no data are available.
        This has already run through the check that any data is available(eg. there is data at that site for the year provided)
        """
        with self.assertLogs(level="INFO") as cm:
            by_tile_aop(
                dpid=self.dpid,
                site=self.site,
                year=2023,
                include_provisional=True,
                easting=self.easting,
                northing=self.northing,
            )
            self.assertIn(
                "INFO:root:Provisional data are included. To exclude provisional data, use input parameter include_provisional=False.",
                cm.output,
            )

    # pyproj is part of the requirements, so this is not needed - this was carried over from the R package
    # @ patch('builtins.input', return_value='n')
    # @ patch('importlib.import_module')
    # @ patch('logging.info')
    # def test_pyproj_not_installed(self, logging_mock, import_module_mock, input_mock):
    #     # Setup the mock to raise ImportError when pyproj is attempted to be imported
    #     import_module_mock.side_effect = ImportError(
    #         "No module named 'pyproj'")

    #     by_tile_aop(dpid="DP3.30015.001", site='BLAN', year=2022,
    #                 easting=243758.81, northing=4330667.37,
    #                 include_provisional=False, verbose=True)

    #     # Check if logging.info was called with the correct message
    #     # There will also be a message about the include_provisional, so can't use assert_called_once_with
    #     logging_mock.assert_any_call(
    #         "Package pyproj is required for this function to work at the BLAN site. Install and re-try."
    #     )

    @patch("builtins.input", return_value="n")
    def test_blan_utm_info_message(self, input_mock):
        """
        Test that the by_tile_aop() function returns the expected message about UTM zone conversion when BLAN is the site.
        """
        with self.assertLogs(level="INFO") as cm:
            by_tile_aop(
                dpid="DP3.30015.001",
                site="BLAN",
                year=2022,
                easting=243758.81,
                northing=4330667.37,
                verbose=True,
            )
            self.assertIn(
                "INFO:root:Blandy (BLAN) plots include two UTM zones, flight data are all in 17N. "
                "Coordinates in UTM zone 18N have been converted to 17N to download the correct tiles. "
                "You will need to make the same conversion to connect airborne to ground data.",
                cm.output,
            )
            self.assertIn("INFO:root:UTM 17N Easting(s): 762717.81", cm.output)
            self.assertIn("INFO:root:UTM 17N Northing(s): 4330881.38", cm.output)
            self.assertIn(
                "INFO:root:UTM (x, y) lower-left coordinates of tiles to be downloaded:",
                cm.output,
            )
            self.assertIn("INFO:root:(762000, 4330000)", cm.output)
