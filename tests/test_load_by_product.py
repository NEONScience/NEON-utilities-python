# -*- coding: utf-8 -*-
"""
Created on 15 Aug 2024

@author: Claire Lunch

Unit tests for load_by_product()

"""

# import required packages
from src.neonutilities.unzip_and_stack import load_by_product
import pandas as pd
import os

token = os.environ.get("token")


def test_load_by_product_IS():
    """
    Test that load_by_product() accesses and stacks the example data correctly.
    """
    tlist = load_by_product(dpid='DP1.00005.001', site=['TOOL', 'PUUM'],
                            startdate='2022-06', enddate='2022-07',
                            check_size=False, progress=False,
                            release='RELEASE-2024', cloud_mode=True,
                            token=token)
    assert list(tlist.keys()) == ['IRBT_1_minute', 'IRBT_30_minute', 'citation_00005_RELEASE-2024',
                                  'issueLog_00005', 'readme_00005', 'science_review_flags_00005',
                                  'sensor_positions_00005', 'variables_00005']
    tm = tlist['IRBT_30_minute']
    assert len(tm) == 23424
    assert max(tm['endDateTime']) == pd.Timestamp('2022-08-01 00:00:00+0000', tz='UTC')
    assert min(tm['endDateTime']) == pd.Timestamp('2022-06-01 00:30:00+0000', tz='UTC')
