# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:17:04 2024

@author: bhass

Unit tests for get_issue_log

These will test static part of issue logs, since change and issue logs are subject to change.

Notes:
- These tests require an internet connection to run, as they are actually making API calls

"""

import unittest
from src.neonutilities.get_issue_log import get_issue_log


class TestGetIssueLog(unittest.TestCase):
    def test_get_issue_log_chm(self):
        # Call the function for CHM
        df = get_issue_log('DP3.30015.001')
        # Check the column names
        self.assertEqual(list(df.columns), ['id', 'parentIssueID', 'issueDate',
                                            'resolvedDate', 'dateRangeStart',
                                            'dateRangeEnd', 'locationAffected',
                                            'issue', 'resolution'])

        self.assertEqual(
            df.loc[0, 'issue'][0:57], 'Original algorithm did not take advantage of improvements')
        self.assertEqual(df.loc[0, 'issueDate'], '2022-01-19T00:00:00Z')

    def test_get_issue_log_eddy(self):
        # Call the function for bundled eddy covariance
        df = get_issue_log('DP4.00200.001')

        # Check the column names
        self.assertEqual(list(df.columns), ['dpid', 'id', 'parentIssueID', 'issueDate',
                                            'resolvedDate', 'dateRangeStart',
                                            'dateRangeEnd', 'locationAffected',
                                            'issue', 'resolution'])

        # # Check a few rows
        # # Replace with the expected value
        # self.assertEqual(df.loc[0, 'col1'], 'expected value')
        # # Replace with the expected value
        # self.assertEqual(df.loc[10, 'col2'], 'expected value')
