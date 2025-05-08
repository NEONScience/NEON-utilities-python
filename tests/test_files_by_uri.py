# -*- coding: utf-8 -*-
"""
Created on 14 Apr 2025

@author: Zachary Nickerson

Unit tests for files_by_uri()

"""

# import required packages
from src.neonutilities.files_by_uri import files_by_uri
import os
import shutil

def test_files_by_uri_NEF():
    """
    Test that the function works for NEF files available from DP1.10017.001 (tabular data saved in testdata)
    """
    testdir = "./testdata/NEON_uri_testdata/10017_DELA_202306_RELEASE2025"
    files_by_uri(testdir,
                 savepath=testdir,
                 check_size=False,
                 unzip=True,
                 save_zipped_files=False,
                 progress=True)
    # There should be 36 .NEF files in the directory
    files = os.listdir(os.path.join(testdir,"GCS_files"))
    nef_files = [file for file in files if file.lower().endswith('.nef')]
    assert len(nef_files) == 36
    # Remove the .NEF files saved in the testdir
    shutil.rmtree(os.path.join(testdir,"GCS_files"))

def test_files_by_uri_ZIP():
    """
    Test that the function works for ZIP files available from DP4.00131.001 (tabular data saved in testdata)
    """
    testdir = "./testdata/NEON_uri_testdata/00131_allSites_2022_RELEASE2025"
    files_by_uri(testdir,
                 savepath=testdir,
                 check_size=False,
                 unzip=False,
                 save_zipped_files=False,
                 progress=True)
    # There should be 21 .ZIP files in the directory
    files = os.listdir(os.path.join(testdir,"GCS_files"))
    zip_files = [file for file in files if file.lower().endswith('.zip')]
    assert len(zip_files) == 21
    # Remove the .NEF files saved in the testdir
    shutil.rmtree(os.path.join(testdir,"GCS_files"))    
    
def test_files_by_uri_ZIP_unzip():
    """
    Test that the function works for ZIP files available from DP4.00131.001, and unzip the files (tabular data saved in testdata)
    """
    testdir = "./testdata/NEON_uri_testdata/00131_allSites_2022_RELEASE2025"
    files_by_uri(testdir,
                 savepath=testdir,
                 check_size=False,
                 unzip=True,
                 save_zipped_files=False,
                 progress=True)
    # There should be 294 files that are not .CSV in the directory
    files = os.listdir(os.path.join(testdir,"GCS_files"))
    geo_files = [file for file in files if not file.lower().endswith('.csv')]
    assert len(geo_files) == 294
    # Remove the .NEF files saved in the testdir
    shutil.rmtree(os.path.join(testdir,"GCS_files"))

# Potentially add a test for the microbial file types