#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import os
import platform
import logging
import glob
import importlib.metadata
import pandas as pd
from tqdm import tqdm
from urllib.parse import urlparse
from .unzip_and_stack import unzip_zipfile
from .helper_mods.metadata_helpers import convert_byte_size 

# Set global user agent
vers = importlib.metadata.version('neonutilities')
plat = platform.python_version()
osplat = platform.platform()
usera = f"neonutilities/{vers} Python/{plat} {osplat}"

def files_by_uri(filepath,
                 savepath=None,
                 check_size=True,
                 unzip=True,
                 save_zipped_files=False,
                 progress=True,
                 ):
    """

    Get files from NEON GCS Bucket using URLs in stacked data
    
    Parameters
    --------
    filepath: The location of the NEON data containing URIs. Can be either a local directory containing NEON tabular data or a list object containing tabular data.
    savepath: The location to save the output files from the GCS bucket, optional. Defaults to creating a "GCS_zipFiles" folder in the filepath directory.
    check_size: should the user be told the total file size before downloading, defaults to True? (true/false)
    unzip: indicates if the downloaded files from GCS buckets should be unzipped into the same directory, defaults to True. Supports .zip and .tar.gz files currently. (true/false)
    save_zipped_files: Should the unzipped monthly data folders be retained, defaults to False? Supports .zip and .tar.gz files currently. (true/false)
    progress: Should a progress bar be displayed?
    
    Return
    --------
    A folder in the working directory (or in savepath, if specified), containing all files meeting query criteria.
    
    Example
    --------
    ZN NOTE: Insert example when function is coded
    
    >>> example
    
    Created on Fri Aug 9 2024
    
    @author: Zachary Nickerson
    """ 
    
    # check that filepath points to either a directory or a Python list object
    if not isinstance(filepath, dict):
        if not os.path.exists(filepath):
            raise Exception("Input filepath is not a list object in the environment nor an existing file directory.")
            
    # if filepath is a dictionary, make a savepath
    if isinstance(filepath, dict):
        tabList = filepath
        if not os.path.exists(savepath):
            try:
                os.makedirs(savepath)
            except OSError:
                print(f"Could not create savepath directory. Files will be saved to {os.getcwd()}/GCS_zipFiles")
                savepath = f"{os.getcwd()}/GCS_zipFiles"
    else:
        # if filepath is a directory, read in contents
        files = [os.path.join(filepath, f) for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]
        tabList = {}
        for j, file in enumerate(files):
            try:
                tabList[file] = pd.read_csv(file)
            except Exception as e:
                print(f"File {file} could not be read.")
                tabList[f"error{j}"] = None
                continue
        tabList = {k: v for k, v in tabList.items() if not k.startswith('error')}
        
    # Check for the variables file in the filepath
    varList = [k for k in tabList.keys() if "variables" in k]
    if len(varList) == 0:
        raise Exception("Variables file was not found in specified filepath.")
    if len(varList) > 1:
        raise Exception("More than one variables file found in filepath.")
    varFile = tabList[varList[0]]
    
    URLs = varFile[varFile['dataType'] == 'uri']
    
    # All of the tables in the package with URLs to download
    allTables = URLs['table'].unique()
    
    # Loop through tables and fields to compile a list of URLs to download
    URLsToDownload = []
    
    # Remove allTables values that aren't in tabList
    allTables = {key for key in tabList.keys() for substr in allTables if substr in key}
    
    if len(allTables) < 1:
        raise Exception('No tables with URIs available in download package contents.')
    
    # Loop through tables
    for table in allTables:
        tableData = tabList[table]
        # Find URLs per table that are in URLs.fieldName
        URLsPerTable = [url for url in tableData if url in URLs['fieldName'].values]
        # Append the URLs from the fields found
        URLsToDownload.extend([tableData[url] for url in URLsPerTable])
    
    # Remove duplicates from the list of URLs
    uniqueURLs = set()
    for lst in URLsToDownload:
        uniqueURLs.update(lst)
    URLsToDownload = uniqueURLs
    
    # Remove None values
    URLsToDownload = [url for url in URLsToDownload if url is not None]
    URLsToDownload = [url for url in URLsToDownload if str(url) != 'nan']
    URLsToDownload = [url for url in URLsToDownload if str(url) != '']

    
    if len(URLsToDownload) == 0:
        raise Exception("There are no URLs other than 'None' for the stacked data.")
    
    # Create directory only if it does not already exist
    if not savepath is None:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
    else:
        # Make the savepath the working directory
        savepath = os.getcwd()
    
    # Check the existence and size of each file from URL
    if check_size:
        logging.info(f"Checking size of downloading {len(URLsToDownload)} files by URI")
        sz = []
        for urlfile in tqdm(URLsToDownload, disable=not progress):
            response = requests.head(urlfile,
                                    headers={"User-Agent": usera},
                                    allow_redirects=True)
            
            # Return nothing if response failed
            if response.status_code != 200:
                logging.info("Connection error for a subset of urls. Check outputs for missing data.")
                # return None
            
            # Compile file sizes
            flszi = int(response.headers['Content-Length'])
            sz.append(flszi)
        
        # Check download size
        download_size = convert_byte_size(sum(sz))
        if input(f"Continuing will download {len(URLsToDownload)} files totaling approximately {download_size}. Do you want to proceed? (y/n) ") != "y":
            logging.info("Download halted.")
            return None
        else:
            logging.info(f"Downloading {len(URLsToDownload)} files totaling approximately {download_size}.")
        
    # Download URLs
    for j in tqdm(URLsToDownload, disable=not progress):
        parsed_url = urlparse(j)
        filename = os.path.basename(parsed_url.path)
        file_path = os.path.join(savepath, filename)
        # Download the file
        response = requests.get(j,
                                headers={"User-Agent": usera})
        with open(file_path, 'wb') as out_file:
            out_file.write(response.content)
        # If the file type is zip and unzip is True, unzip
        if unzip is True:
            if(file_path.endswith('.zip')):
                unzip_zipfile(zippath=file_path)
                # Remove zip files after being unzipped
                if save_zipped_files is False:
                    os.remove(file_path)

