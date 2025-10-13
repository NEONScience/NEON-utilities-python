import pandas as pd
import os
import zipfile
from datetime import datetime
import re
import gzip
import shutil
import h5py
from tqdm import tqdm
import numpy as np
import warnings
import logging
from .citation import get_citation
from .get_issue_log import get_eddy_issue_log


def list_h5_contents(file_path):
    """
    List the contents of an H5 data file
    
    Parameters
    --------
        file_path (str): Path to the H5 file

    Return
    --------
        A data frame of H5 file contents
        
    """
    contents_ = []
    
    with h5py.File(file_path, 'r') as hdf:
        def create_listObj(group, otype):
            if group.count('/') == 0:
                name = group
                group = '/'
            else:
                groupSplit = group.split('/')
                name = groupSplit[-1]
                group = '/' + '/'.join(groupSplit[0:-1])
            
            contents_.append({'group': group, 'name': name, 'otype': type(otype).__name__})
        hdf.visititems(create_listObj)

    listObj = pd.DataFrame(contents_)
    return listObj


def time_stamp_set(tabDict):
    """
    Generate consensus SAE time stamps from a set of tables. Used in stack_eddy(), not intended for independent use.

    Parameters
    --------
        tabDict: A dictionary of SAE data frames

    Return
    --------
        A data frame of time stamps (start and end times) aggregated from the input tables
    
    """
    nameSet = ['timeBgn', 'timeEnd']
    
    # get a set of time stamps to initiate the table. leave out qfqm to exclude 
    #   - filler records created as placeholders for days with no data
    #   - turbulent flux and footprint end time stamps don't quite match the others
    timeSet = {}
    for key in tabDict.keys():
        if not re.search('qfqm|turb|foot', key):
            timeSet[key] = tabDict[key].copy()
    
    # initiate the table with consensus set of time stamps
    timeSetInit = timeSet[list(timeSet.keys())[0]][nameSet]
    
    if len(timeSet.keys()) == 1:
        pass
    else:
        for q in list(timeSet.keys())[1:]:
            # check for additional start time stamps
            timeSetTemp = timeSet[q][nameSet]
            timeSetTempMerg = pd.DataFrame(timeSetTemp['timeBgn'])
            timeSetInitMerg = pd.DataFrame(timeSetInit['timeBgn'])
            misTime = timeSetTempMerg.merge(timeSetInitMerg, on = 'timeBgn', how = 'left', indicator = True)
            misTime = misTime[misTime['_merge'] == 'left_only'].drop(columns = ['_merge'])
    
            if len(misTime) == 0:
                pass
            else:
                # combine all, then de-dup
                timeSetInit = pd.concat([timeSetInit, timeSetTemp], axis = 0, ignore_index = True).drop_duplicates()
    return timeSetInit


def eddy_stamp_check(tab):
    """
    Convert SAE time stamps to datetime and check for missing data

    Parameters
    --------
        tab: A data frame of SAE data

    Return
    --------
        The same table of SAE data, with time stamps converted and empty records representing a single day (filler records inserted during processing) removed.
    
    """
    tab = tab.copy()
    # convert time stamps
    tBgnErr = False
    tEndErr = False
    
    try:
        tabBP = pd.to_datetime(tab['timeBgn'].apply(lambda x: x.decode('utf-8')), format="%Y-%m-%dT%H:%M:%S.%fZ").dt.tz_localize('UTC').dt.tz_convert('GMT')
    except: 
        tBgnErr = True
    
    try:
        tabEP = pd.to_datetime(tab['timeEnd'].apply(lambda x: x.decode('utf-8')), format="%Y-%m-%dT%H:%M:%S.%fZ").dt.tz_localize('UTC').dt.tz_convert('GMT')
    except:
        tEndErr = True
    
    # if conversion failed, keep character time stamps and pass along message
    err = False
    
    if tBgnErr:
        err = True
    else:
        tab['timeBgn'] = tabBP
    
    if tEndErr:
        err = True
    else:
        tab['timeEnd'] = tabEP
    
    # if conversion was successful, check for single-day empty records and remove
    if err:
        pass
    else:
        dayDiff = [(EP - BP).total_seconds() for EP, BP in zip(tabEP, tabBP)]
        dayDup = [i for i, diff in enumerate(dayDiff) if diff > 86399]
        if len(dayDup) == 0:
            pass
        else:
            tab.drop(dayDup, inplace = True)

    return [tab, err]


def get_attributes(fil, sit, attType, valName = None):
    """
    Extract attribute metadata from H5 files

    Parameters
    --------
        fil: File path to the H5 file to extract attributes from
        sit: The site, for site attributes. Must match site of file path
        attType: The type of attributes to retrieve
        valName: If CO2 validation metadata are requested, the H5 name of the level where they can be found
    
    Return
    --------
        A dictionary containing the extracted attributes

    """
    if fil.count('basic') >= 1:
        mnth = re.search(r"20[0-9]{2}-[0-9]{2}", fil).group()
    else:
        mnth = re.search(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}", fil).group()
    
    if attType == 'site':
        lev = str(sit)
    if attType == 'root':
        lev = '/'
    if attType == 'val':
        lev = valName
    
    try: 
        with h5py.File(fil, 'r') as hdf:
            gAttr = dict(hdf[lev].attrs)
            for key, val in gAttr.items():
                try:
                    gAttr[key] = [item.decode('utf-8') for item in val]
                except:
                    pass
            gAttr['site'] = [str(sit)]
            gAttr['date'] = [mnth]
    except:
        gAttr = {'site': [str(sit)], 'date': [mnth]}
    
    return gAttr


def get_variables_eddy(tabDict):
    """
    Extract variable names and units from SAE H5 files and return in user-friendly form. Used in stack_eddy(), not intended for independent use.
    
    Parameters
    -------- 
        tabDict: A dictionary of SAE data frames

    Return
    -------- 
        A data frme of variable names and units, aggregated from the input tables
    
    """
    variables = pd.DataFrame({'category': [], 'system': [], 'variable': [], 'stat': [], 'units': []})
    for p in tabDict.keys():
        for q in tabDict[p].keys():
            if 'unit' in tabDict[p][q].keys():
                var_nm = []
                keySplit = q.split('/')
                var_nm.append(keySplit[3])
                var_nm.append(keySplit[4])
                var_nm.append(keySplit[-1])
                units = tabDict[p][q]['unit']
            
                if len(units) > 1:
                    variables_q = pd.DataFrame({'category': [var_nm[0]] * len(units), 'system': [var_nm[1]] * len(units), 'variable': [var_nm[2]] * len(units)})
                    names = tabDict[p][q]['data'].columns
                    
                    if len(units) == len(names):
                        variables_q['stat'] = names
                    else:
                        if 'index' in names:
                            variables_q['stat'] = [name for name in names if name != 'index']
                        else:
                            variables_q['stat'] = [name for name in names if name not in ['timeBgn', 'timeEnd']]
            
                    variables_q['units'] = units
                    variables = pd.concat([variables, variables_q], ignore_index = True)
                else:
                    variables_q = pd.DataFrame({'category': [var_nm[0]], 'system': [var_nm[1]], 'variable': [var_nm[2]], 'stat': None, 'units': units})
                    variables = pd.concat([variables, variables_q], ignore_index = True)
            
    variables.drop_duplicates(inplace = True, ignore_index = True)
    
    if len(variables) == 0:
        variables = None
    
    return variables


def get_vars_eddy(filepath):
    """
    Extracts a data frame of table metadata from a single HDF5 file. Specific to eddy covariance data product DP4.00200.001. Can inform inputs to stack_eddy(); variables listed in 'name' are available inputs to the 'var' parameter in stack_eddy().
    
    Parameters
    --------
        filepath (str): The path to the H5 file
    
    Return
    --------
        A data frame of the metadata for each data table in the HDF5 file
    
    """
    try:
        listObj = list_h5_contents(filepath)
        listDataObj = listObj.loc[listObj['otype'] == 'Dataset']
    
        listObjSpl = listDataObj.copy()
        listObjSpl[['blank', 'site', 'level', 'category', 'system', 'horvertmi']] = listObjSpl['group'].str.split('/', expand = True)
        listObjSpl[['hor', 'ver', 'tmi']] = listObjSpl['horvertmi'].str.split('_', expand = True)
        listObjSpl.drop(columns=['group','blank', 'horvertmi'], axis=1, inplace=True)
    
        return listObjSpl
    except:
        raise ValueError(filepath + ' could not be read.')


def stack_eddy(filepath, level='dp04', var=None, avg=None, metadata=False, runLocal=False):
    """
    Convert data of choice from HDF5 to tabular format. Specific to eddy covariane data product: DP4.00200.001

    Parameters
    --------
        filepath: One of 1) a folder containing NEON EC H5 files, 2) a zip file of DP4.00200.001 data downloaded from the data portal, 3) a list of H5 files, 4) a single EC H5 file
        level: The level of data to extract; one of dp01, dp02, dp03, dp04
        var: The variable set to extract. Can be any of the variables in the "name" level of the "system" level of the H5 file; use getVarsEddy() function to see the available variables. From the inputs, all variables from "name" and all variables from "system" will be returned, but if variables from both "name" and "system" are specified, the function will return only the intersecting set. This allows the user to, e.g., return only the pressure data ("pres") from the CO2 storage system ("co2Stor"), instead of all the pressure data from all instruments.
        avg: The averaging interval to extract, in minutes.
        metadata: Should the output include metadata from the attributes of the H5 files? Defaults to false. Even when false, variable definitions, issue logs, and science review flags will be included.
        runLocal: Set to True to omit any calls to the NEON API. Data are extracted and reformatted from local files, but citation and issue log are not retrieved.

    Return
    --------
        varMergDict: A dictionary of dataframes. One data frame per site, plus one dat frame containing the metadata (objDesc) table and one data frame containing units for each variable.
    
    """

    files = []
    releases = []
    skipKeys = []
    
    # check input types
    if isinstance(var, str):
        var = [var]
    if not isinstance(level, str):
        raise ValueError("level must be one of dp01, dp02, dp03, dp04")
    if not level in ["dp01", "dp02", "dp03", "dp04"]:
        raise ValueError("level must be one of dp01, dp02, dp03, dp04")

    # check for list of h5 files as input
    if isinstance(filepath, list):
        if sum(path.count(".h5") for path in filepath) == len(filepath):
            files = filepath
        else:
            raise ValueError("Input list of files must be .h5 files.")
        for path in filepath:
            if not os.path.exists(path):
                raise ValueError("Files not found in specified filepaths. Check that the input list contains the correct filepath.")
            
    # get list of files, unzipping if necessary
    if len(files) == 0 and isinstance(filepath, str) and filepath.endswith(".zip"):
        outpath = filepath.replace(".zip", "")
        
        os.makedirs(outpath, exist_ok = True)
        
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_list = zip_ref.namelist()
            
            if any(".zip" in name for name in zip_list):
                zip_ref.extractall(outpath)
            
            else:
                # get release info before discarding file paths
                all_dirs = [os.path.dirname(name) for name in zip_list]
                pattern = '(RELEASE|PROVISIONAL|LATEST).*'
                release_status = []
                for file in all_dirs:
                    match_ = re.finditer(pattern, file)
                    for match in match_:
                        release_status.append({'name': file, 'release': match.group(0)})
                
                release_status = pd.DataFrame(release_status)
                releases = release_status['release'].unique()
                
                # write release status file
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                release_status.to_csv(os.path.join(outpath, f"release_status_{timestamp}.csv"), index=False)
                
                # unzip
                for file in zip_list:
                    zip_ref.extract(file, outpath)
        
        filepath = outpath

    # allow for a single H5 file, or folder of H5 files
    if len(files) == 0 and isinstance(filepath, str) and filepath.endswith(".h5"):
        files = [filepath]
    else: 
        if len(files) == 0:
            for root, dirs, filenames in os.walk(filepath):
                for filename in filenames:
                    files.append(os.path.join(root, filename))

    # unzip files if necessary
    if sum(file.count(".zip") for file in files) == len(files):
        for file in files:
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(filepath)
        files = os.listdir(filepath)

    # after unzipping, check for .gz
    if any(file.endswith(".h5.gz") for file in files):
        gzfiles = [file for file in files if re.search(".h5.gz", file)]
        
        for gzfile in gzfiles:
            with gzip.open(gzfile, 'rb') as f_in:
                with open(gzfile.removesuffix(".gz"), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(gzfile)

        files = []
        for root, dirs, filenames in os.walk(filepath):
            for filename in filenames:
                files.append(os.path.join(root, filename))

    # need the H5 files for data extraction and the SRF tables
    scienceReviewList = [file for file in files if re.search("science_review_flags", file)]
    files = [file for file in files if re.search(".h5$", file)]

    # check for no files
    if len(files) == 0:
        raise ValueError("No .h5 files found in specified file path. Check the inputs and file contents.")

    # check for original zip files and use to determine releases and citations
    if len(releases) == 0 and isinstance(filepath, str) and os.path.isdir(filepath):
        allFiles = os.listdir(filepath)
        relfl = [file for file in allFiles if re.search("release_status", file)]
        if len(relfl) == 1:
            reltab = pd.read_csv(relfl)
            releases = reltab['release'].unique()
        else:
            allFileNames = [os.path.splitext(file)[0] for file in allFiles]
            pattern = '(RELEASE|PROVISIONAL|LATEST).*'
            release_status = []
            for file in allFileNames:
                match_ = re.finditer(pattern, file)
                for match in match_:
                    release_status.append(match.group(0))
            
            releases = list(set(release_status))

    # get DOIs and generate citations(s)
    if not runLocal:
        citP = None
        citR = None
        if "PROVISIONAL" in releases:
            try:
                citP = get_citation(dpid = "DP4.00200.001", release = "PROVISIONAL")
            except:
                citP = None
        if sum(release.count("RELEASE") for release in releases) == 0:
            pass
        else: 
            if sum(release.count("RELEASE") for release in releases) > 1:
                raise ValueError("Attempting to stack multiple data releases together. This is not appropriate, check your input data")
            else:
                rel = [release for release in releases if re.search("RELEASE", release)]
                try:
                    citR = get_citation(dpid = "DP4.00200.001", release = rel[0])
                except:
                    citR = None

    # determine basic vs expanded package and check for inconsistencies
    pkgCount = sum(bool(re.search("basic\\.[a-zA-Z0-9]+\\.(h5)$", file)) for file in files)

    if len(files) == pkgCount:
        pkg = 'basic'
    elif pkgCount == 0:
        pkgCount = len([re.findall("expanded\\.[a-zA-Z0-9]+\\.h5$", file) for file in files])
        pkg = 'expanded'

    if len(files) != pkgCount:
        raise ValueError("File path contains both basic and expanded package files, these can't be stacked together.")

    if pkg == 'basic' and metadata == True:
        print("For the basic package, attribute metadata are the values from the beginning of the month. To get attributes for each day, use the expanded package.")

    # make empty dictionary for the data tables
    tableDict = {}

    for file in files:
        tableDict[file.replace(".h5", "")] = []

    # extract data from each file
    for file in tqdm(files, desc = "Extracting data"):
        try:
            if isinstance(filepath, list):
                h5path = file
            else:
                h5path = os.path.join(filepath, file)
                
            listObj = list_h5_contents(h5path)
            listDataObj = listObj.loc[listObj['otype'] == 'Dataset'].reset_index()
            listDataName = listDataObj['group'] + '/' + listDataObj['name']
            
            listObjSpl = pd.DataFrame(listDataObj['group'].copy())
            colSpl = listObjSpl['group'].str.split('/', expand = True)
            for i in range(colSpl.shape[1], 7):
                colSpl[i] = None
            colSpl.columns = ['blank', 'site', 'level', 'category', 'system', 'horvertmi', 'subsys']
            listObjSpl = pd.concat([listObjSpl, colSpl], axis=1)
            listObjSpl.drop(columns=['group', 'blank'], axis=1, inplace=True)
            
            # filter by variable/level selections
            levelInd = pd.Index([listDataName.index[i] for i, name in enumerate(listDataName) if re.search(level, name)])
            
            if level not in ["dp04", "dp03", "dp02"] and var is not None:
                if len([value for value in listObjSpl['system'].values if value in var]) > 0:
                    if len([value for value in listDataObj['name'].values if value in var]) > 0:
                        varInd = listObjSpl.loc[listObjSpl['system'].isin(var)].index.intersection(listDataObj.loc[listDataObj['name'].isin(var)].index)
                    else:
                        varInd = listObjSpl.loc[listObjSpl['system'].isin(var)].index
                else: 
                    if len([value for value in listDataObj['name'].values if value in var]) > 0:
                        varInd = listDataObj.loc[listDataObj['name'].isin(var)].index
                    else:
                        raise ValueError("No data found for variables: " + ", ".join(var))
            else:
                varInd = pd.Index(range(0, len(listDataName)))
            
            if level not in ["dp04", "dp03", "dp02"] and avg is not None:
                avgInd = pd.Index([listDataName.index[i] for i, name in enumerate(listDataName) if re.search(str(avg) + 'm', name)])
            else:
                if level == "dp01" and avg is None:
                    raise ValueError("If level is dp01, avg is a required input.")
                else:
                    avgInd = pd.Index(range(0, len(listDataName)))
            
            # exclude footprint grid data from expanded packages
            if sum(name.count("foot/grid") for name in listDataName) > 0:
                gridInd = pd.Index([listDataName.index[i] for i, name in enumerate(listDataName) if not re.search('foot/grid', name)])
            else:
                gridInd = pd.Index(range(0, len(listDataName)))
            
            # index that includes all filtering criteria
            ind = gridInd.intersection(levelInd).intersection(varInd).intersection(avgInd)
            
            # check that you haven't filtered to nothing
            tableDictKey = file.replace(".h5", "")
            if len(ind) == 0:
                tableDict["skip" + str(i)] = tableDict.pop(tableDictKey, None)
                skipKeys.append("skip" + str(i))
                logging.info("There are no data meeting the criteria level " + str(level) + ", averaging interval " + str(avg) + ", and variables " + ", ".join(var))
            else:
                listDataName = listDataName[ind]
                
                # add extracted data to the dictionary
                dataListDict = {}
                with h5py.File(h5path, 'r') as hdf:
                    for dataSet in listDataName:
                        dataListDict[dataSet] = {}
                        dataListDict[dataSet]['data'] = pd.DataFrame(hdf[dataSet][:])
                        try:
                            dataListDict[dataSet]['unit'] = [b.decode('utf-8') for b in list(hdf[dataSet].attrs['unit'])]
                        except:
                            pass
                        
                tableDict[tableDictKey] = dataListDict
        except ValueError:
            logging.info('An input value was invalid.')
            raise
        except Exception:
            logging.info(file + " could not be read.")
            pass
        else:
            pass

    # remove skipped files from dict list
    for key in skipKeys:
        tableDict.pop(key, None)

    # check for no data left
    if len(tableDict) == 0:
        raise ValueError("No data met requested criteria in any file")
    
    # get variable units
    variables = get_variables_eddy(tableDict)

    # convert all time stamps to time format, then filter out instances with:
    # 1) only one record for a day
    # 2) all values = NaN
    # these are instances when a sensor was offline, and they don't join correctly
    err = False

    for k1 in tableDict.keys():
        for k2 in tableDict[k1].keys():
            tabTemp = eddy_stamp_check(tableDict[k1][k2]['data'])
            if tabTemp[1]:
                err = True
            tableDict[k1][k2]['data'] = tabTemp[0]

    if err:
        print("Some time stamps could not be converted. Variable join may be affected; check data carefully for disjointed time stamps")

    # within each site-month set, join matching tables
    # create empty dict for the tables
    mergTableDict = {}
    for key in tableDict.keys():
        mergTableDict[key] = {}

    for key in tqdm(tableDict.keys(), desc = "Joining data variables by file"):
        namesSpl = []
        for key2 in tableDict[key].keys():
            namesSpl.append(key2.split('/')[1:])
        
        namesSpl = pd.DataFrame(namesSpl)
        nc = len(namesSpl.columns)
        
        # dp01 and dp02 have sensor levels
        if nc == 6:
            sens = namesSpl[4].unique()
        
            # join for each sensor level
            for si in sens:
                # get all tables for a sensor level and get variable names
                tbsub = {}
                nmsub = []
                inds = namesSpl.loc[namesSpl[4] == si, :]
                for ind, row in inds.iterrows():
                    tkey = '/' + '/'.join(row)
                    tbsub[tkey] = tableDict[key][tkey]['data'].copy()
                    nmsub.append('.'.join([row[2], row[3], row[5]]))
                    mergTabl = time_stamp_set(tbsub)
        
                for i, ib in enumerate(tbsub.keys()):
                    tbsub[ib].rename(columns = {col: (nmsub[i] + '.' + col if 'timeBgn' not in col and 'timeEnd' not in col else col) for col in tbsub[ib].columns}, inplace = True)
                    tbsub[ib].drop(columns = 'timeEnd', inplace = True)
        
                    mergTabl = pd.merge(mergTabl, tbsub[ib], on = 'timeBgn', how = 'left')
        
                mergTableDict[key][si] = mergTabl

        # dp03 and dp04 - no sensor levels
        # get consensus time stamps
        else:
            tbsub = {}
            nmsub = []
            for ind, row in namesSpl.iterrows():
                tkey = '/' + '/'.join(row)
                tbsub[tkey] = tableDict[key][tkey]['data'].copy()
                nmsub.append('.'.join([row[2], row[3], row[4]]))
                mergTabl = time_stamp_set(tbsub)
            
            for i, ibl in enumerate(tbsub.keys()):
                tbsub[ibl].rename(columns = {col: (nmsub[i] + '.' + col if 'timeBgn' not in col and 'timeEnd' not in col else col) for col in tbsub[ibl].columns}, inplace = True)
                tbsub[ibl].drop(columns = 'timeEnd', inplace = True)
            
                mergTabl = pd.merge(mergTabl, tbsub[ibl], on = 'timeBgn', how = 'left')
            
                mergTableDict[key] = mergTabl

    # for dp01 and dp02, stack tower levels and calibration gases
    if level in ['dp01', 'dp02']:
        verMergDict = {}
        for ni in mergTableDict.keys():
            verMergList = []
            for mi in mergTableDict[ni].keys():
                verSpl = mi.split('_')
                
                if len(verSpl) == 3:
                    mergTableDict[ni][mi].insert(0, 'verticalPosition', verSpl[1])
                    mergTableDict[ni][mi].insert(0, 'horizontalPosition', verSpl[0])
                else:
                    mergTableDict[ni][mi].insert(0, 'verticalPosition', verSpl[0])
                    mergTableDict[ni][mi].insert(0, 'horizontalPosition', None)

                verMergList.append(mergTableDict[ni][mi])
            verMergDict[ni] = pd.concat(verMergList, ignore_index = True)

    else:
        verMergDict = mergTableDict.copy()

    # check for weird isotope joining - return to this with a avg = 9 example
    if level == 'dp01' and avg in [9, '9', '09']:
        allNm = []
        for key in verMergDict.keys():
            allNm.append(verMergDict[key].columns)
        allNm = np.unique(allNm)

        if sum(col.count('dlta13CCo2') for col in allNm) > 0 and sum(col.count('dlta18OH2o') for col in allNm) > 0 or sum(col.count('dlta2HH2o') for col in allNm) > 0:
            dup_iso_list = [(x['timeEnd'].iloc[:20] - x['timeBgn'].iloc[:20]).dt.total_seconds().mean(skipna=True) < 537 for x in verMergList]
            dup_iso = any(dup_iso_list)

            if dup_iso:
                warnings.warn("Stacking appears to include both carbon and water isotopes, with inconsistent time stamps. "
                            "Carbon isotopes are measured every 6 minutes, water isotopes every 9 minutes. "
                            "This issue affects RELEASE-2023 and provisional data published between RELEASE-2023 and RELEASE-2024. "
                            "Check data carefully. The recommended workflow is to stack the carbon and water isotope data separately. ")
                
    # set up tables for data and metadata
    # stack months within each site
    sites = np.unique([re.search('\\.([A-Z]+)\\.', key).group(1) for key in verMergDict.keys()])

    # which attributes should be extracted?
    vNames = np.unique([list(verMergDict[key].columns) for key in verMergDict.keys()])
    if metadata:
        if any('rtioMoleDryCo2Vali' in vN for vN in vNames):
            varNames = np.concatenate((sites, ['variables', 'objDesc', 'siteAttributes', 'codeAttributes', 'validationAttributes', 'issueLog', 'scienceReviewFlags']))
            numTabs = 7
        else:
            varNames = np.concatenate((sites, ['variables', 'objDesc', 'siteAttributes', 'codeAttributes', 'issueLog', 'scienceReviewFlags']))
            numTabs = 6
    else:
        varNames = np.concatenate((sites, ['variables', 'objDesc', 'issueLog', 'scienceReviewFlags']))
        numTabs = 4

    # set up final dict
    varMergDict = {}
    for name in varNames:
        varMergDict[str(name)] = []

    for site in tqdm(sites, desc = 'Stacking files by month'):
        mergList = [value for key, value in verMergDict.items() if site in key]
        varMergDict[site] = pd.concat(mergList, ignore_index = True)
        
        # sort by site, hor and ver, and date
        if 'verticalPosition' in varMergDict[site].columns:
            varMergDict[site] = varMergDict[site].sort_values(by=['horizontalPosition', 'verticalPosition', 'timeBgn'], ignore_index=True)
        else:
            varMergDict[site] = varMergDict[site].sort_values(by=['timeBgn'], ignore_index=True)
            
    # attributes, objDesc, SRF table, and issue log
    # attributes are only included if metadata == TRUE
    if metadata:
        siteAttr = []
        codeAttr = []
        for site in tqdm(sites, desc='Getting metadata tables'):
            for file_ in files:
                if re.search(site, file_):
                    sAttr = get_attributes(file_, site, attType='site')
                    siteAttr.append(sAttr)
                    cAttr = get_attributes(file_, site, attType='root')
                    codeAttr.append(cAttr)      
        siteAttributes = pd.DataFrame(siteAttr).map(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else ', '.join(map(str, x)))
        codeAttributes = pd.DataFrame(codeAttr).map(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else ', '.join(map(str, x)))
        varMergDict['siteAttributes'] = siteAttributes
        varMergDict['codeAttributes'] = codeAttributes
        # get CO2 validation attributes if CO2 variables were extracted 
        if 'validationAttributes' in varNames: 
            valAttr = []
            for site in tqdm(sites, desc='Getting validation metadata tables'):
                for okey, idict in tableDict.items():
                    for ikey in idict:
                        if 'rtioMoleDryCo2Vali' and site in ikey:
                            vAttr = get_attributes(okey+'.h5', site, 'val', valName=ikey)
                            valAttr.append(vAttr)
            valAttributes = pd.DataFrame(valAttr).map(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else ', '.join(map(str, x)))
            varMergDict['validationAttributes'] = valAttributes

    # get one objDesc table and add it and variables table to dict
    try:
        with h5py.File(os.path.join(filepath, files[0]), 'r') as h5:
            objDesc = h5['//objDesc'][:]
            objDesc = pd.DataFrame([(x[0].decode('utf-8'), x[1].decode('utf-8')) for x in objDesc], columns = ['Object', 'Description'])
    except:
        # if processing gets this far without failing, don't fail here, just return data without objDesc table
        objDesc = None

    varMergDict['variables'] = variables
    varMergDict['objDesc'] = objDesc

    # get issue log
    if not runLocal:
        try:
            varMergDict['issueLog'] = get_eddy_issue_log(dpid = 'DP4.00200.001')
        except:
            print('Issue log file not accessed - issue log can be found on the data product details pages.')
            varMergDict.pop('issueLog', None)

    # aggregate the science_review_flags files
    if len(scienceReviewList) > 0:
        outputScienceReview = pd.DataFrame()
        for srFile in scienceReviewList:
            srTemp = pd.read_csv(os.path.join(filepath, srFile))
            outputScienceReview = pd.concat([outputScienceReview, srTemp], ignore_index = True)
        # remove duplicates
        outputScienceReview.drop_duplicates(inplace=True, ignore_index=True)

        # check for non-identical duplicates with the same ID and keep the most recent one
        if len(outputScienceReview['srfID'].unique()) != len(outputScienceReview):
            outputScienceReview['lastUpdateDateTime_s'] = pd.to_datetime(outputScienceReview['lastUpdateDateTime'])
            outputScienceReview = outputScienceReview.sort_values(by=['srfID', 'Time'], ascending = [True, False])
            outputScienceReview.drop_duplicates(subset='srfID', keep='first', inplace=True)
            outputScienceReview.drop(columns=['lastUpdateDateTime_s'], inplace=True)

        varMergDict['scienceReviewFlags'] = outputScienceReview

    else:
        varMergDict.pop('scienceReviewFlags', None)

    # add citations to output
    if not runLocal:
        if citP is not None:
            varMergDict['citation_00200_PROVISIONAL'] = citP
        if citR is not None:
            keyString = 'citation_00200_' + rel[0]
            varMergDict[keyString] = citR

    return varMergDict