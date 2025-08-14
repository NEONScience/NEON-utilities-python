#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import pyarrow as pa
from pyarrow import dataset
from pyarrow import fs
import zipfile
import platform
import glob
import re
import warnings
import importlib_resources
from tqdm import tqdm
from .tabular_download import zips_by_product
from .get_issue_log import get_issue_log
from .citation import get_citation
from .helper_mods.api_helpers import readme_url
from .helper_mods.api_helpers import get_api
from .read_table_neon import get_variables, cast_table_neon
from . import __resources__
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

varschema = pa.schema(
    [
        ("table", pa.string()),
        ("fieldName", pa.string()),
        ("description", pa.string()),
        ("dataType", pa.string()),
        ("units", pa.string()),
        ("sampleCode", pa.string()),
        ("downloadPkg", pa.string()),
        ("pubFormat", pa.string()),
        ("primaryKey", pa.string()),
        ("categoricalCodeName", pa.string()),
    ]
)


def unzip_zipfile(zippath):
    """
    Unzip a zip file either at just the top level or recursively through the file.

    Parameters
    --------
    zippath: The filepath of the input file.

    Return
    --------
    A list of unzipped files to be used in stack_by_table.

    Example
    --------
    unzip_zipfile(zippath=filepath)

    Created on Tue Mar 5 2024

    @author: Zachary Nickerson

    Updated Feb 2025 to use "with" so that zip files properly close out
    and fix error handling for Windows filepath character length limits

    """

    # Error handling on inputs
    if zippath[-4:] in [".zip", ".ZIP"]:
        outpath = os.path.dirname(zippath)
        level = "all"
    else:
        outpath = zippath
        level = "in"

    if level == "all":
        with zipfile.ZipFile(zippath, "r") as zip_ref:
            tl = zip_ref.namelist()
            # print('tl:',tl)
            # Construct full paths as they will be after extraction
            # this is used in stack_by_table, eg. # litterlst = nu.stack_by_table("./testdata/NEON_litterfall.zip", savepath="envt")
            full_extracted_paths = [
                os.path.join(os.path.abspath(zippath).replace(".zip", ""), zipname)
                for zipname in tl
            ]
            # print('full_extracted_paths:',full_extracted_paths)
            # print('len(full_extracted_paths):',[len(x) for x in full_extracted_paths])

            # Error handling for filepath character length limitations on Windows
            # This is a warning
            if platform.system() == "Windows" and any(
                len(x) > 260 for x in full_extracted_paths
            ):
                longest_path = max(full_extracted_paths, key=len)
                warnings.warn(
                    "Filepaths on Windows are limited to 260 characters. "
                    "Attempting to extract a filepath that is > 260 characters long. "
                    "Move your working or savepath directory closer to the root directory or enable "
                    "long path support in Windows."
                )

            try:
                # Unzip file and get list of zips within (this will still fail if paths are too long on Windows)
                zip_ref.extractall(path=outpath)
                zps = glob.glob(zippath[:-4] + "/*.zip")
                # If there are more zips within parent zip file, unzip those as well
                # this might be deprecated (shouldn't happen any  more). level as an input might also be deprecated
                # leaving in in case anything changes in tabular data
                if len(zps) > 0:
                    print(
                        "Multiple zip files are contained within the parent zip file. Need an example to properly code this up.\n"
                    )
            except FileNotFoundError as e:
                print(e)
                raise OSError(
                    "Filepaths on Windows are limited to 260 characters. "
                    f"Attempting to extract a filepath that is {len(longest_path)} characters long. "
                    "Move your working or savepath directory closer to the root directory or enable "
                    "long path support in Windows."
                )

    if level == "in":
        zps = glob.glob(outpath + "/*.zip")

        for i in range(0, len(zps)):
            with zipfile.ZipFile(zps[i], "r") as zip_refi:
                tl = zip_refi.namelist()

                # Construct full paths as they will be after extraction
                full_extracted_paths = [
                    os.path.join(zps[i].replace(".zip", ""), zipname) for zipname in tl
                ]
                longest_path = max(full_extracted_paths, key=len)
                # print('full extracted paths:',full_extracted_paths)
                # print('len(full_extracted_paths):',[len(x) for x in full_extracted_paths])

                # Warning for filepath character length limitations on Windows
                if platform.system() == "Windows" and any(
                    len(x) > 260 for x in full_extracted_paths
                ):
                    warnings.warn(
                        "Filepaths on Windows are limited to 260 characters. "
                        "Attempting to extract a filepath that is > 260 characters long. "
                        "Move your working or savepath directory closer to the root directory "
                        "or enable long path support in Windows.",
                        UserWarning,
                    )

                try:
                    outpathi = zps[i][:-4]
                    zip_refi.extractall(path=outpathi)
                except FileNotFoundError as e:
                    raise OSError(
                        "ERROR: Filepaths on Windows are limited to 260 characters. "
                        f"Attempting to extract a filepath that is {len(longest_path)} characters long. "
                        "Move your working or savepath directory closer to the root directory or enable "
                        "long path support in Windows."
                    )
                    print(e)
            os.remove(zps[i])

    return None


def find_datatables(folder, f_names=True):
    """

    Find data tables

    Parameters
    --------
    folder: The filepath location of the unzipped NEON download package folder.
    f_names: Full names - if true, then return the full file names including enclosing folders - if false, return only the file names

    Return
    --------
    A data frame of file names

    Example
    --------
    filenames = find_datatables(folder=folder, f_names=False)
    see stack_data_files_parallel

    Created on Wed Apr 17 2024

    @author: Zachary Nickerson
    """

    # get all .csv files in the folder and its subfolders
    csv_files = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)

    # if fnames is True, return the full file paths; otherwise, return the base names
    if f_names:
        return csv_files
    else:
        return [os.path.basename(f) for f in csv_files]


def get_recent_publication(filepaths):
    """

    Returns the most recent files for those that do not need stacking

    Parameters
    --------
    filepaths: List of file paths

    Return
    --------
    The filepath of the file with the most recent publication date

    Created on Wed Apr 17 2024

    @author: Zachary Nickerson
    """

    # extract the publication dates from the file paths
    pub_dates = [
        re.search(r"20\d{2}\d{2}\d{2}", os.path.basename(f)) for f in filepaths
    ]
    pub_dates = [m.group(0) for m in pub_dates if m is not None]

    # get the most recent publication date
    recent_pub_date = max(pub_dates)

    # get the file paths that include the most recent publication date
    recent_files = [f for f in filepaths if recent_pub_date in os.path.basename(f)]

    return recent_files


def string_schema(v):
    """

    Assign all variables to string. Fallback option if table read with true schema fails

    Parameters
    --------
    v: A pandas table containing variable definitions

    Return
    --------
    A pyarrow schema for all string data types based on the variable names

    Created on Oct 29 2024

    @author: Claire Lunch
    """

    for i in range(0, len(v)):
        nm = v.fieldName[i]
        if i == 0:
            vschema = pa.schema([(nm, pa.string())])
        else:
            nfield = pa.field(nm, pa.string())
            vschema = vschema.append(nfield)

    return vschema


def unknown_string_schema(v):
    """

    Assign all variables to string. Fallback option if variables file can't be found

    Parameters
    --------
    v: A list of column names

    Return
    --------
    A pyarrow schema for all string data types based on the column names

    Created on Oct 29 2024

    @author: Claire Lunch
    """

    for i in range(0, len(v)):
        nm = v[i]
        if i == 0:
            vschema = pa.schema([(nm, pa.string())])
        else:
            nfield = pa.field(nm, pa.string())
            vschema = vschema.append(nfield)

    return vschema


def table_type_formats(flname):
    """

    Small helper function to define mapping of table types to file name formats

    Parameters
    --------
    flname: A file name, split into components

    Return
    --------
    A table type

    Created on 6 May 2024

    @author: Claire Lunch
    """

    flen = len(flname)
    if flen <= 6:
        return "lab"
    else:
        ymr = re.compile("[0-9]{4}-[0-9]{2}")
        if any([f for f in flname if ymr.search(f)]):
            return "site-date"
        else:
            return "site-all"


def find_table_types(datatables):
    """

    Find unique data tables and their types

    Parameters
    --------
    datatables: A list of data files

    Return
    --------
    An array of unique table names and their types

    Created on 3 May 2024

    @author: Claire Lunch
    """

    dt = [os.path.basename(d) for d in datatables]
    splitnames = [d.split(".") for d in dt]
    td = []
    for i in range(0, len(splitnames)):
        for j in range(2, len(splitnames[i])):
            s = splitnames[i][j]
            if not s == "sensor_positions" and not s == "science_review_flags":
                if "_" not in s:
                    continue
                else:
                    ss = str.replace(s, "_pub", "")
                    td.append(ss)

    if len(td) == 0:
        logging.info(
            "No data tables found, only metadata. Try downloading expanded package, and check availability on the NEON data portal."
        )
        return
    else:
        tn = list(set(td))

    tt = {}
    for k in range(0, len(tn)):
        tnk = tn[k]
        tnkr = re.compile("^" + tnk + "$|^" + tnk + "_pub$")
        tnknames = [trnn for trnn in splitnames for tr in trnn if tnkr.search(tr)]
        ttklist = list(map(table_type_formats, tnknames))
        ttk = list(set(ttklist))
        if len(ttk) > 1:
            raise ValueError(
                f"In files to be stacked, table {tnk} has been published under conflicting schedules. To avoid this problem, either work only with released data, or stack released and provisional data separately."
            )
            return
        else:
            tt[tnk] = ttk[0]

    return tt


def find_lab_names(flpths):
    """

    Small helper function to find names of labs from filenames of lap-specific files

    Parameters
    --------
    flpths: A list of file paths

    Return
    --------
    A list of unique lab names

    Created on 5 June 2024

    @author: Claire Lunch
    """

    flnms = [os.path.basename(f) for f in flpths]
    splitnames = [f.split(".") for f in flnms]
    labnames = [f[1] for f in splitnames]
    return list(set(labnames))


def find_sites(flpths):
    """

    Small helper function to find sites from file names

    Parameters
    --------
    flpths: A list of file paths

    Return
    --------
    A list of unique sites

    Created on 5 June 2024

    @author: Claire Lunch
    """

    flnms = [os.path.basename(f) for f in flpths]
    sr = re.compile("[.][A-Z]{4}[.]")
    sites = [sr.search(f).group(0) for f in flnms]
    sites = list(set(sites))
    sites = [re.sub(pattern="[.]", repl="", string=s) for s in sites]
    return sites


def remove_srf_dups(srftab):
    """

    Small helper function to remove duplicates and updated records from science review flag tables

    Parameters
    --------
    srftab: A pandas table of SRF records

    Return
    --------
    A de-duplicated pandas table of SRF records

    Created on 6 June 2024

    @author: Claire Lunch
    """

    idset = list(set(srftab["srfID"]))
    srfsublist = []
    for i in idset:
        srfsubi = srftab[srftab.srfID == i]
        indi = srfsubi.idxmax()["lastUpdateDateTime"]
        srfsublist.append(srfsubi.loc[indi])
    srfsub = pd.DataFrame(srfsublist)
    return srfsub


def align_sp_cols(sptab):
    """

    Small helper function to align column names between old and new sensor positions tables

    Parameters
    --------
    sptab: A pandas table of sensor positions records

    Return
    --------
    A pandas table of sensor positions records with only the current column names

    Created on 16 Sept 2024

    @author: Claire Lunch
    """

    oldcols = {
        "name": "sensorLocationID",
        "description": "sensorLocationDescription",
        "start": "positionStartDateTime",
        "end": "positionEndDateTime",
        "referenceName": "referenceLocationID",
        "referenceDescription": "referenceLocationIDDescription",
        "referenceStart": "referenceLocationIDStartDateTime",
        "referenceEnd": "referenceLocationIDEndDateTime",
        "referenceLatitude": "locationReferenceLatitude",
        "referenceLongitude": "locationReferenceLongitude",
        "referenceElevation": "locationReferenceElevation",
    }
    for k in list(oldcols.keys()):
        if all(sptab[k].isna()):
            sptab.drop(columns=k, inplace=True)
        else:
            sptab[oldcols[k]] = sptab[oldcols[k]].fillna(sptab[k])
            sptab.drop(columns=k, inplace=True)

    return sptab


def sort_dat(pdata):
    """

    Small helper function to sort data rows by location and date

    Parameters
    --------
    pdata: A pandas table of data records

    Return
    --------
    A pandas table of data records sorted by location and date

    Created on 17 Oct 2024

    @author: Claire Lunch
    """

    # sort the table by site, then HOR/VER, then date, all ascending
    pcols = pdata.columns.to_list()
    datevar = None
    if "collectDate" in pcols:
        datevar = "collectDate"
    else:
        if "endDate" in pcols:
            datevar = "endDate"
        else:
            if "startDate" in pcols:
                datevar = "startDate"
            else:
                if "date" in pcols:
                    datevar = "date"
                else:
                    if "endDateTime" in pcols:
                        datevar = "endDateTime"
                    else:
                        if "startDateTime" in pcols:
                            datevar = "startDateTime"
                        else:
                            if "yearMonth" in pcols:
                                datevar = "yearMonth"
                            else:
                                if "year" in pcols:
                                    datevar = "year"
    if "horizontalPosition" not in pcols:
        try:
            if datevar is None:
                pdata.sort_values(
                    by=["siteID"], ascending=[True], inplace=True, ignore_index=True
                )
            else:
                pdata.sort_values(
                    by=["siteID", datevar],
                    ascending=[True, True],
                    inplace=True,
                    ignore_index=True,
                )
        except Exception:
            try:
                pdata.sort_values(
                    by=[datevar], ascending=[True], inplace=True, ignore_index=True
                )
            except Exception:
                pass

    else:
        try:
            if datevar is None:
                pdata.sort_values(
                    by=["siteID", "horizontalPosition", "verticalPosition"],
                    ascending=[True, True, True],
                    inplace=True,
                    ignore_index=True,
                )
            else:
                pdata.sort_values(
                    by=["siteID", "horizontalPosition", "verticalPosition", datevar],
                    ascending=[True, True, True, True],
                    inplace=True,
                    ignore_index=True,
                )
        except Exception:
            pass

    return pdata


def stack_frame_files(framefiles, dpid, seqtyp=None, cloud_mode=False):
    """

    Helper function to stack "data frame" files. These files do not go through the normal
    publication process, they are stored and published as a fixed unit. NEON uses this
    workflow for very large tabular files that can't be handled by the standard OS
    data pipeline.

    Parameters
    --------
    framefiles: A list of filepaths pointing to the data frame files
    dpid: The data product identifier of the product being stacked
    seqtyp: For microbe community data, sequence type 16S or ITS
    cloud_mode: Use cloud mode to transfer files cloud-to-cloud? Defaults to False.

    Return
    --------
    A stacked pandas table

    Created on 24 Oct 2024

    @author: Claire Lunch
    """

    # no variables files for these, use custom files in package resources
    frame_file_file = (
        importlib_resources.files(__resources__) / "frame_file_variables.csv"
    )
    frame_file_variables = pd.read_csv(frame_file_file, index_col=None)
    # v = pd.concat([v, frame_file_variables], ignore_index=True)

    fdict = {
        "DP1.30012.001": "FSP",
        "DP1.10081.001": "MCC",
        "DP1.20086.001": "MCC",
        "DP1.20141.001": "MCC",
        "DP1.20190.001": "REA",
        "DP1.20193.001": "REA",
        "DP1.10081.002": "MCT",
        "DP1.20086.002": "MCT",
        "DP1.20141.002": "MCT",
        "DP4.00132.001": "BAT",
    }

    fvars = pa.Table.from_pandas(frame_file_variables)
    ftab = fvars.filter(pa.compute.field("table") == fdict[dpid])

    fpkgvar = ftab.to_pandas()
    fschema = get_variables(fpkgvar)

    if cloud_mode:
        gcs = fs.GcsFileSystem(anonymous=True)
        framebuckets = [
            re.sub(pattern="https://storage.neonscience.org/", repl="", string=b)
            for b in framefiles
        ]
        fdat = dataset.dataset(
            source=framebuckets, filesystem=gcs, format="csv", schema=fschema
        )
    else:
        fdat = dataset.dataset(source=framefiles, format="csv", schema=fschema)

    try:
        fdattab = fdat.to_table()
    except Exception:
        stringschema = unknown_string_schema(fdat.head(num_rows=0).column_names)
        if cloud_mode:
            fdat = dataset.dataset(
                source=framebuckets, filesystem=gcs, format="csv", schema=stringschema
            )
        else:
            fdat = dataset.dataset(source=framefiles, format="csv", schema=stringschema)
        fdattab = fdat.to_table()
        logging.info(
            "Large file schema did not match expectations; all variable types set to string."
        )

    fpdat = fdattab.to_pandas()

    nm = "per_sample"

    if dpid == "DP1.20190.001":
        nm = "rea_conductivityRawData"
    elif dpid == "DP1.20193.001":
        nm = "sbd_conductivityRawData"
    elif dpid == "DP1.30012.001":
        nm = "fsp_rawSpectra"
    elif dpid == "DP4.00132.001":
        nm = "bat_processedSonarFile"
    elif dpid == "DP1.10081.001":
        nm = f"mcc_soilPerSampleTaxonomy_{seqtyp}"
    elif dpid == "DP1.20086.001":
        nm = f"mcc_benthicPerSampleTaxonomy_{seqtyp}"
    elif dpid == "DP1.20141.001":
        nm = f"mcc_surfaceWaterPerSampleTaxonomy_{seqtyp}"
    elif dpid == "DP1.10081.002":
        nm = f"mct_soilPerSampleTaxonomy_{seqtyp}"
    elif dpid == "DP1.20086.002":
        nm = f"mct_benthicPerSampleTaxonomy_{seqtyp}"
    elif dpid == "DP1.20141.002":
        nm = f"mct_surfaceWaterPerSampleTaxonomy_{seqtyp}"

    return {"frmdat": fpdat, "frmnm": nm}


def format_readme(readmetab, tables):
    """

    Remove site-specific information from the most recently published readme file in the download, and add generic information about the neonutilities download.

    Parameters
    --------
    readmetab:  A readme document as a pandas table
    tables: A list of data tables included in download

    Return
    --------
    A modified readme file formatted to remove site-specific information and include standard information about the neonutilities download

    Created on Fri Jul 12 2024

    @author: Zachary Nickerson
    """

    rd = readmetab
    if len(tables) > 0:
        # replace query specific text
        dpackind = rd[0].str.contains("CONTENTS").idxmax()
        rd.loc[
            dpackind + 2, 0
        ] = f"This data product contains up to {len(tables)} data tables:"
        rd.loc[dpackind + 3 : dpackind + 3 + len(tables) - 1, 0] = tables
        rd.loc[
            dpackind + 3 + len(tables), 0
        ] = "If data are unavailable for the particular sites and dates queried, some tables may be absent."
        qind = rd[0].str.contains("QUERY").idxmax()
        downpackind = rd[0].str.contains("Basic download package").idxmax()

        # Remove specific rows
        remove_indices = (
            list(range(qind, dpackind))
            + list(range(dpackind + 4 + len(tables), downpackind))
            + rd.index[rd[0].str.contains("Date-Time")].tolist()
        )
        remove_indices = [index for index in remove_indices if index in list(rd.index)]
        rd = rd.drop(remove_indices)

        # add disclaimer
        disclaimer = pd.DataFrame(
            {
                0: [
                    "###################################",
                    "########### Disclaimer ############",
                    "This is the most recent readme publication based on all site-date combinations used during stackByTable.",
                    "Information specific to the query, including sites and dates, has been removed. The remaining content reflects general metadata for the data product.",
                    "##################################",
                ]
            }
        )
        rd = pd.concat([disclaimer, rd], ignore_index=True)
    return rd


def stack_data_files_parallel(folder, 
                              package, 
                              dpid, 
                              progress=True, 
                              cloud_mode=False,
                              datasetq=False):
    """

    Join data files in a unzipped NEON data package by table type

    Parameters
    --------
    folder: The filepath location of the unzipped NEON download package folder.
    package: basic or expanded data package
    dpid: Data product ID of product to stack.
    progress: Should a progress bar be displayed?
    cloud_mode: Use cloud mode to transfer files cloud-to-cloud? If used, stack_by_table() expects a list of file urls as input. Defaults to False.
    datasetq: Return an arrow dataset for a single table? Defaults to False.

    Return
    --------
    One file for each table type is created and written.

    Created on Tue Apr 2 2024

    @author: Zachary Nickerson
    """

    releases = []

    dpnum = dpid[4:9]

    if cloud_mode:
        filenames = [os.path.basename(f) for f in folder[0]]
        filepaths = folder[0]
        gcs = fs.GcsFileSystem(anonymous=True)
    else:
        # Get filenames without full path
        filenames = find_datatables(folder=folder, f_names=False)

        # Get filenames with full path
        filepaths = find_datatables(folder=folder, f_names=True)

    # dictionary for outputs
    stacklist = {}

    # handle per-sample (data frame) tables separately
    if (
        dpid
        in [
            "DP1.30012.001",
            "DP1.10081.001",
            "DP1.20086.001",
            "DP1.20141.001",
            "DP1.20190.001",
            "DP1.20193.001",
            "DP1.10081.002",
            "DP1.20086.002",
            "DP1.20141.002",
            "DP4.00132.001",
        ]
        and len([f for f in filenames if not f.startswith("NEON.")]) > 0
    ):
        framefiles = [
            f for f in filepaths if not os.path.basename(f).startswith("NEON.")
        ]
        filepaths = [f for f in filepaths if os.path.basename(f).startswith("NEON.")]
        filenames = [f for f in filenames if os.path.basename(f).startswith("NEON.")]

        # stack frame files
        if progress:
            logging.info(
               f"Stacking NEON {dpid} per-sample files. These files may be very large; download data in smaller subsets if performance problems are encountered.\n"
            )

        # subset microbe community data by taxonomic group
        # and stack both sets
        if dpid in [
            "DP1.10081.001",
            "DP1.20086.001",
            "DP1.20141.001",
            "DP1.10081.002",
            "DP1.20086.002",
            "DP1.20141.002",
        ]:
            bacteriafiles = [b for b in framefiles if re.search("[_]16S[_]", b)]
            fungifiles = [b for b in framefiles if re.search("[_]ITS[_]", b)]

            if len(bacteriafiles) > 0:
                fpdat16 = stack_frame_files(
                    bacteriafiles, dpid=dpid, seqtyp="16S", cloud_mode=cloud_mode
                )
                stacklist[fpdat16["frmnm"]] = fpdat16["frmdat"]

            if len(fungifiles) > 0:
                fpdatIT = stack_frame_files(
                    fungifiles, dpid=dpid, seqtyp="ITS", cloud_mode=cloud_mode
                )
                stacklist[fpdatIT["frmnm"]] = fpdatIT["frmdat"]

        else:
            fpdat = stack_frame_files(
                framefiles, dpid=dpid, seqtyp=None, cloud_mode=cloud_mode
            )
            stacklist[fpdat["frmnm"]] = fpdat["frmdat"]

    # make a dictionary, where filenames are the keys to the filepath values
    filelist = dict(zip(filenames, filepaths))

    datafls = filelist

    # if there are no datafiles, exit
    if len(datafls) == 0:
        print(f"No NEON {dpid} data files are present in specified file path.\n")
        return None

    # if there is one or more than one file, stack files

    # get table types
    table_types = find_table_types(filenames)
    if any(re.search("sensor_positions", path) for path in filepaths):
        table_types["sensor_positions"] = "site-all"
    if any(re.search("science_review_flags", path) for path in filepaths):
        table_types["science_review_flags"] = "site-date"
    tables = list(table_types.keys())

    # metadata files
    # get variables and validation files using the most recent publication date
    if any(re.search("variables.20", path) for path in filepaths):
        varpath = get_recent_publication(
            [path for path in filepaths if "variables.20" in path]
        )[0]
        if cloud_mode:
            vp = dataset.dataset(
                source=re.sub("https://storage.googleapis.com/", "", varpath),
                filesystem=gcs,
                format="csv",
                schema=varschema,
            )
            va = vp.to_table()
            v = va.to_pandas()
        else:
            v = pd.read_csv(varpath, sep=",")

        # if science review flags are present but missing from variables file, add variables
        if "science_review_flags" not in list(v["table"]):
            if any("science_review_flags" in path for path in filepaths):
                science_review_file = (
                    importlib_resources.files(__resources__)
                    / "science_review_variables.csv"
                )
                science_review_variables = pd.read_csv(
                    science_review_file, index_col=None
                )
                v = pd.concat([v, science_review_variables], ignore_index=True)

        # if sensor positions are present but missing from variables file, add variables
        if any("sensor_positions" in path for path in filepaths):
            sensor_positions_map = (
                importlib_resources.files(__resources__)
                / "sensor_positions_variables_mapping.csv"
            )
            sensor_positions_internal_variables = pd.read_csv(
                sensor_positions_map, index_col=None
            )
            if "sensor_positions" not in list(v["table"]):
                sensor_positions_file = (
                    importlib_resources.files(__resources__)
                    / "sensor_positions_variables.csv"
                )
                sensor_positions_variables = pd.read_csv(
                    sensor_positions_file, index_col=None
                )
                v = pd.concat([v, sensor_positions_variables], ignore_index=True)

        # save the variables file
        vlist = {k: v for k, v in v.groupby("table")}
        stacklist[f"variables_{dpnum}"] = v

    # get validation file
    if any(re.search("validation", path) for path in filepaths):
        valpath = get_recent_publication(
            [path for path in filepaths if "validation" in path]
        )[0]
        if cloud_mode:
            vp = dataset.dataset(
                source=re.sub("https://storage.googleapis.com/", "", valpath),
                filesystem=gcs,
                format="csv",
                schema=None,
            )
            va = vp.to_table()
            val = va.to_pandas()
        else:
            val = pd.read_csv(valpath, sep=",")
        stacklist[f"validation_{dpnum}"] = val

    # get categoricalCodes file
    if any(re.search("categoricalCodes", path) for path in filepaths):
        ccpath = get_recent_publication(
            [path for path in filepaths if "categoricalCodes" in path]
        )[0]
        if cloud_mode:
            cp = dataset.dataset(
                source=re.sub("https://storage.googleapis.com/", "", ccpath),
                filesystem=gcs,
                format="csv",
                schema=None,
            )
            ca = cp.to_table()
            cc = ca.to_pandas()
        else:
            cc = pd.read_csv(ccpath, sep=",")
        stacklist[f"categoricalCodes_{dpnum}"] = cc

    # get readme file
    if cloud_mode:
        readmefiles = filepaths
    else:
        readmefiles = glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
    if any(re.search("readme.20", path) for path in readmefiles):
        readmepath = get_recent_publication(
            [path for path in readmefiles if "readme.20" in path]
        )[0]
        rd = None
        if cloud_mode:
            try:
                rd = readme_url(readmepath)
            except Exception:
                pass
        else:
            try:
                rd = pd.read_table(readmepath, delimiter="\t", header=None)
            except Exception:
                pass
        if rd is None:
            pass
        else:
            rd = format_readme(rd, tables)
            # save the readme
            stacklist[f"readme_{dpnum}"] = rd

    # stack tables according to types
    if progress:
        logging.info("Stacking data files")
    arrowvars = pa.Table.from_pandas(stacklist[f"variables_{dpnum}"])
    for j in tqdm(tables, disable=not progress):
        # create schema from variables file, for only this table and package
        vtab = arrowvars.filter(pa.compute.field("table") == j)
        if len(vtab) == 0:
            vtab = arrowvars.filter(pa.compute.field("table") == j + "_pub")
        if j == "sensor_positions":
            vtab = pa.Table.from_pandas(sensor_positions_internal_variables)

        if package == "basic":
            vtabpkg = vtab.filter(pa.compute.field("downloadPkg") == "basic")
        else:
            vtabpkg = vtab

        tablepkgvar = vtabpkg.to_pandas()
        if len(tablepkgvar) == 0:
            # set to string if variables file can't be found
            tableschema = None
            logging.info(
                f"NEON {dpid} variables file not found for table {j}. Data types will be inferred if possible."
            )
        else:
            tableschema = get_variables(tablepkgvar)

        # subset the list of files to the relevant table
        tabler = re.compile("[.]" + j + "[.]|[.]" + j + "_pub[.]")
        tablepaths = [f for f in filepaths if tabler.search(f)]

        # subset the list of files for lab-specific tables:
        # get the most recent file from each lab
        if table_types[j] == "lab":
            labs = find_lab_names(tablepaths)
            labrecent = list()
            for k in labs:
                labr = re.compile(k)
                labpaths = [f for f in tablepaths if labr.search(f)]
                labrecent.append(get_recent_publication(labpaths)[0])
            tablepaths = labrecent

        # subset the list of files for site-all tables:
        # get the most recent file from each site
        if table_types[j] == "site-all":
            sites = find_sites(tablepaths)
            siterecent = list()
            for k in sites:
                sr = re.compile(k)
                sitepaths = [f for f in tablepaths if sr.search(f)]
                siterecent.append(get_recent_publication(sitepaths)[0])
            tablepaths = siterecent

        # read data and append file names
        if cloud_mode:
            tablebuckets = [
                re.sub(pattern="https://storage.googleapis.com/", repl="", string=b)
                for b in tablepaths
            ]
            dat = dataset.dataset(
                source=tablebuckets, filesystem=gcs, format="csv", schema=tableschema
            )
        else:
            dat = dataset.dataset(source=tablepaths, format="csv", schema=tableschema)

        if tableschema is None:
            cols = dat.head(num_rows=0).column_names
        else:
            cols = tableschema.names
        cols.append("__filename")

        # attempt to stack to table. if it fails, stack as all string fields and warn
        stringset = False
        try:
            dattab = dat.to_table(columns=cols)
        except Exception:
            try:
                if tableschema is None:
                    stringschema = unknown_string_schema(
                        dat.head(num_rows=0).column_names
                    )
                else:
                    stringschema = string_schema(tablepkgvar)
                if cloud_mode:
                    dat = dataset.dataset(
                        source=tablebuckets,
                        filesystem=gcs,
                        format="csv",
                        schema=stringschema,
                    )
                else:
                    dat = dataset.dataset(
                        source=tablepaths, format="csv", schema=stringschema
                    )
                dattab = dat.to_table(columns=cols)
                logging.info(
                    f"Table {j} schema did not match data; all variable types set to string. Data type casting will be attempted after stacking step."
                )
                stringset = True
            except Exception:
                logging.info(
                    f"Failed to stack table {j}. Check input data and variables file."
                )
                continue
            
        # for dataset query, done here
        if datasetq:
            return dat
        else:
            pdat = dattab.to_pandas()

        if stringset:
            try:
                pdat = cast_table_neon(pdat, tablepkgvar)
            except Exception:
                logging.info(
                    f"Data type casting failed for table {j}. Variable types set to string."
                )

        # append publication date
        pubr = re.compile("20[0-9]{6}T[0-9]{6}Z")
        pubval = [pubr.search(os.path.basename(p)).group(0) for p in pdat["__filename"]]
        pdat = pdat.assign(publicationDate=pubval)

        # append release tag
        if cloud_mode:
            pdat["release"] = pdat["__filename"].map(folder[1])
            releases.append(list(set(folder[1].values())))
        else:
            pubrelr = re.compile("20[0-9]{6}T[0-9]{6}Z\\..*\\/")
            pubrelval = [pubrelr.search(p).group(0) for p in pdat["__filename"]]
            relval = [re.sub(".*\\.", "", s) for s in pubrelval]
            relval = [re.sub("\\/", "", s) for s in relval]
            pdat = pdat.assign(release=relval)
            releases.append(list(set(relval)))

        # append fields to variables file
        if f"variables_{dpnum}" in stacklist.keys():
            added_fields_file = (
                importlib_resources.files(__resources__) / "added_fields.csv"
            )
            added_fields = pd.read_csv(added_fields_file, index_col=None)
            added_fields_all = added_fields[-2:]
            added_fields_all.insert(0, "table", j)
            try:
                vlist[j] = pd.concat([vlist[j], added_fields_all], ignore_index=True)
            except Exception:
                pass

        # for IS products, append domainID, siteID, HOR, VER
        if "siteID" not in pdat.columns.to_list() and not table_types[j] == "lab":
            dr = re.compile("D[0-2]{1}[0-9]{1}")
            domval = [dr.search(d).group(0) for d in pdat["__filename"]]
            pdat.insert(0, "domainID", domval)

            sr = re.compile("D[0-9]{2}[.][A-Z]{4}[.]")
            sitel = [sr.search(s).group(0) for s in pdat["__filename"]]
            siteval = [
                re.sub(pattern="D[0-9]{2}[.]|[.]", repl="", string=s) for s in sitel
            ]
            pdat.insert(1, "siteID", siteval)

            if j != "sensor_positions":
                locr = re.compile("[.][0-9]{3}[.][0-9]{3}[.][0-9]{3}[.][0-9]{3}[.]|[.][0-9]{3}[.][0-9]{3}[.][0-9]{3}[.][0-9]{2}[A-Z]{1}[.]")
                indtemp = [locr.search(l) for l in pdat["__filename"]]
                if None in indtemp:
                    pdat = sort_dat(pdat)
                else:
                    indxs = [lt.group(0) for lt in indtemp]
                    hor = [indx[5:8] for indx in indxs]
                    ver = [indx[9:12] for indx in indxs]
                    pdat.insert(2, "horizontalPosition", hor)
                    pdat.insert(3, "verticalPosition", ver)

                    # sort table rows
                    pdat = sort_dat(pdat)

                # append fields to variables file
                if f"variables_{dpnum}" in stacklist.keys():
                    added_fields_IS = added_fields[0:4]
                    added_fields_IS.insert(0, "table", j)
                    try:
                        vlist[j] = pd.concat(
                            [added_fields_IS, vlist[j]], ignore_index=True
                        )
                    except Exception:
                        pass

        else:
            # for OS tables, sort by site and date
            pdat = sort_dat(pdat)

        # for SRF files, remove duplicates and modified records
        if j == "science_review_flags":
            pdat = remove_srf_dups(pdat)

        # for sensor position files, align column names
        if j == "sensor_positions":
            pdat = align_sp_cols(pdat)

        # remove filename column
        pdat = pdat.drop(columns=["__filename"])
        
        # add table to list
        if j == "science_review_flags" or j == "sensor_positions":
            stacklist[f"{j}_{dpnum}"] = pdat
        else:
            stacklist[j] = pdat

    # final variables file
    stacklist[f"variables_{dpnum}"] = pd.concat(vlist, ignore_index=True)

    # get issue log table
    # token omitted here since it's not otherwise used in stacking functions
    # consider a runLocal option, like in R stackEddy()
    try:
        stacklist[f"issueLog_{dpnum}"] = get_issue_log(dpid=dpid, token=None)
    except Exception:
        pass

    # get relevant citation(s)
    try:
        releases = sum(releases, [])
        releases = list(set(releases))
        if "PROVISIONAL" in releases:
            try:
                stacklist[f"citation_{dpnum}_PROVISIONAL"] = get_citation(
                    dpid=dpid, release="PROVISIONAL"
                )
            except Exception:
                pass
        relr = re.compile("RELEASE-20[0-9]{2}")
        rs = [relr.search(r).group(0) for r in releases if relr.search(r)]
        if len(rs) == 1:
            stacklist[f"citation_{dpnum}_{rs[0]}"] = get_citation(
                dpid=dpid, release=rs[0]
            )
        if len(rs) > 1:
            logging.info(
                f"Multiple NEON data releases were stacked together for {dpid}. This is not appropriate, check your input data."
            )
    except Exception:
        pass

    return stacklist


def stack_by_table(
    filepath, savepath=None, 
    save_unzipped_files=False, 
    progress=True, 
    cloud_mode=False,
    datasetq=False,
):
    """

    Join data files in a zipped or unzipped NEON data package by table type.
    This function can be used on a zip file downloaded from the NEON data portal or
    on a set of files downloaded by zips_by_product().

    Parameters
    ------------------
    filepath: str
        The location of the zip file or downloaded files.

    savepath: str, optional
        The location to save the output files to. If omitted, output files will be
        saved in the same location as the input file.

    save_unzipped_files: bool, optional
        Should the unzipped monthly data folders be retained? Defaults to False.

    progress: bool, optional
        Should the function display progress bars as it runs? Defaults to True.

    cloud_mode: bool, optional
        Use cloud mode to transfer files cloud-to-cloud? If used, stack_by_table()
        expects a list of file urls as input. Defaults to False; in general this
        option should be used via load_by_product(), in which stack_by_table() is a
        helper function.
        
    datasetq: bool, optional
        Should the function return an arrow dataset for a single table, instead 
        of a set of stacked tables? Defaults to False.


    Return
    -------------------
    All files are unzipped and one file for each table type is created and written.
    If savepath="envt" is specified, output is a named list of tables; otherwise,
    function output is null and files are saved to the location specified.

    Example
    ------------------
    To stack PAR data (DP1.00024.001) downloaded from the NEON data portal

    >>> pardat = stack_by_table("/filepath/NEON_par.zip")

    Notes
    --------

    Windows Path Length Limitations:
    When using this function, you may encounter path length limitations  on Windows systems.
    Windows has a default maximum path length of 260 characters, which can cause download and unzip
    functions to fail if this limit is exceeded. If the file path exceeds 260 characters on a Windows system,
    the package will issue a warning.You can choose to ignore or filter these warnings using Python's warnings
    module if you prefer not to see them. If the function is unable to unzip a folder due to path length
    limitations, an OSError will be raised.

    Created on Tue Mar 5 2024

    @author: Zachary Nickerson
    """

    # determine contents of filepath and unzip as needed
    if cloud_mode:
        files = filepath[0]
    else:
        # Is the filepath input a zip file or an unzipped file?
        if filepath[-4:] in [".zip", ".ZIP"]:
            folder = False
        else:
            folder = True

        # Get list of files nested (and/or zipped) in filepath
        if not folder:
            zip_ref = zipfile.ZipFile(filepath, "r")
            files = zip_ref.namelist()
        else:
            files = glob.glob(filepath + "/**", recursive=True)

    # Error handling if there are no standardized NEON Portal data tables in the list of files
    if not any(re.search(r"NEON.D[0-9]{2}.[A-Z]{4}.", x) for x in files):
        logging.info("NEON data files are not present in the specified filepath.")
        return

    # Determine dpid
    # this regexpr allows for REV = .001 or .002
    dpid_listlist = []
    for f in range(len(files)):
        dpid_listlist.append(
            re.findall(re.compile("DP[1-4][.][0-9]{5}[.]00[1-2]{1}"), files[f])
        )
    dpid = [x for dpid_list in dpid_listlist for x in dpid_list]
    dpid = list(set(dpid))
    if not len(dpid) == 1:
        logging.info(
            "NEON Data product ID could not be determined. Check that filepath contains data files, from a single NEON data product."
        )
        return
    else:
        dpid = dpid[0]

    # Determine download package
    package_listlist = []
    for f in range(len(files)):
        package_listlist.append(re.findall(re.compile("basic|expanded"), files[f]))
    package = [x for package_list in package_listlist for x in package_list]
    package = list(set(package))
    if "expanded" in package:
        package = "expanded"
    else:
        package = "basic"

    # Error message for AOP data
    if dpid[4] == "3" and not dpid == "DP1.30012.001":
        logging.info(
            "This is a NEON AOP data product, files cannot be stacked. Use by_file_aop() or by_tile_aop() to download data."
        )
        return

    # Error messafe for SAE data
    if dpid == "DP4.00200.001":
        logging.info(
            "This NEON eddy covariance data product is in HDF5 format. Stack using the stackEddy() function in the R package version of neonUtilities."
        )
        return

    # Exceptions for digital hemispheric photos
    if dpid == "DP1.10017.001" and package == "expanded":
        save_unzipped_files = True
        logging.info(
            "Note: NEON Digital hemispheric photos (in NEF format) cannot be stacked; only the CSV metadata files will be stacked."
        )

    # Warning about all sensor soil data
    # Test and modify the file length for the alert, this should be a lot better with arrow
    if dpid in ["DP1.00094.001", "DP1.00041.001"] and len(files) > 24:
        logging.info(
            "Warning! Attempting to stack NEON soil sensor data. Note that due to the number of soil sensors at each site, data volume is very high for these data. Consider dividing data processing into chunks and/or using a high-performance system."
        )

    # If all checks pass, unzip and stack files
    if cloud_mode:
        stackedlist = stack_data_files_parallel(
            folder=filepath,
            package=package,
            dpid=dpid,
            progress=progress,
            cloud_mode=True,
            datasetq=datasetq,
        )

    else:
        # If the filepath is a zip file
        if not folder:
            unzip_zipfile(zippath=filepath)
            stackpath = filepath[:-4]

        # If the filepath is a directory
        if folder:
            if any(".zip" in file for file in files):
                unzip_zipfile(zippath=filepath)
            stackpath = filepath

        # Stack the files
        stackedlist = stack_data_files_parallel(
            folder=stackpath, package=package, dpid=dpid, 
            progress=progress, cloud_mode=False, datasetq=False,
        )

        # delete input files
        if not save_unzipped_files:
            ufl = glob.glob(stackpath + "/**.*/*", recursive=True)
            for fl in ufl:
                try:
                    os.remove(fl)
                except Exception:
                    pass
            dirlist = glob.glob(stackpath + "/*", recursive=True)
            for d in dirlist:
                try:
                    os.rmdir(d)
                except Exception:
                    pass
            if os.listdir(stackpath) == [".DS_Store"]:
                os.remove(stackpath + "/.DS_Store")
            if os.listdir(stackpath) == []:
                os.rmdir(stackpath)

    # Check if stackedlist is None (eg. if there is a failure in downloading the data due to long paths on windows)
    if stackedlist is None:
        logging.info("ERROR! Stacking failed due to previous errors.")
        return None
    
    # if returning an arrow dataset, done here
    if datasetq:
        return stackedlist

    # sort the dictionary of tables
    mk = list(stackedlist.keys())
    mk.sort()
    stackedlistsort = {i: stackedlist[i] for i in mk}

    # write files to path if requested
    if savepath == "envt":
        return stackedlistsort
    else:
        if savepath is None:
            stacked_files_dir = os.path.join(stackpath, "stackedFiles")

        elif savepath != "envt":
            stacked_files_dir = os.path.join(savepath, "stackedFiles")

        if not os.path.exists(stacked_files_dir):
            os.makedirs(stacked_files_dir)

        for k in stackedlistsort.keys():
            tk = stackedlistsort[k]
            if "citation" in k:
                with open(
                    f"{stacked_files_dir}/{k}.txt", mode="w+", encoding="utf-8"
                ) as f:
                    f.write(tk)
            else:
                if "readme" in k:
                    tk.to_csv(f"{stacked_files_dir}/{k}.txt", sep="\t", index=False)
                else:
                    # Write numeric data to up to 15 digits, rounding to
                    # the precision just below the maximum. This mimics the
                    # default behavior in R, to ensure the two languages'
                    # versions of neonUtilities match. For an interesting
                    # discussion of the issues around this, see
                    # https://github.com/pandas-dev/pandas/issues/16452
                    tk.to_csv(
                        f"{stacked_files_dir}/{k}.csv",
                        index=False,
                        float_format="%.15g",
                    )

        return None


def load_by_product(
    dpid,
    site="all",
    startdate=None,
    enddate=None,
    package="basic",
    release="current",
    timeindex="all",
    tabl="all",
    check_size=True,
    include_provisional=False,
    cloud_mode=False,
    progress=True,
    token=None,
):
    """
    This function downloads product-site-month data package files from NEON, unzips
    and stacks the data files, and loads to the environment.

    Parameters
    ----------------
    dpid: str
        Data product identifier in the form DP#.#####.###

    site: str
        Either the string 'all', or one or more 4-letter NEON site codes. Defaults to 'all'.

    startdate: str, optional
        Earliest date of data to download, in the form YYYY-MM

    enddate: str, optional
        Latest date of data to download, in the form YYYY-MM

    package: str, optional
        Download package to access, either basic or expanded. Defaults to 'basic'.

    release: str, optional
        Data release to download. Defaults to the most recent release.

    timeindex: str, optional
        Either 'all', or the time index of data to download, in minutes. Only applicable to sensor (IS) data. Defaults to 'all'.

    tabl: str, optional
        Either 'all', or the name of a single data table to download. Only applicable to observational (OS) data. Defaults to 'all'.

    check_size: bool, optional
        True or False, should the user approve the total file size before downloading? Defaults to True. When working in batch mode, or other non-interactive workflow, use check_size=False.

    include_provisional: bool, optional
        Should Provisional data be returned in the download? Defaults to False. See
        https://www.neonscience.org/data-samples/data-management/data-revisions-releases
        for details on the difference between provisional and released data.

    cloud_mode: bool, optional
        Use cloud mode to transfer files cloud-to-cloud? Should only be used if the destination location is in the cloud. Defaults to False.

    progress: bool, optional
        Should the function display progress bars as it runs? Defaults to True.

    token: str, optional
        User-specific API token from data.neonscience.org user account. See
        https://data.neonscience.org/data-api/rate-limiting/ for details about
        API rate limits and user tokens. If omitted, download uses the public rate limit.

    Return
    ---------------
    A dictionary of data and metadata tables, as pandas tables.

    Example
    ---------------
    Download water quality data from COMO (Como Creek) in 2018

    >>> wq = load_by_product(dpid="DP1.20288.001", site="COMO",
                             startdate="2018-01", enddate="2018-12",
                             token=None)

    Notes
    --------
    Windows Path Length Limitations:
    When using this function, you may encounter path length limitations  on Windows systems.
    Windows has a default maximum path length of 260 characters, which can cause download and unzip
    functions to fail if this limit is exceeded. If the file path exceeds 260 characters on a Windows system,
    the package will issue a warning.You can choose to ignore or filter these warnings using Python's warnings
    module if you prefer not to see them. If the function is unable to unzip a folder due to path length
    limitations, an OSError will be raised.

    Created on June 12 2024

    @author: Claire Lunch
    """

    savepath = os.getcwd()

    if cloud_mode:
        flist = zips_by_product(
            dpid=dpid,
            site=site,
            startdate=startdate,
            enddate=enddate,
            package=package,
            release=release,
            timeindex=timeindex,
            tabl=tabl,
            check_size=check_size,
            include_provisional=include_provisional,
            cloud_mode=True,
            progress=progress,
            token=token,
            savepath=savepath,
        )

        outlist = stack_by_table(
            filepath=flist,
            savepath="envt",
            cloud_mode=True,
            save_unzipped_files=False,
            progress=progress,
        )

    else:
        zips_by_product(
            dpid=dpid,
            site=site,
            startdate=startdate,
            enddate=enddate,
            package=package,
            release=release,
            timeindex=timeindex,
            tabl=tabl,
            check_size=check_size,
            include_provisional=include_provisional,
            progress=progress,
            token=token,
            savepath=savepath,
        )

        stackpath = savepath + "/filesToStack" + dpid[4:9] + "/"

        outlist = stack_by_table(
            filepath=stackpath,
            savepath="envt",
            save_unzipped_files=False,
            progress=progress,
        )

    return outlist


def dataset_query(
    dpid,
    site="all",
    startdate=None,
    enddate=None,
    package="basic",
    release="current",
    tabl=None,
    hor=None,
    ver=None,
    include_provisional=False,
    token=None,
):
    """
    This function uses the query endpoint of the NEON API to find the full 
    list of files for a given data product, release, site(s), and date range, 
    then turns them into an arrow dataset.

    Parameters
    ----------------
    dpid: str
        Data product identifier in the form DP#.#####.###

    site: str
        Either the string 'all', or one or more 4-letter NEON site codes. Defaults to 'all'.

    startdate: str, optional
        Earliest date of data to include in dataset, in the form YYYY-MM

    enddate: str, optional
        Latest date of data to include in dataset, in the form YYYY-MM

    package: str, optional
        Download package to access, either basic or expanded. Defaults to 'basic'.

    release: str, optional
        Data release to access. Defaults to the most recent release.

    tabl: str
        The name of a single data table to access.

    hor: str, optional
        The horizontal index of data to access. Only applicable to sensor (IS) data.

    ver: str, optional
        The vertical index of data to access. Only applicable to sensor (IS) data.

    include_provisional: bool, optional
        Should Provisional data be returned in the download? Defaults to False. See
        https://www.neonscience.org/data-samples/data-management/data-revisions-releases
        for details on the difference between provisional and released data.

    token: str, optional
        User-specific API token from data.neonscience.org user account. See
        https://data.neonscience.org/data-api/rate-limiting/ for details about
        API rate limits and user tokens. If omitted, download uses the public rate limit.

    Return
    ---------------
    An arrow dataset for the data requested.

    Example
    ---------------
    Create a dataset of mammal trap data from HARV (Harvard Forest) in 2018

    >>> mamds = dataset_query(dpid="DP1.10072.001", site="HARV",
                              tabl="mam_pertrapnight", package="expanded",
                              startdate="2018-01", enddate="2018-12",
                              token=None)

    Created on August 12 2025

    @author: Claire Lunch
    """

    # check that needed inputs are present and data product is tabular
    # consider turning opening checks in zips_by_product() into a function
    
    # query the /products endpoint for the product requested
    if release == "current" or release == "PROVISIONAL":
        prodreq = get_api(
            api_url="https://data.neonscience.org/api/v0/products/" + dpid, token=token
        )
    else:
        prodreq = get_api(
            api_url="https://data.neonscience.org/api/v0/products/"
            + dpid
            + "?release="
            + release,
            token=token,
        )
        
        if prodreq is None:
            if release == "LATEST":
                logging.info(
                    f"No data found for NEON data product {dpid}. LATEST data requested; check that token is valid for LATEST access."
                )
                return
            else:
                logging.info(
                    f"No data found for NEON data product {dpid}."
                )
                return

    avail = prodreq.json()
    pubtype = avail["data"]["productPublicationFormatType"]

    if pubtype=="AOP Data Product Type":
        raise ValueError(
            f"{dpid} is a remote sensing data product. Remote sensing data can't be queried using this function."
            )
        
    if tabl is None:
        raise ValueError(
            "Table name (tabl=) is a required input to this function."
            )
        
    if pubtype in ["TIS Data Product Type","AIS Data Product Type"]:
        if dpid in ["DP4.00200.001",
                   "DP1.00007.001","DP1.00010.001","DP1.00034.001","DP1.00035.001",
                   "DP1.00036.001","DP1.00037.001","DP1.00099.001","DP1.00100.001",
                   "DP2.00008.001","DP2.00009.001","DP2.00024.001","DP3.00008.001",
                   "DP3.00009.001","DP3.00010.001","DP4.00002.001","DP4.00007.001",
                   "DP4.00067.001","DP4.00137.001","DP4.00201.001","DP1.00030.001"]:
            raise ValueError(
                f"{dpid} is an eddy covariance data product and can't be queried using this function."
                )
        else:
            if bool(re.search(pattern="^ais[_]", string=tabl)):
                tabl = tabl
            else:
                if site=="all" or not isinstance(site, str):
                    raise ValueError(
                        f"{dpid} is a sensor data product and can only be queried at a single site using this function. If you need data at multiple sites, run multiple queries or use loadByProduct()."
                        )
                if hor is None or ver is None:
                    raise ValueError(
                        f"{dpid} is a sensor data product. hor and ver indices must be provided to use this query function."
                        )
                if len(hor)!=3 or len(ver)!=3 or not isinstance(hor, str) or not isinstance(ver, str):
                    raise ValueError(
                        "hor and ver must be 3-digit codes in string format, and must each be a single code. Only one sensor location can be queried."
                        )
    
    # get list of files
    flist = zips_by_product(
        dpid=dpid,
        site=site,
        startdate=startdate,
        enddate=enddate,
        package=package,
        release=release,
        tabl=tabl,
        check_size=False,
        include_provisional=include_provisional,
        cloud_mode=True,
        token=token,
        savepath=None,
    )
    
    # if IS data, subset by hor and ver
    # keep the variables file for schema creation
    if ver is not None:
        hvr = re.compile("[.]00[0-9]{1}[.]" + hor + "[.]" + ver + "[.][0-9]{2}[A-Z0-9]{1}[.]|variables")
        d0 = [r for r in flist[0] if hvr.search(r)]
        d1 = {r: flist[1][r] for r in flist[1] if hvr.search(r)}
    
        flist[0] = d0
        flist[1] = d1
    
    # send to stacking function to turn into dataset
    # stacking stops right before the dataset-to-table step
    outds = stack_by_table(
        filepath=flist,
        savepath="envt",
        cloud_mode=True,
        save_unzipped_files=False,
        progress=False,
        datasetq=True,
    )

    return outds