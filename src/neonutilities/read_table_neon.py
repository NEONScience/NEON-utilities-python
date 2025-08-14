#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pyarrow as pa
from pyarrow import dataset
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_variables(v):
    """

    Get correct data types for each field in a data table

    Parameters
    --------
    v: A pandas table containing variable definitions

    Return
    --------
    A pyarrow schema for data types based on the variables file

    Created on Wed Apr 17 2024

    @author: Zachary Nickerson
    """

    # function assumes variables are loaded as a pandas data frame.

    # create pyarrow schema by translating NEON data types to pyarrow types
    for i in range(0, len(v)):
        nm = v.fieldName[i]
        typ = pa.string()
        if v.dataType[i] == "real":
            typ = pa.float64()
        if v.dataType[i] in ["integer", "unsigned integer", "signed integer"]:
            typ = pa.int64()
        if v.dataType[i] in ["string", "uri"]:
            typ = pa.string()
        if v.dataType[i] == "dateTime":
            if v.pubFormat[i] in [
                "yyyy-MM-dd'T'HH:mm:ss'Z'(floor)",
                "yyyy-MM-dd'T'HH:mm:ss'Z'",
                "yyyy-MM-dd'T'HH:mm:ss'Z'(round)",
                "yyyy-MM-dd'T'HH:mm'Z'(floor)",
                "yyyy-MM-dd'T'HH:mm'Z'",
                "yyyy-MM-dd'T'HH:mm'Z'(round)",
                "yyyy-MM-dd'T'HH'Z'(floor)",
                "yyyy-MM-dd'T'HH'Z'",
                "yyyy-MM-dd'T'HH'Z'(round)"
            ]:
                typ = pa.timestamp("s", tz="UTC")
            else:
                if v.pubFormat[i] in ["yyyy-MM-dd(floor)", "yyyy-MM-dd"]:
                    typ = pa.date64()
                else:
                    if v.pubFormat[i] in ["yyyy(floor)", "yyyy(round)"]:
                        typ = pa.int64()
                    else:
                        typ = pa.string()
        if i == 0:
            vschema = pa.schema([(nm, typ)])
        else:
            nfield = pa.field(nm, typ)
            vschema = vschema.append(nfield)

    return vschema


def read_table_neon(data_file, var_file):
    """

    Read a NEON data table with correct data types for each variable.

    Parameters
    -------------------
    data_file: str
        Filepath to a data table to load.

    var_file: str
        Filepath to a variables file matching the data table.

    Return
    -------------------
    A data frame of a NEON data table, with column classes assigned by data type.

    Example
    -------------------
    >>> dattab = read_table_neon(data_file="path to data table file",
                                 var_file="path to variables file matching data table")

    Created on Thu Feb 1 2024

    @author: Zachary Nickerson
    """

    # Read in variables file and check type
    if isinstance(var_file, str):
        try:
            v = pd.read_csv(var_file)
        except Exception:
            logging.info(
                "Table read failed because var_file must be either a NEON variables table or a file path to a NEON variables table."
            )
            return

    # Check this is a valid variables file
    if any(x in ["category", "system", "stat"] for x in list(v.columns)):
        print(
            "var_file appears to match DP4.00200.001. Data wrangling for surface-atmosphere exchange data is currently only available in the R package version of neonUtilities."
        )
        return
    else:
        if not any(x in ["table", "fieldName", "dataType"] for x in list(v.columns)):
            logging.info(
                "var_file is not a variables file, or is missing critical values."
            )
            return

    # get field names from the data table without loading table
    tabcols = pd.read_csv(data_file, index_col=0, nrows=0).columns.tolist()

    # subset variables file to the relevant fields
    vtab = v[v["fieldName"].isin(tabcols)]
    vtabu = vtab.drop_duplicates(subset="fieldName", ignore_index=True)

    # generate schema
    tableschema = get_variables(vtabu)

    # Check that most fields have a corrsponding value in variables
    m = len(tabcols) - len(tableschema)
    if m == len(tabcols):
        logging.info(
            "Cannot correct data types because the variables file does not match the data file."
        )
        return
    if m > 4:
        logging.info(
            f"{m} fieldNames are present in data file but not in variables file. Data load may be affected; if possible, unknown fields are read as character strings."
        )

    # read data and append file names
    dat = dataset.dataset(source=data_file, format="csv", schema=tableschema)
    dattab = dat.to_table()
    pdat = dattab.to_pandas()

    return pdat


def date_convert(dates):
    """

    Convert a string to a date type, or, if possible date formats fail, keep as a string.

    Parameters
    --------
    dates: A date or datetime in string format

    Return
    --------
    The date in date format, or, if conversion fails, the original string.

    Example
    --------
    >>> d = date_convert("2023-08-01")

    Created on 2 May 2024

    @author: Claire Lunch
    """

    try:
        dout = pd.to_datetime(dates, format="ISO8601", utc=True)
    except Exception:
        try:
            dout = pd.to_datetime(dates, format="%Y-%m-%dT%H:%M:%S", utc=True)
        except Exception:
            try:
                dout = pd.to_datetime(dates, format="%Y-%m-%dT%H:%M", utc=True)
            except Exception:
                try:
                    dout = pd.to_datetime(dates, format="%Y-%m-%dT%H", utc=True)
                except Exception:
                    try:
                        dout = pd.to_datetime(dates, format="%Y-%m-%d", utc=True)
                    except Exception:
                        dout = dates
    return dout


def get_variables_pandas(v):
    """

    Get correct data types for each field in a data table

    Parameters
    --------
    v: A pandas table containing variable definitions

    Return
    --------
    A dictionary of field names and pandas data types based on the variables file

    Created on Oct 31 2024

    @author: Claire Lunch
    """

    dtdict = {}
    vf = v["fieldName"]
    for i in range(0, len(vf)):
        nm = vf[i]
        if v["dataType"][i] == "real":
            typ = "Float64"
        if v["dataType"][i] in ["integer", "unsigned integer", "signed integer"]:
            typ = "Int64"
        if v["dataType"][i] in ["string", "uri"]:
            typ = "string"
        if v["dataType"][i] == "dateTime":
            typ = "datetime64[ns, UTC]"
        dtdict[nm] = typ

    return dtdict


def cast_table_neon(data_table, var_table):
    """

    Cast a NEON data table to the correct data types for each variable, if possible.

    Parameters
    --------
    data_table: NEON data table as a pandas table.
    var_table: NEON variables file as a pandas table.

    Return
    --------
    A data frame of a NEON data table, with column classes assigned by data type.

    Example
    --------
    >>> dattab = cast_table_neon(data_table=brd_perpoint,
                                 var_table=variables_10003)

    Created on Oct 30 2024

    @author: Claire Lunch
    """

    # Check inputs formatting
    if not isinstance(data_table, pd.DataFrame):
        logging.info("Data table input is not a pandas data frame.")
        return None

    if not isinstance(var_table, pd.DataFrame):
        logging.info("Variables table input is not a pandas data frame.")
        return None

    # Check this is a valid variables file
    if any(x in ["category", "system", "stat"] for x in list(var_table.columns)):
        logging.info(
            "var_table appears to match DP4.00200.001. Data wrangling for surface-atmosphere exchange data is currently only available in the R package version of neonUtilities."
        )
        return None
    else:
        if not any(
            x in ["table", "fieldName", "dataType"] for x in list(var_table.columns)
        ):
            logging.info(
                "var_table is not a variables file, or is missing critical values."
            )
            return None

    # get data types
    vdt = get_variables_pandas(var_table)

    # get field names from the data table
    tabcols = list(data_table.columns)
    cast_table = data_table

    # iterate over columns and try to cast each
    for i in tabcols:
        if i not in vdt.keys():
            continue
        else:
            if vdt[i] in ["Float64", "Int64"]:
                try:
                    dtemp = cast_table[i].replace(r"^\s*$", np.nan, regex=True)
                    cast_table[i] = dtemp.astype(vdt[i])
                except Exception:
                    logging.info(
                        f"Field {i} could not be cast to type {vdt[i]}. Data read as string type."
                    )
                    cast_table[i] = data_table[i]
                    continue
            if vdt[i] == "datetime64[ns, UTC]" and not i == "publicationDate":
                try:
                    cast_table[i] = date_convert(data_table[i])
                except Exception:
                    logging.info(
                        f"Field {i} could not be cast to type {vdt[i]}. Data read as string type."
                    )
                    cast_table[i] = data_table[i]
                    continue

    return cast_table
