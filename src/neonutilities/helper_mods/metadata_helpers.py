#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os


def get_recent(fl_set, fltype):
    """

    Given a set of file names and urls for NEON data files, returns the url for the most recently published item of type=fltype, based on publication datetime stamp in the file name.

    Parameters
    --------
    fl_set: A set of file names and urls for NEON data files, in the format returned via the data endpoint of the NEON API.
    fltype: The type of file to be returned (e.g. variables, sensor positions).

    Return
    --------
    The most recently published object (contains file name, url, size).

    Created on Apr 18 2024

    @author: Claire Lunch
    """

    # subset to files of specified type
    ty = re.compile(fltype)
    flt = [f for f in fl_set if ty.search(f["name"])]

    if len(flt) == 0:
        return None

    # get max date stamp in subset
    pdater = re.compile("[0-9]{8}T[0-9]{6}Z")
    flvar = [pdater.search(os.path.basename(f["name"])).group(0) for f in flt]

    if len(flvar) == 0:
        return None

    recdate = max(flvar)

    # get url matching max date stamp
    maxr = re.compile(recdate)
    flmax = [f for f in flt if maxr.search(os.path.basename(f["name"]))]

    return [flmax[0]]


def convert_byte_size(size_bytes):
    """
    This function converts the file size in bytes to a more readable format.
    It converts bytes to Kilobytes (KB), Megabytes (MB), Gigabytes (GB), or Terabytes (TB)
    depending on the size of the input. This uses 1024 bytes as the base for conversion.

    Parameters
    --------
    size_bytes: int or float
        The file size in bytes. It should be a non-negative number.

    Returns
    --------
    str
        A string that represents the file size in a more readable format.
        The format includes the size number followed by the size unit (KB, MB, GB, or TB).

    Raises
    --------
    None

    Examples
    --------
    >>> convert_byte_size(5000)
    '4.9 KB'

    >>> convert_byte_size(2000000)
    '1.9 MB'

    >>> convert_byte_size(3000000000)
    '2.8 GB'

    >>> convert_byte_size(4000000000000)
    '3.6 TB'
"""
    si_prefix = [[40, 'T'],
                 [30, 'G'],
                 [20, 'M'],
                 [10, 'K'],
                 [ 0, '' ]]
    
    for si_row in si_prefix:
        if (size_bytes >= 2 ** si_row[0]):
            break
    
    return f'{(size_bytes / 2 ** si_row[0]):.1f} {si_row[1]}B'
