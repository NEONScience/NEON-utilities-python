### 2025-10-27 v1.2.0

Bug fixes:

* Fixed time stamp conversion in stack_by_table() - now working for all time stamp formats in tabular data
* Horizontal and vertical index labeling fixed in DP4.00001.001

Enhancements:

* Added stack_eddy() function to extract data from DP4.00200.001 HDF5 files and stack into tabular format
* Added a workflow to by_file_aop() and by_tile_aop() to check for files already in local storage and offer options for handling



### 2025-05-27 v1.1.0

Updates for initial approval by pyOpenSci. Documentation updates, and:

Bug fixes:

* Fixed error workflow for file path length errors in Windows
* cloud_mode in load_by_product() was failing when site="all"; fixed

Enhancements:

* Added files_by_uri() function to download files linked by URLs in NEON tabular data


### 2024-12-04 v1.0.1

Bug fixes:

* Stacking was failing when data type inconsistencies were found in some sensor and observational data products. Stacking functions updated to handle these edge cases.


### 2024-09-23 v1.0.0

Initial public release of neonutilities 1.0.0

