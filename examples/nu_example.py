import neonutilities as nu
import os

# download some veg structure data
veg = nu.load_by_product(
    dpid="DP1.10098.001",
    site=["WREF", "RMNP"],
    startdate="2022-01",
    enddate="2023-12",
    include_provisional=True,
    check_size=False,
    token=os.environ.get("NEON_TOKEN"),
)
# see what data tables were returned
veg.keys()

# download 30-minute PAR data
par = nu.load_by_product(
    dpid="DP1.00024.001",
    site="RMNP",
    startdate="2023-06",
    enddate="2023-07",
    timeindex=30,
    package="expanded",
    include_provisional=True,
    check_size=False,
    token=os.environ.get("NEON_TOKEN"),
)

# download CHM tiles covering the veg structure plots at WREF
pppy = veg["vst_perplotperyear"]
east = pppy["easting"].to_list()
north = pppy["northing"].to_list()

nu.by_tile_aop(
    dpid="DP3.30015.001",
    site="WREF",
    year=2023,
    easting=east,
    northing=north,
    buffer=20,
    include_provisional=True,
    check_size=False,
    savepath="INSERT FILE PATH",
    token=os.environ.get("NEON_TOKEN"),
)
