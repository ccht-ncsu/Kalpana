import os
import sys
# sys.path.append(r'/home/tacuevas/github/Kalpana/kalpana')
## this is not necessary if the module is in the same folder
from downscaling import runStatic
'''
Example for doing the static downscaling using an existing grass location, and importing
the DEM with the mesh elements size. Both inputs were created in the example_01.
There is a short description of all inputs below, more detail can be found in the
docstring of the function in the github repository.
'''


## full path of the maxele file
ncFile = r'../adds/inputs_examples/maxele.63.nc'
## contour levels to use in the downscaling
## from 0 to 11 (included) every 1
levels = [0,20, 1]
## output CRS
epsgOut = 6346
## full path for the shape file with the maxele contours
## same path is used for saving rasters and the grass location
pathOut = r'/home/tomas/Downloads/maxele_florence.shp'
## version of grass 8.2 and 8.3 works
grassVer = 8.2
## path of the downscaling rasters
pathRasFiles = r'../adds/inputs_examples'
## rasters filenames, can be a list if more than one. 
## 'all' for importing ALL THE FILES in pathRasFiles 
rasterFiles = 'NC_CoNED_subset_100m.tif'
## full path of the raster with the mesh element size
meshFile = r'/home/tomas/Downloads/NC9_NCConED_25m.tif'
## crs of adcirc output (default value)
epsgIn = 4326
## vertical unit of the maxele
vUnitIn = 'm'
## vertical unit of the downscaled water levels
vUnitOut = 'm'
## name of the maxele variable to downscale. Always 'zeta_max' for downscaling
var = 'zeta_max'
## contours type. Always 'polygon' for downscaling
conType = 'polygon'
## full path of file (kml, kmz, shp, gpkg or tif) to crop the domain.
## in this case we will use the same downscaling raster bounding box as the subdomain
subDomain = os.path.join(pathRasFiles, rasterFiles)
## epsg code or crs of the subDomain. In this case, as we are using the downscaling dem bounding box
## as the subdomain, the same epsg code must be specified.
epsgSubDom = 6346
## boolean for exporting the mesh as a shape file from maxele, not necessary in this
## case since mesh was exported as preprocess. In example_03 it is exported.
exportMesh = False
## full path of pickle file with vertical datum differences for all mesh nodes
## proprocess step
dzFile = r'../adds/dzDatumsNOAA/dzDaums_noaaTideGauges_msl_navd88.csv'
## threshold to do apply the vertical datum difference
zeroDif = -20
##threshold to define the percentage of the dz given by the spatial interpolation to be applied.
maxDif = -5
## only tide stations closed than this  threshold are used to interpolate the vertical datum difference
distThreshold = 0.5
## number of points to query for the inverse distance-weighted interpolation
k = 7
## full path of the grass location if a existing one will be used
## if None a new location called 'grassLoc' is created. A new location is created in
## example_03
nameGrassLocation = r'/home/tomas/Downloads/grassLoc'
## Boolean for creating grass location, in this example it was created as a preprocess
## step. In example_03 it is created.
createGrassLocation = False
## Method for assigning the crs to the grass location. Default and faster option
createLocMethod = 'from_raster'
## variable to downscale, can be 'zMax', 'zMean' and 'zMin'. With 'zMean', the mean value
## of each contour is used.
attrCol = 'zMean'
## how many times the representative length the results are grown in the downscaling
repLenGrowing = 1.0 
## remove wet cells with water level below the ground surface
compAdcirc2dem = True
## transform the water level to water depth
floodDepth = False
## export downscaled results as shape files. Slows down the process a couple of minutes
ras2vec = False
## boolean for exporing raw maxele as a DEM. Useful for debugging
exportOrg = False
## full path of the shapefile with levees
leveesFile = None
## boolean for reprojecting the downscaled dem back to lat/lon
finalOutToLatLon = False

#################### calling downscaling
runStatic(ncFile, levels, epsgOut, pathOut, grassVer, pathRasFiles, rasterFiles, meshFile, epsgIn=epsgIn, 
            vUnitIn=vUnitIn, vUnitOut=vUnitOut, var=var, conType =conType, subDomain=subDomain, epsgSubDom=epsgSubDom, 
            exportMesh= exportMesh, dzFile=dzFile, zeroDif=zeroDif, maxDif=maxDif, distThreshold=distThreshold, k=k, 
            nameGrassLocation=nameGrassLocation, createGrassLocation=createGrassLocation, createLocMethod=createLocMethod, 
            attrCol=attrCol, repLenGrowing=repLenGrowing, compAdcirc2dem=compAdcirc2dem, floodDepth=floodDepth, 
            ras2vec=ras2vec, exportOrg=exportOrg, leveesFile = leveesFile, finalOutToLatLon=finalOutToLatLon)
