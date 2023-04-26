#!/usr/bin/env python
# coding: utf-8

## add github repo to path
#import sys
#sys.path.append(r'/rsstu/users/j/jcdietri/DHS-CRCoE-2016-2020/tacuevas/github/Kalpana')
import os
from kalpana.downscaling import runStatic

'''
Example for doing the static downscaling using an existing grass location, and importing
the DEM with the mesh elements size. Both inputs were created in the example_01.
There is a short description of all inputs below, more detail can be found in the
docstring of the function in the github repository.
'''


## full path of the maxele file
ncFile = r'/mnt/drive1/GoogleDrive/NCSU/NCSU/Kalpana/Docker/inputs/maxele.63.nc'
## contour levels to use in the downscaling
## from 0 to 11 (included) every 1
levels = [0, 11, 1]
## output CRS
epsgOut = 6543
## full path for the shape file with the maxele contours
## same path is used for saving rasters and the grass location
pathOut = r'/mnt/drive1/GoogleDrive/NCSU/NCSU/Kalpana/Examples_github/example02/maxele_florence.shp'
## version of grass 8.2 and 8.3 works
grassVer = 8.2
## path of the downscaling rasters
pathRasFiles = r'/mnt/drive1/GoogleDrive/NCSU/NCSU/Kalpana/Docker/downscaling/north_carolina/inputs'
## rasters filenames, can be a list if more than one. 
## 'all' for importing ALL THE FILES in pathRasFiles 
rasterFiles = 'ncDEMs_epsg6543'
## full path of the raster with the mesh element size
meshFile = r'/mnt/drive1/GoogleDrive/NCSU/NCSU/Kalpana/Examples_github/example01/NC9.tif'
## crs of adcirc output (default value)
epsgIn = 4326
## vertical unit of the maxele
vUnitIn = 'm'
## vertical unit of the downscaled water levels
vUnitOut = 'ft'
## name of the maxele variable to downscale. Always 'zeta_max' for downscaling
var = 'zeta_max'
## contours type. Always 'polygon' for downscaling
conType = 'polygon'
## full path of file (kml, kmz, shp, gpkg or tif) to crop the domain.
## in this case we will use the same downscaling raster bounding box as the subdomain
subDomain = os.path.join(pathRasFiles, rasterFiles)
## epsg code or crs of the subDomain. In this case, as we are using the downscaling dem bounding box
## as the subdomain, the same epsg code must be specified.
epsgSumDom = 6543
## boolean for exporting the mesh as a shape file from maxele, not necessary in this
## case since mesh was exported as preprocess. In example_03 it is exported.
exportMesh = False
## full path of pickle file with vertical datum differences for all mesh nodes
## proprocess step
dzFile = r'/mnt/drive1/GoogleDrive/NCSU/NCSU/Kalpana/Docker/downscaling/north_carolina/inputs/NC9mesh_from_tss2navd88.pkl'
## threshold to do apply the vertical datum difference, below -20 vyperdatum gives weird
## results
zeroDif = -20
## full path of the grass location if a existing one will be used
## if None a new location called 'grassLoc' is created. A new location is created in
## example_03
nameGrassLocation = r'/mnt/drive1/GoogleDrive/NCSU/NCSU/Kalpana/Examples_github/example01/grassLoc'
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
## define clumpling threshold from mesh
clumpThreshold = 'from_mesh'
## percentage of the minimum element area to scale the clumping threshold
perMinElemArea = 1
## export downscaled results as shape files. Slows down the process a couple of minutes
ras2vec = False
## boolean for exporing raw maxele as a DEM. Useful for debugging
exportOrg = False
## boolean for reprojecting the downscaled dem back to lat/lon
finalOutToLatLon = True

#################### calling downscaling
runStatic(ncFile, levels, epsgOut, pathOut,  grassVer, pathRasFiles, rasterFiles, meshFile,
          epsgIn, vUnitIn, vUnitOut, var, conType, subDomain, epsgSumDom, exportMesh, dzFile, 
          zeroDif, nameGrassLocation, createGrassLocation, createLocMethod, attrCol, repLenGrowing, 
          compAdcirc2dem, floodDepth, clumpThreshold, perMinElemArea, ras2vec, exportOrg,
          finalOutToLatLon)
