#!/usr/bin/env python
# coding: utf-8

## add github repo to path
#import sys
#sys.path.append(r'/rsstu/users/j/jcdietri/DHS-CRCoE-2016-2020/tacuevas/github/Kalpana')
from kalpana.downscaling import runStatic

'''
Example for doing the static downscaling using an existing grass location, and importing
the DEM with the mesh elements size. Both inputs were created in the example_01.
There is a short description of all inputs below, more detail can be found in the
docstring of the function in the github repository.
'''

## full path of the maxele file
ncFile = r'/data/north_carolina/inputs/maxele.63.nc' ## only in example02

## contour levels to use in the downscaling
## from 0 to 11 (included) every 1
levels = [0, 11, 1] ## only in example02

## output CRS
epsgOut = 6543

## full path for the shape file with the maxele contours
## same path is used for saving rasters and the grass location
pathOut = r'/data/north_carolina/outputs/maxele_florence.shp'

## version of grass 8.2 and 8.3 works
grassVer = 8.3

## path of the downscaling rasters
pathRasFiles = r'/data/north_carolina/inputs/'

## rasters filenames, can be a list if more than one. 
## 'all' for importing ALL THE FILES in pathRasFiles 
rasterFiles = 'ncDEMs_epsg6543'

## full path of the raster with the mesh element size
meshFile = r'/data/north_carolina/outputs/NC9.tif' ## only in example02

## crs of adcirc output (default value)
epsgIn = 4326

## vertical unit of the maxele
vUnitIn = 'm' ## only in example02

## vertical unit of the downscaled water levels
vUnitOut = 'ft' ## only in example02

## name of the maxele variable to downscale. Always 'zeta_max' for downscaling
var = 'zeta_max' ## only in example02

## contours type. Always 'polygon' for downscaling
conType = 'polygon' ## only in example02

## full path of file (kml, kmz, shp or gpkg) to crop the domain.
subDomain = None ## only in example02

## epsg code or crs of the subDomain. In this case, as we are using the downscaling dem bounding box
## as the subdomain, the same epsg code must be specified.
epsgSubDom = 6543

## boolean for exporting the mesh as a shape file from maxele, not necessary in this
## case since mesh was exported as preprocess. In example_03 it is exported.
exportMesh = False ## only in example02

## full path of pickle file with vertical datum differences for all mesh nodes
## proprocess step
dzFile = '/data/north_carolina/inputs/NC9mesh_from_tss2navd88.pkl' ## only in example02

## threshold to do apply the vertical datum difference, below -20 vyperdatum gives weird
## results
zeroDif = -20 ## only in example02

## full path of the grass location if a existing one will be used
## if None a new location called 'grassLoc' is created. A new location is created in
## example_03
nameGrassLocation = 'grassLoc' ## only in example02

## Boolean for creating grass location, in this example it was created as a preprocess
## step. In example_03 it is created.
createGrassLocation = False ## only in example02

## Method for assigning the crs to the grass location. Default and faster option
createLocMethod = 'from_raster' ## only in example02

## variable to downscale, can be 'zMax', 'zMean' and 'zMin'. With 'zMean', the mean value
## of each contour is used.
attrCol = 'zMean' ## only in example02

## how many times the representative length the results are grown in the downscaling
repLenGrowing = 1.0  ## only in example02

## remove wet cells with water level below the ground surface
compAdcirc2dem = True ## only in example02

## transform the water level to water depth
floodDepth = False ## only in example02

## define clumpling threshold from mesh
clumpThreshold = 'from_mesh' ## only in example02

## percentage of the minimum element area to scale the clumping threshold
perMinElemArea = 1 ## only in example02

## export downscaled results as shape files. Slows down the process a couple of minutes
ras2vec = False ## only in example02

## boolean for exporing raw maxele as a DEM. Useful for debugging
exportOrg = False ## only in example02

#################### calling downscaling
runStatic(ncFile, levels, epsgOut, pathOut,  grassVer, pathRasFiles, rasterFiles, meshFile,
          epsgIn, vUnitIn, vUnitOut, var, conType, subDomain, epsgSubDom, exportMesh, dzFile, 
          zeroDif, nameGrassLocation, createGrassLocation, createLocMethod, attrCol, repLenGrowing, 
          compAdcirc2dem, floodDepth, clumpThreshold, perMinElemArea, ras2vec, exportOrg)
