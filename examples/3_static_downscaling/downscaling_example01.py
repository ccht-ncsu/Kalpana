## first step is to add the github repo to the path, so functions can be imported
import os
import sys
# sys.path.append(r'/home/tacuevas/github/Kalpana/kalpana')
from downscaling import meshRepLen2raster

'''
This script creates a grass location importing the DEM for downscaling and also creates
a new DEM with same resolution and extend with the size of the mesh triangles. This step
is key for the downscaling and can be run in advance, since only depends on the mesh.
'''

fort14 = r'../adds/inputs_examples/fort.14' ## path of the fort.14 file
epsgIn = 4326 ## CRS for lat/lon
epsgOut = 6346 ## CRS of downscaling DEM
pathOut = r'/home/tomas/Downloads/NC9_NCConED_25m.shp' ## full path of the output shapefile 
grassVer = 8.2
pathRasFiles = r'../adds/inputs_examples'
rasterFiles = 'North_Carolina_CoNED_Topobathy_DEM_25m_version_20.tif'
## in this case we will use the same downscaling raster bounding box as the subdomain. 
subDomain=os.path.join(pathRasFiles, rasterFiles)
nameGrassLocation=None
createGrassLocation=True
createLocMethod='from_raster'

meshRepLen2raster(fort14, epsgIn, epsgOut, pathOut, grassVer, pathRasFiles, rasterFiles, 
                  subDomain=subDomain, nameGrassLocation=nameGrassLocation, 
                  createGrassLocation=createGrassLocation, 
                  createLocMethod=createLocMethod)
