## first step is to add the github repo to the path, so functions can be imported
#import sys
#sys.path.append(r'/rsstu/users/j/jcdietri/DHS-CRCoE-2016-2020/tacuevas/github/Kalpana')
## in this case is not necessary since the module is in the same folder
from kalpana.downscaling import meshRepLen2raster

'''
This script creates a grass location importing the DEM for downscaling and also creates
a new DEM with same resolution and extend with the size of the mesh triangles. This step
is key for the downscaling and can be run in advance, since only depends on the mesh.
'''

fort14 = '/data/north_carolina/inputs/fort.14' ## path of the fort.14 file *
epsgIn = 4326 ## CRS for lat/lon 
epsgOut = 6543 ## projected CRS for NC 
pathOut = '/data/north_carolina/outputs/NC9.shp' ## full path of the output shapefile 
grassVer = 8.3 
pathRasFiles = '/data/north_carolina/inputs/' 
rasterFiles = 'ncDEMs_epsg6543' 
sbFile = '/data/north_carolina/inputs/NCSubDomain.shp' ## only in 

meshRepLen2raster(fort14, epsgIn, epsgOut, pathOut, grassVer, pathRasFiles, rasterFiles, 
                  subDomain=sbFile, nameGrassLocation=None, createGrassLocation=True, 
                  createLocMethod='from_raster')
