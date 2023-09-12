# %%
import os
from downscalingHeadLoss import preCompCostSurface
from loguru import logger

# %%
## grass version
grassVer = 8.2

## boolean flag for creating a GRASS GIS location
createGrassLocation = True

## path to the grass location
pathGrassLocation = r'/mnt/drive1/Insyncs/NCSU/Kalpana/Debug/headLoss01_Coned10m_r1'

## path of the downscaling raster
pathRasFiles = r'/mnt/drive1/Insyncs/NCSU/Kalpana/Data/topo'

## downscaling raster file name
rasterFiles = r'NC_CoNED_res10m.tif'

## full path of the land cover DEM
manningRasPath = r"/mnt/drive1/Insyncs/NCSU/Kalpana/Data/manning/NLCD_2016_Land_Cover_L48_20190424.img"

## rules list to convert from land cover to manning
manningLandCover = r"/home/tacuevas/github/Kalpana/adds/manning/landCover_manning.txt"

## coordinate reference system 
epsg = 6346

## full path of the output dem with the raw cost
pathOutRawCostRas = os.path.join(pathGrassLocation, 'rawCostRaster_NCconed10m.tif')

## full path of the output dem with the total cost
pathOutTotalCostRas = os.path.join(pathGrassLocation, 'totalCostRaster_NCconed10m.tif')

## full path of the output downscaling DEM. In case the downscaling DEM does not have bathy,
## cells with water land cover class are set to bathymetry 0 so water cells are included in the calculations.
pathOutDownDemCorr = os.path.join(pathGrassLocation, 'downDemCorr_NCconed10m.tif')


preCompCostSurface(grassVer, createGrassLocation, pathGrassLocation, pathRasFiles, rasterFiles, epsg, manningRasPath, manningLandCover, 
                                    pathOutRawCostRas, pathOutTotalCostRas, pathOutDownDemCorr, nameGrassLocation = None, createLocMethod = 'from_raster', URConstant=1, 
                                    k=1, waterClass=11, minArea=20000000, res=5, slopeFactor=-0.2125, walkCoeefs=[0, 1, -1, -1])