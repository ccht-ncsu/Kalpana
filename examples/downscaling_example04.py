# %%
import os
from downscalingHeadLoss import preCompCostSurface
from loguru import logger

# %%
grassVer = 8.2
createGrassLocation = True
pathGrassLocation = r'/mnt/drive1/Insyncs/NCSU/Kalpana/Debug/headLoss01_Coned10m_r1'
pathRasFiles = r'/mnt/drive1/Insyncs/NCSU/Kalpana/Data/topo'
rasterFiles = r'NC_CoNED_res10m.tif'
manningRasPath = r"/mnt/drive1/Insyncs/NCSU/Kalpana/Data/manning/NLCD_2016_Land_Cover_L48_20190424.img"
manningLandCover = r"/home/tacuevas/github/Kalpana/adds/manning/landCover_manning.txt"
epsg = 6346
pathOutRawCostRas = os.path.join(pathGrassLocation, 'rawCostRaster_NCconed10m.tif')
pathOutTotalCostRas = os.path.join(pathGrassLocation, 'totalCostRaster_NCconed10m.tif')
pathOutDownDemCorr = os.path.join(pathGrassLocation, 'downDemCorr_NCconed10m.tif')



preCompCostSurface(grassVer, createGrassLocation, pathGrassLocation, pathRasFiles, rasterFiles, epsg, manningRasPath, manningLandCover, 
                                    pathOutRawCostRas, pathOutTotalCostRas, pathOutDownDemCorr, nameGrassLocation = None, createLocMethod = 'from_raster', URConstant=1, 
                                    k=1, waterClass=11, minArea=20000000, res=5, slopeFactor=-0.2125, walkCoeefs=[0, 1, -1, -1])