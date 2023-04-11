import os
import sys
from kalpana.downscaling import runStatic
import time
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    now = time.time()
    ## read
    
    with open(r'/home/kalpana/inputs/runKalpanaStatic.inp') as inp:
        lines = inp.readlines()[3:]
        lines = [x.split()[0] for x in lines]
        state = lines[0]
        mesh = lines[1]
        storm = lines[2]
        advisory = int(lines[3])
        levels = lines[4]
        epsgOut = int(lines[5])
        rasterFiles = lines[6]
        meshFile = lines[7]
        epsgIn = int(lines[8])
        vUnitOut = lines[9]
        subDomain = lines[10] ## 'None' to None
        subDom_epsg = lines[11]
        exportMesh = lines[12] ## 'True' or 'False' to boolean
        dzFile = lines[13]
        repLenGrowing = float(lines[14])
        compAdcirc2dem = lines[15] ## 'True' or 'False' to boolean
        floodDepth = lines[16] ## 'True' or 'False' to boolean
        clumpThreshold = lines[17]
        perMinElemArea = float(lines[18])
        ras2vec = lines[19] ## 'True' or 'False' to boolean

    ## transform strings to boolean objects
    subDomain = None if subDomain == 'None' else subDomain
    exportMesh = True if exportMesh == 'True' else False
    compAdcirc2dem = True if compAdcirc2dem == 'True' else False
    floodDepth = True if floodDepth == 'True' else False
    ras2vec = True if ras2vec == 'True' else False
    
    ncFile = r'/home/kalpana/inputs/maxele.63.nc'
    aux = Path(r'/home/kalpana/downscaling')

    levels = levels.split(',')
    levels = [float(levels[0]), float(levels[1]), float(levels[2])]
    pathOut = str(aux/state/'maxele.shp')
    grassVer = 8.2
    pathRasFiles = aux/state/'inputs'
    dzFile = str(pathRasFiles/dzFile)
    vUnitIn = 'm'
    var = 'zeta_max'
    conType = 'polygon'
    zeroDif = -20
    nameGrassLocation = str(aux/state/'grassLoc')
    createGrassLocation = False
    createLocMethod = 'from_raster'
    attrCol = 'zMean'
    exportOrg = False
    meshFile = str(aux/state/meshFile)
    
    if subDomain == rasterFiles:
        subDomain = os.path.join(pathRasFiles, rasterFiles)

    runStatic(ncFile, levels, epsgOut, pathOut,  grassVer, str(pathRasFiles), rasterFiles, meshFile,
              epsgIn, vUnitIn, vUnitOut, var, conType, subDomain, subDom_epsg, exportMesh, dzFile, zeroDif, 
              nameGrassLocation, createGrassLocation, createLocMethod, attrCol, repLenGrowing, 
              compAdcirc2dem, floodDepth, clumpThreshold, perMinElemArea, ras2vec, exportOrg)
              
    # shutil.copyfile(aux/state/'maxele_level_downscaled.tif', path_to_store)
