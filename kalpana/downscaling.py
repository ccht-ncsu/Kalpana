import shutil
import os
import subprocess
import sys
import time
import dask
#from tqdm.dask import TqdmCallback # Changed
import geopandas as gpd
import rioxarray as rxr
from rasterio.crs import CRS
from rasterio.enums import Resampling
from export import nc2shp, mesh2gdf, fort14togdf, readSubDomain # Changed
from loguru import logger # Changed
import numpy as np
from scipy.ndimage import label, find_objects

'''
    All functions using GRASS GRIS has an argument 'pkg' which is the grass package.
    The package can't be imported before runing the 'grassEnvVar' function so it's
    not directly imported in this module and will be imported when the functions
    are executed.
    
    EXPLAIN WORKFLOW
'''

def delFiles(listFiles, typeFiles, pkg):
    ''' Delete files form a grass location
        Parameters
            listFiles: list
                list of strings with the name of files to delete
            typeFiles: stromg
                raster, vector (see grass manual for more options)
                https://grass.osgeo.org/grass80/manuals/g.remove.html
        Returns
            None
    '''
    for f in listFiles:
        pkg.run_command('g.remove', flags = 'fb', quiet = True, 
                      type = typeFiles, name = f)

def rastersToList(pathRasters, rasterFiles):
    ''' Create list with full path of rasters. 
        Parameters
            pathRasters: str
                path of the raster files. All must be in the same folder
            rasterFiles: list or str
                name(s) of the raster file(s).
        Return
            rasterFiles: list
                list with the complete path of each raster
    '''
    #### check type of rasterFiles input
    if type(rasterFiles) == str:
        rasterFiles = [rasterFiles]
    # else:
        # logger.info("Warning: if DEM raster resolutions do not match, the aggregate DEM " \ # Changed
                # "resolution will match the resolution of the first input raster.")
    rasterFiles = [os.path.join(pathRasters, r) for r in rasterFiles]
    return rasterFiles

def grassEnvVar(grassVer):
    ''' Include grass to the environmental variables
        Parameters
            grassVer: float
                Version of the grass software (The code was writen for v8.0).
        Return
            None
    '''
    ########### Launch GRASS
    if sys.platform == 'win32':
        grassBin = f'C:\\Program Files\GRASS GIS {grassVer}\\grass{10*grassVer:0.0f}.bat'
        startCmd = [grassBin, '--config', 'path']

        p = subprocess.Popen(startCmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        out = out.strip().decode('utf-8')

        if p.returncode != 0:
            logger.info(f'ERROR: {err}', file=sys.stderr) # Changed
            logger.info(f'ERROR: Cannot find GRASS GIS {grassVer} start script: {startCmd}', file=sys.stderr) # Changed
            sys.exit(-1)

        ########### Add environment variables
        gisbase = out.strip(os.linesep)
        os.environ['GISBASE'] = gisbase

        ########### Init GRASS environment
        gpydir = os.path.join(gisbase, "etc", "python")
        sys.path.append(gpydir)
    
    elif sys.platform == 'linux':
        sys.path.append(subprocess.check_output(["grass", "--config", "python_path"], text=True).strip())

    else:
        logger.info('OS not known! only windows and linux are supported') # Changed
        
def createGrassLoc(grassVer, locPath, createLocMethod, myepsg, rasFile):
    ''' Create a Grass location for the downscaling methods. 
        Parameters
            grassVer: float
                Version of the grass software (The code was writen for v8.0).
            locPath: str
                path where the location will be created
            createLocMethod: str
                Two options "from_epsg" (default) or "from_raster" otherwise an error will be thrown.
            myepsg: int
                epsg code (https://epsg.io/), only used if createLocMehot == "from_epsg".
            rasFile: str
                complete path of the raster which will be used to define the coordinate system
        Return
            None
    '''
    ########### Create location
    if sys.platform == 'win32':
        grassBin = f'C:\\"Program Files"\\"GRASS GIS {grassVer}"\\grass{10*grassVer:0.0f}.bat'
    elif sys.platform == 'linux':
        grassBin = 'grass'
    else:
        logger.info('OS not known! only windows and linux are supported') # Changed
    #### check if location already exist
    if os.path.isdir(locPath):
        shutil.rmtree(locPath)
    else:
        pass

    #### epsg can be obtained from a raster or specified by the user
    if createLocMethod == 'from_epsg':
        startCmd = grassBin + ' -c epsg:' + str(myepsg) + ' -e ' + locPath
    elif createLocMethod == 'from_raster':
        startCmd = grassBin + ' -c ' + rasFile + ' -e ' + locPath
    else:
        sys.exit('The create location method specified is incorrect. See the docstring!')

    p = subprocess.Popen(startCmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    if p.returncode != 0:
        logger.info(f'ERROR: {err}', file = sys.stderr) # Changed
        logger.info(f'"ERROR: Cannot create location {startCmd}', file = sys.stderr) # Changed
        sys.exit(-1)

def initGrass(locPath, pkg, mapset='PERMANENT'):
    ''' Set grass working environment
        Parameters
            locPath: str
                path of the grass location
            mapset: str. Default PERMANENT
                mapset where the core data of the project can be stored
    '''
    #### init grass environment
    pkg.init(os.path.join(locPath, mapset))

def importRasters(rasFiles, pkg, myepsg):
    ''' Import rasters to Grass gis location.
        Parameters
            rasFiles: list
                list with complete path of each raster file
            myepsg: int
                Output coordinate reference system
        Returns
            rasterOutList: list
                list with imported raster files
    '''
    #import grass.script as gs
    rasterOutList = []
    for ras in rasFiles:
        ## get filename without extension
        rasObj = os.path.basename(ras).split('.')[0]
        rasterOutList.append(rasObj)
        ## load raster with rasterio xarray
        r = rxr.open_rasterio(ras)
        ## get crs
        crs = r.rio.crs
        if crs == None: ## file is not georrefenced
            pkg.run_command('r.in.gdal', overwrite = True, input = ras, output=rasObj, 
                                flags = 'o')
        else:
            crs = crs.to_string()
            if '4326' in crs:
                ## reproject file before loading it into grass location
                reprojectRas(ras, myepsg)
                rasNew = os.path.splitext(ras)[0] + f'_epsg{myepsg}.tif'
                pkg.run_command('r.import', overwrite = True, input = rasNew, output = rasObj)
            else:
                pkg.run_command('r.import', overwrite = True, input = ras, output = rasObj)

    return rasterOutList
    
def importRasters_parallel(rasFiles, pkg, myepsg):
    ''' Import rasters to Grass gis location.
        Parameters
            rasFiles: list
                list with complete path of each raster file
            myepsg: int
                Output coordinate reference system
        Returns
            rasterOutList: list
                list with imported raster files
    '''    
    ras = rasFiles[0]
    rasObj = os.path.basename(ras).split('.')[0]
    rasterOutList = [rasObj]
    ## load raster with rasterio xarray
    r = rxr.open_rasterio(ras)
    ## get crs
    crs = r.rio.crs
    if crs == None: ## file is not georrefenced
        pkg.run_command('r.in.gdal', overwrite = True, input = ras, output=rasObj, 
                            flags = 'o', quiet = True)
    else:
        crs = crs.to_string()
        if '4326' in crs:
            ## reproject file before loading it into grass location
            reprojectRas(ras, myepsg)
            rasNew = os.path.splitext(ras)[0] + f'_epsg{myepsg}.tif'
            pkg.run_command('r.import', overwrite = True, input = rasNew, output = rasObj, quiet = True)
        else:
            pkg.run_command('r.import', overwrite = True, input = ras, output = rasObj, quiet = True)
    
    ## parallel execution for the rest of rasters
    if len(rasFiles) > 1:
        @dask.delayed
        def importRastersParallel(ras, pkg, myepsg,):
            ''' delayed functions to import rasters in parallel
            '''
            rasObj = os.path.basename(ras).split('.')[0]
            ## load raster with rasterio xarray
            r = rxr.open_rasterio(ras)
            ## get crs
            crs = r.rio.crs
            if crs == None: ## file is not georrefenced
                pkg.run_command('r.in.gdal', overwrite = True, input = ras, output=rasObj, 
                                    flags = 'o', quiet = True)
            else:
                crs = crs.to_string()
                if '4326' in crs:
                    ## reproject file before loading it into grass location
                    reprojectRas(ras, myepsg)
                    rasNew = os.path.splitext(ras)[0] + f'_epsg{myepsg}.tif'
                    pkg.run_command('r.import', overwrite = True, input = rasNew, output = rasObj, quiet = True)
                else:
                    pkg.run_command('r.import', overwrite = True, input = ras, output = rasObj, quiet = True)
            return rasObj

        tasks = [importRastersParallel(r, pkg, myepsg) for r in rasFiles[1:]]
        #with TqdmCallback(desc = "Importing DEMs"): # Changed
        logger.info('Begin importing DEMs') # Changed # Changed
        rasterOutList2 = dask.compute(tasks, scheduler = 'processes') # Changed
        logger.info('Finish importing DEMs') # Changed # Changed
        rasterOutList = rasterOutList + rasterOutList2[0]
                
    return rasterOutList

def setDownscalingDEM(rasList, pkg, lim=500):
    ''' Create grass region based in input rasters
        Parameters
            rasList: list
                list of the raster objects imported
            lim: int. Default 500
                maximum number of raster to be imported at once before patching them
        Return
            res: float
                downscaling dem resolution in map units
    '''
    ## new line
    pkg.run_command('g.region', raster = rasList, quiet = True)
    
    if 1 < len(rasList) < lim:
        pkg.run_command('r.patch', input = rasList, output = 'dem',
                        overwrite = True, quiet = True)
        delFiles(rasList, 'raster', pkg)
    
    elif len(rasList) >= lim:
        pDems = []
        for i in range(int(len(rasList)/lim) + 1):
            sublist = rasList[i*lim:(i+1)*lim]
            pkg.run_command('r.patch', input = sublist, output = f'dem{i:03d}',
                overwrite = True, quiet = True)
            pDems.append(f'dem{i:03d}')
            delFiles(sublist, 'raster', pkg)
        pkg.run_command('r.patch', input = pDems, output = 'dem',
                overwrite = True, quiet = True)
        delFiles(pDems, 'raster', pkg)
    
    else:
        pkg.run_command('g.rename', raster = (rasList[0], 'dem'), 
                        overwrite = True, quiet = True)

def vertUnitConvert(conv, pkg):
    ''' Convert vertical units from meter 2 feet or feet 2 meter.
        Parameters
            conv: str
                "m2ft" or "ft2m"
        Return
            None
    '''
    # #import grass.script as gs
    demOrg = f"dem_{conv.split('2')[0]}"
    pkg.run_command('g.rename',raster = f"dem, {demOrg}")
    
    if conv == 'm2ft':
        pkg.mapcalc("$output=if(!isnull(demOrg),demOrg*3.2808399,null())",
                        output="dem",
                        overwrite=True,
                        quiet=True)
    
    elif conv == 'ft2m':
        pkg.mapcalc("$output=if(!isnull(demOrg),demOrg/3.2808399,null())",
                        output="dem",
                        overwrite=True,
                        quiet=True)
    else:
        sys.exit('Wrong argument. Available options are: m2ft or ft2m')

def setGrassEnv(grassVer, pathGrassLocation, createGrassLocation, pkg0, pkg1,
                pathRasFiles, rasterFiles, createLocMethod, myepsg):
    ''' Initialize grass environment
        Parameters
            grassVer: float
                Version of the grass software (The code was writen for v8.0).
            pathGrassLocation: str
                path and name of the grass location
            createGrassLocation: boolean
                True for creating a new location and loading DEMs, false to use an existing location with DEMs already imported
            pathRasFiles: str
                folder of the raster files
            rasterfiles: str or list
                name or names of the raster files to import
            createLocMethod: str
                Two options "from_epsg" (default) or "from_raster" otherwise an error will be thrown.
            myepsg: int
                psg code (https://epsg.io/), only used if createLocMehot == "from_epsg".
    '''
    if createGrassLocation == True:
        ta = time.time()
        rasFiles = rastersToList(pathRasFiles, rasterFiles) ## list with path of rasters to import
        logger.info(f'        rasters to list: {(time.time() - ta)/60: 0.3f} min') # Changed
        if os.path.exists(rasFiles[0]):
            ta = time.time()
            createGrassLoc(grassVer, pathGrassLocation, createLocMethod, myepsg, rasFiles[0])
            logger.info(f'        create location: {(time.time() - ta)/60: 0.3f} min') # Changed
        else:
            sys.exit(f'Raster file does not exist: {rasFiles[0]}')
        
        ta = time.time()
        initGrass(pathGrassLocation, pkg1)
        logger.info(f'        init grass: {(time.time() - ta)/60: 0.3f} min') # Changed
        
        ta = time.time()
        grassRasList = importRasters_parallel(rasFiles, pkg0, myepsg) ## try rasters with different resolution? check if rasters are in different crs that location
        ### oarallel
        logger.info(f'        import raster: {(time.time() - ta)/60: 0.3f} min') # Changed
        
        ta = time.time()
        setDownscalingDEM(grassRasList, pkg0)
        logger.info(f'        set downscaling dem: {(time.time() - ta)/60: 0.3f} min') # Changed
    
    else:
        ta = time.time()
        initGrass(pathGrassLocation, pkg1)
        logger.info(f'        init grass: {(time.time() - ta)/60: 0.3f} min') # Changed
        
def setupGrowing(kalpanaShp, attrCol, mesh2ras, meshFile, pkg, myepsg, exportOrg):
    ''' Preprocess kalpana shape file
        Parameters
            kalpanaShp: str
                path of the shape file with AdCirc results
            attrCol: str
                name of the attribute column
            mesh2ras: boolean
                True for converting the mesh vector layer to raster
            meshFile: str
                path of the shapefile with the mes elements or the rasterized version of it.
                if mesh2ras is False, meshFile must be the path of a raster, if True of a shapefile.
            myepsg: int
                Output coordinate reference system
            exportOrg: boolean. Default False
                True to export the raw adcirc outputs (without growing) as a DEM. Useful for debuging.
                
    '''
    ## import shape file with max water level
    t0 = time.time()
    pkg.run_command('v.in.ogr', input = kalpanaShp, overwrite = True,
                    quiet = True, snap = 0.000001, min_area = 10,
                    flags = 'o', stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Changed
    logger.info(f'        Import kalpana shapefile: {(time.time() - t0)/60:0.2f} min') # Changed

    t0 = time.time()
    pkg.run_command('v.to.rast', input = os.path.basename(kalpanaShp[:-4]), 
                    type = 'area', output = 'kalpanaRast', use = 'attr', 
                    quiet = True, attribute_column = attrCol, overwrite = True)
    logger.info(f'        Kalpana shape to raster: {(time.time() - t0)/60:0.2f} min') # Changed
 
    if exportOrg == True: #export adcirc output without growing as tif
        pkg.run_command('r.out.gdal', input = 'kalpanaRast', flags = 'cm', format = 'GTiff', 
                        nodata = -9999, output = f'{kalpanaShp[:-4]}.tif', overwrite = True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Changed
    
    if mesh2ras == True: #exportMesh is True so meshFile is a shapefile
    
        t0 = time.time()
        pkg.run_command('v.in.ogr', input = meshFile, overwrite = True,
                        quiet = True, min_area = 10, flags = 'o', snap = 0.1)
        logger.info(f'        Import mesh shapefile: {(time.time() - t0)/60:0.2f} min') # Changed
 
        t0 = time.time()
        pkg.run_command('v.to.rast', input = os.path.splitext(os.path.basename(meshFile))[0], 
                        type = 'area', use = 'attr', quiet = True, 
                        attribute_column = 'repLen', overwrite = True,
                        output = os.path.splitext(os.path.basename(meshFile))[0])
        logger.info(f'        Mesh shape to raster: {(time.time() - t0)/60:0.2f} min') # Changed
        t0 = time.time()
        pkg.run_command('r.out.gdal', input = os.path.splitext(os.path.basename(meshFile))[0], 
                        flags = 'cm', format = 'GTiff', nodata = -9999, overwrite = True,
                        output = os.path.splitext(meshFile)[0] + '.tif', stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Changed
        logger.info(f'        Mesh exported as raster: {(time.time() - t0)/60:0.2f} min') # Changed
 
    else: # raster mesh already exists
        ## try to find file in the grass location
        ## get only name of the meshFile without extension
        mfile = os.path.splitext(os.path.basename(meshFile))[0]
        ## find file
        mfile_dct = pkg.find_file(mfile)
        ## import if it isn't in the location
        if mfile_dct['name'] == '':
            importRasters_parallel([meshFile], pkg, myepsg)
        else: ## pass if it is already in there
            pass

def staticGrow(repLenFactor, pkg, meshFile):
    ''' Execture r.grow.distance algorithm
        Parameters
            repLenFactor: float. Default 0.5
                factor of the representative length of each triangle used as a maximum
                growing distance. Adcirc results are expanded until the distance to the nearest non-null cel
                is equals to the repLenFactor times the representative length of the triangle of the grown cell is in.
            meshFile: str
                name of the rasterized layer with the mesh elements representative size.
    '''
    # elif growRadius > 0: ## grow raster cells with a limiting distance on growRadius
    # growRadiusSq = growRadius**2 #distance on r.grow is calculated using squared metric so it is the squared of the actual distance
    t0 = time.time()
    # kalpanaRast is the output of nc2shp rasterized
    pkg.run_command('r.grow.distance', input = 'kalpanaRast', metric = 'squared', distance = 'grownRastDist',
                    value = 'grownRastVal', overwrite = True, quiet = True) ## flag m: distance in meters
    t1 = time.time()
    logger.info(f'        Running r.grow algorithm: {(t1 - t0)/60:0.3f} min') # Changed
    repLenFactorSq = repLenFactor**2
    # grownKalpanaRast0 is the horizontaly expanded raster
    pkg.mapcalc("$output = if(!isnull($input), $input, if($dist <= $fac * $radius * $radius, $new, null()))",
          output = 'grownKalpanaRast0', input = 'kalpanaRast', radius = meshFile, base = 'dem', fac = repLenFactorSq,
          new = 'grownRastVal', dist = 'grownRastDist', quiet = True, overwrite = True)
    t2 = time.time()
    logger.info(f'        Limit grown raster using adcirc mesh: {(t2 - t1)/60:0.3f} min') # Changed

def clumping(rasterGrown, rasterOrg, rasterNew, clumpSizeThreshold, pkg):
    ''' Function to deal with clump of disconnected cells generated after remove cells
        where the ground level is larger than the grown adcirc value
        Parameters
            rasterGrown: str
                name of the grown raster to analize
            rasterOrg: str
                name of the original raster with the non grown adcirc results
            rasterNew: str
                name of the outpur raster
            clumpSizeThreshold: int or float
                threshold to delete isolated cells. Groups of cells smaller than threshold
                will be removed if are not connected to the main water area.
    '''
    pkg.mapcalc("$output = if(!isnull($input), -1, null())", output = 'temp1', 
                   input = rasterGrown, quiet = True, overwrite = True)

    pkg.run_command('r.clump', input = 'temp1', output = 'temp2', 
                   quiet = True, overwrite = True)

    areas = pkg.read_command('r.stats', input = 'temp2', sort = 'desc', 
                            flags = 'c', quiet = True).split()
    
    rastInfo = pkg.parse_command('r.info', map = 'dem', flags = 'g', delimiter = '=')
    nsres = float(rastInfo['nsres'])
    ewres = float(rastInfo['ewres'])
    res = 0.5*(nsres + ewres)
    
    clumpThres = int(clumpSizeThreshold / res)
    
    ## find index of '*' --> it is used to indentify the clump masked
    try:
        i = areas.index('*')
        areas.pop(i)
        areas.pop(i)
    except: ## just in case a DEM has not masked value (rare)
        pass
    
    ## get clump ID of the ones with more cells than thres
    clumpsID = [x for x, y in zip(areas[::2], areas[1::2]) if int(y) >=  clumpThres]
    reclassList = ''.join([f"{i} = -1\n" for i in clumpsID])
    
    pkg.write_command('r.reclass', input = 'temp2', output = 'temp3', rules = '-', 
                stdin = reclassList, quiet = True, overwrite = True)
    
    # Passes back grown ADCIRC cells if they coincide with the assigned value (-1).
    pkg.mapcalc("$output = if($A == -1, $B, null())", output = rasterNew, A = 'temp3', 
               B = rasterGrown, quiet = True, overwrite = True)
               
def clumpingV2(rasterOrg, rasterGrown, rasterNew, pkg0, pkg1, leveesShp = None):
    ''' Function to deal with clumps of disconnected cells generated after remove cells
        where the ground level is larger than the grown adcirc value. It keeps if the clumps only if they overlay
        with wet cells in the non downscaled ADCIRC raster
        Parameters
            rasterGrown: str
                name of the grown raster to analize
            rasterOrg: str
                name of the original raster with the non grown adcirc results
            rasterNew: str
                name of the outpur raster
            pkg0: grass.script as gs
            pkg1: from grass.script import array as garray
            leveesShp: string
                full path of the shapefile with the levees as lines. Default None.
  '''
    ## set to 1 cells originaly wet in the raw raster
    pkg0.mapcalc("$output = if(!isnull($input), 1, 0)", output = 'kalpanaRast_mask', 
                                    input = rasterOrg, quiet = True, overwrite = True)
    ## grass raster as numpy array
    kalpanaRast_mask = pkg1.array('kalpanaRast_mask')
    
    if type(leveesShp) == str:
        ## load levees shapefile
        pkg0.run_command('v.import', input = leveesShp , output = 'levees', quiet = True, overwrite = True)
        ## rasterize levees
        pkg0.run_command('v.to.rast', input = 'levees', type = 'line', output = 'leveesRast', use = 'cat', 
                                                    quiet = True, overwrite = True)
         ## remove water from cells were the leveesRast is defined
        pkg0.mapcalc("$output = if(isnull($lev), $water, null())", output = 'grownKalpanaRast2', 
                                        lev = 'leveesRast', water = rasterGrown, quiet = True, overwrite = True)
    else:
        ## rename raster if levees not provided
        pkg0.run_command('g.rename', raster = (rasterGrown, 'grownKalpanaRast2'), 
                                                    overwrite = True, quiet = True)
     
     ## mask raster corrected with levees
    pkg0.mapcalc("$output = if(!isnull($input), 1, 0)", output = 'grownKalpanaRast2_mask', 
                                input = 'grownKalpanaRast2', quiet = True, overwrite = True)
    ## raster as numpy array
    grownKalpanaRast2_mask = pkg1.array('grownKalpanaRast2_mask')
    grownKalpanaRast2 = pkg1.array('grownKalpanaRast2')
    ## find clumps in grown kalpana output
    grownKalpanaRast2_clumps, grownKalpanaRast2_num_clumps = label(grownKalpanaRast2_mask)
    ## get slices with the clumps, it is like a list of smaller dems
    grownKalpanaRast2_clump_slices = find_objects(grownKalpanaRast2_clumps)
    
    ## intersect the clumps in the grown kalpana outputs with the raw raster to keep only the clumps where there was water originally
    intersecting_clumps = []
    for i in range(grownKalpanaRast2_num_clumps):
        ## get values of kalpanaRast mask at the grown kalapana output clump
        dummy0 = kalpanaRast_mask[grownKalpanaRast2_clump_slices[i]]
        ## get values of dummy0 where the slice values match the clump ID
        dummy1 = dummy0[grownKalpanaRast2_clumps[grownKalpanaRast2_clump_slices[i]] == i+1]
       
        if np.any(dummy1 == 1):
             ## if any of the cells in dummy1 are masked, then we store the modified slice
            intersecting_clumps.append((grownKalpanaRast2_clumps[grownKalpanaRast2_clump_slices[i]] == i+1).astype(int))
        else:
            ## add zero matrix of slice size
            intersecting_clumps.append(np.zeros_like(dummy0))
            
    ## reconstruct the full raster
    newRaster = pkg1.array()

    # Iterate through each clump and place it back into the reconstructed matrix
    for slice_obj, clump in zip(grownKalpanaRast2_clump_slices, intersecting_clumps):
        # get the coordinates of the clump in the original raster
        coords = tuple(slice(np.min(s.start, 0), np.min(s.stop, 0)) for s in slice_obj)
        # Place the clump back into the corresponding positions in the matrix
        newRaster[coords] = np.maximum(clump, newRaster[coords])
        
    newRaster.write(mapname="kalpanaRast2_finalMask", overwrite=True)
    
    pkg0.mapcalc("$output = if($mask > 0, $grown, null())", grown = 'grownKalpanaRast2', 
                mask = 'kalpanaRast2_finalMask', output = rasterNew,
                quiet = True, overwrite = True)

def postProcessStatic(compAdcirc2dem, floodDepth, kalpanaShp, pkg0, pkg1, levshp = None, ras2vec=False):
    ''' Postprocess the grown raster
        Parameters
            compAdcirc2dem: boolean
                True for removing ADCIRC cells with values lower than the dem
            flooDepth: boolean
                True for transform water levels to water depth. False for export
                water levelss
            kalpanaShp: str
                path of the shape file with AdCirc results. It is used to save the
                downscaled shp and raster files
            pkg0: grass.script as gs
            pkg1: from grass.script import array as garray
            levShp: string
                full path of the shapefile with the levees as lines. Default None.
            ras2vec: boolean. Default False
                If False output raster are not saved as shapefiles
    '''

    # if compArcirc2dem is True means ADCIRC cells with values lower than the dem
    # elevation will be removed
    if compAdcirc2dem == True:
        ta = time.time()
        # grownKalpanaRast1 is the horizontaly expanded raster with wet areas corrected with ground level
        pkg0.mapcalc("$output = if(!isnull($dem), if($dem > $adcirc, null(), $adcirc), $adcirc)",
                    output = 'grownKalpanaRast1', adcirc = 'grownKalpanaRast0', 
                    dem = 'dem', quiet = True, overwrite = True)
        logger.info(f'        Delete ground level: {(time.time() - ta)/60:0.3f}') # Changed
    else:
        ta = time.time()
        pkg0.run_command('g.rename', raster = ('grownKalpanaRast0', 'grownKalpanaRast1'), 
                      overwrite = True, quiet = True)
        logger.info(f'        Rename: {(time.time() - ta)/60:0.3f}') # Changed
 
    ta = time.time()
    #clumping('grownKalpanaRast1', 'kalpanaRast', 'grownKalpanaRastLevel', clumpThreshold, pkg)
    clumpingV2('kalpanaRast', 'grownKalpanaRast1', 'grownKalpanaRastLevel', pkg0, pkg1, levshp)
    logger.info(f'        Delete unconnected clumps: {(time.time() - ta)/60:0.3f}') # Changed
    
    pathOut = os.path.dirname(kalpanaShp)
    fileOut = os.path.basename(kalpanaShp)[:-4]# remove extension
    
    ta = time.time()
    # export raster as tif
    pkg0.run_command('r.out.gdal', input = 'grownKalpanaRastLevel', flags = 'mc', format = 'GTiff', nodata = -9999, 
                   output = os.path.join(pathOut, f'{fileOut}_level_downscaled.tif'), 
                   overwrite = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info(f'        export as tif level: {(time.time() - ta)/60:0.3f}') # Changed
 
    if ras2vec == True:
        ta = time.time()
        #Export to ESRI shapefile
        pkg0.run_command('r.to.vect', input = 'grownKalpanaRastLevel', output = 'grownKalpanaVectLevel', type = 'area')
        logger.info(f'        ras to vector: {(time.time() - ta)/60:0.3f}') # Changed
 
        ta = time.time()
        pkg0.run_command('v.out.ogr', input = 'grownKalpanaVectLevel', 
                          output = os.path.join(pathOut, f'{fileOut}_level_downscaled'), 
                          type = 'area', format = 'ESRI_Shapefile', flags = 'se', quiet = True, overwrite = True)
        logger.info(f'        export as shp level: {(time.time() - ta)/60:0.3f}') # Changed
 
    if floodDepth == True: ## export water depth in flooded cells
        ta = time.time()
        pkg0.mapcalc("$output = if(!isnull($dem) && !isnull($grown), $grown - $dem, null())", 
                                        grown = 'grownKalpanaRastLevel', dem = 'dem', output = 'grownKalpanaRastDepth', 
                                        quiet = True, overwrite = True)
        logger.info(f'        compute water depth: {(time.time() - ta)/60:0.3f}') # Changed
 
        ta = time.time()
        # export raster as tif
        pkg0.run_command('r.out.gdal', input = 'grownKalpanaRastDepth', flags = 'cm', format = 'GTiff', nodata = -9999, 
                       output = os.path.join(pathOut, f'{fileOut}_depth_downscaled.tif'), 
                       overwrite = True)
        logger.info(f'        export as tif depth: {(time.time() - ta)/60:0.3f}') # Changed
 
        ta = time.time()
        if ras2vec == True:
            #Export to ESRI shapefile
            pkg0.run_command('r.to.vect', input = 'grownKalpanaRastDepth', output = 'grownKalpanaVectDepth', type = 'area')
            logger.info(f'        ras to shp depth: {(time.time() - ta)/60:0.3f}') # Changed
 
            ta = time.time()
            pkg0.run_command('v.out.ogr', input = 'grownKalpanaVectDepth', 
                                                        output = os.path.join(pathOut, f'{fileOut}_depth_downscaled'), 
                                                        type = 'area', format = 'ESRI_Shapefile', flags = 'se', quiet = True, overwrite = True)
            logger.info(f'        export as shp depth: {(time.time() - ta)/60:0.3f}') # Changed
 
def runStatic(ncFile, levels, epsgOut, pathOut, grassVer, pathRasFiles, rasterFiles, meshFile, epsgIn=4326, 
                                 vUnitIn='m', vUnitOut='ft', var='zeta_max', conType ='polygon', subDomain=None, epsgSubDom=None, 
                                 exportMesh=False, dzFile=None, zeroDif=-20, nameGrassLocation=None, createGrassLocation=True, 
                                 createLocMethod='from_raster', attrCol='zMean', repLenGrowing=1.0, compAdcirc2dem=True, 
                                 floodDepth=False, ras2vec=False, exportOrg=False, leveesFile = None, finalOutToLatLon=True):
    ''' Run static downscaling method and the nc2shp function of the kalpanaExport module.
        Parameters
        ********************************************************************************************************************
        ****************************** REQUIRED inputs of nc2shp function **************************************************
        ********************************************************************************************************************
            ncFile: string
                path of the adcirc output, must be a netcdf file
            levels:list
                Contour levels. Min, Max and Step. Max IS included.
                Values must be in vUnitOut vertical unit.
            epsgOut: int
                coordinate system of the output shapefile
            pathout: string
                complete path of the output file (*.shp)
        ********************************************************************************************************************
        ***************************************** REQUIRED inputs of static method *****************************************
        ********************************************************************************************************************
            grassVer: float
                Version of the grass software (The code was writen for v8.0 but tested for 8.2 and 8.3 versions).
            pathRasFiles: str
                path of the raster files
            rasterFiles: list or str
                name(s) of the raster file(s). If equals to 'all', all dems inside the 'pathRasters' will be used.
                In case 'all' is used, the pathRasters folder must has only DEM files.
            meshFile: string
                path of the raster file with the mesh elements if it was generated beforehand. Otherwise. it is the filename to
                save the mesh shapefile and after the raster with the representative mesh elements size.
                This file does not depend of the simulation itself, it is only related to the mesh. This means it is not 
                necessary to export the adcirc mesh as a shapefile each time a simulation is downscaled. 
                For speed up the downscaling process, this input should not point to a raster only the first time a mesh is used!
        ********************************************************************************************************************
        ***************************************** OPTIONAL inputs of nc2shp function ***************************************
        ********************************************************************************************************************
            epsgIn: int. Default 4326.
                coordinate system of the adcirc input.
            vUnitIn, vUnitOut: string. Default for vUnitIn is 'm' and 'ft' for vUnitOut
                input and output vertical units. For the momment only supported 'm' and 'ft'
            var: string. DEFAULT zeta_max
                Name of the variable to export
            conType: string. DEFAULT polygon
                'polyline' or 'polygon'
            subDomain: str or list. Default None
                complete path of the subdomain polygon kml,  shapelfile or tif, or list with the
                uper-left x, upper-left y, lower-right x and lower-right y coordinates. The crs must be the same of the
                adcirc input file. It is recommended to use the same downscaling raster.
            epsgSubDom:int
                coordinate reference system of the subDomain
            exportMesh: boolean. Default False
                True to export and save it as a shapefile. It needs to be true in case the raster with the representative
                mesh size wasn't generated before. If that raster, which is required for the downscaled, is not available,
                this parameter must be True.
            dzFile: str
                full path of the pickle file with the vertical difference between datums
                for each mesh node
            zeroDif: int
                threshold for using nearest neighbor interpolation to change datum. Points below
                this value won't be changed.
        ********************************************************************************************************************
        ***************************************** OPTIONAL inputs of static method *****************************************
        ********************************************************************************************************************
            nameGrassLocation: str. DEFAULT None
                path and name of the grass location. If None the grass location will be called 'grassLoc' and save in the 
                same path of the extracted shape file (pathout).
            createGrassLocation: boolean. DEFAULT True
                True for creating a new location and loading DEMs, false to use an existing location with DEMs already imported
            createLocMethod: str. DEFAULT 'from_raster'
                Two options "from_epsg" or "from_raster" otherwise an error will be thrown.
            attrCol: str. DEFAULT 'avgVal'
                name of the attribute column
            repLenGrowing: float. Default 1.0
                factor of the representative length of each triangle used as a maximum
                growing distance. Adcirc results are expanded until the distance to the nearest non-null cel
                is equals to the repLenFactor times the representative length of the triangle of the grown cell is in.
            compAdcirc2dem: boolean. DEFAULT True
                True for removing ADCIRC cells with values lower than the dem.
            floodDepth: boolean . DEFAULT True
                True for transform water levels to water depth. False for export water levelss.
            ras2vec: boolean. Default False
                For speed up the process is recommended that raster files should not be converted to shapefiles (False).
            exportOrg: boolean. Default False
                True to export the raw adcirc outputs (without growing) as a DEM. Useful for debuging.
            leveesFile: string
                full path of the shapefile with the levees as lines. Default None.       
            finalOutToLatLon: boolean. Default True
                True to reproject final downscaled dem to lat/lon.
    '''
    t0 = time.time()
    pathaux = os.path.dirname(pathOut)
    if not os.path.exists(pathaux):
        os.mkdir(pathaux)
    
    if exportMesh == True:
        gdf, mesh = nc2shp(ncFile, var, levels, conType, pathOut, epsgOut, 
                           vUnitOut, vUnitIn, epsgIn, subDomain, epsgSubDom, exportMesh,
                           os.path.splitext(os.path.basename(meshFile))[0], dzFile, zeroDif)
        meshFile = os.path.join(pathaux, os.path.splitext(os.path.basename(meshFile))[0] + '.shp')
    else:
        gdf = nc2shp(ncFile, var, levels, conType, pathOut, epsgOut, vUnitOut, 
                     vUnitIn, epsgIn, subDomain, epsgSubDom, dzFile = dzFile, zeroDif = zeroDif)
        #Not needed anymore since we are isomg clumpingV2 fx. the mesh is not used to get a clumpig threshold 
        #mesh = gpd.read_file(os.path.splitext(meshFile)[0]+'.shp', ignore_geometry = True)
    
    if epsgOut == 4326:
        sys.exit('Downscaling can not be done for the lat lon crs.')
    
#    if clumpThreshold == 'from_mesh':
#        thres = mesh.elemArea.min() * perMinElemArea ## meter**2
#        if vUnitIn == 'm' and vUnitOut == 'ft':
#            thres = thres * (3.28084)**2 ## ft**2
#        elif vUnitIn == 'm' and vUnitOut == 'm':
#            pass
#    else:
#        thres = clumpThreshold

    logger.info(f'Static downscaling started') # Changed
    t1 = time.time()
    
    grassEnvVar(grassVer)
    ## import grass
    import grass.script as gs
    import grass.script.setup as gsetup
    from grass.script import array as garray
    
    if nameGrassLocation == None:
        pathGrassLocation = os.path.join(pathaux, 'grassLoc')
    else:
        pathGrassLocation = os.path.join(pathaux, nameGrassLocation)

    logger.info(f'    Start Setup grass environment') # Changed
    t11 = time.time()
    
    if rasterFiles == 'all':
        rasterFiles = os.listdir(pathRasFiles)
    ## setup grass env
    setGrassEnv(grassVer, pathGrassLocation, createGrassLocation, gs, gsetup,
                                    pathRasFiles, rasterFiles, createLocMethod, epsgOut)
    
    t2 = time.time()
    logger.info(f'    Setup grass environment: {(t2 - t11)/60:0.2f} min') # Changed
 
    ## setup growing
    logger.info(f'    Start Downscaling preprocess') # Changed
    ## here the thres must be in square meters
    setupGrowing(pathOut, attrCol, exportMesh, meshFile, gs, epsgOut, exportOrg)
    t3 = time.time()
    logger.info(f'    Downscaling preprocess: {(t3 - t2)/60:0.3f} min') # Changed
 
    ## grow
    logger.info(f'    Start growing') # Changed
    staticGrow(repLenGrowing, gs, os.path.splitext(os.path.basename(meshFile))[0])
    t4 = time.time()
    logger.info(f'    Ready with static grow: {(t4 - t3)/60:0.3f} min') # Changed
 
    ## postprocess
    logger.info(f'    Start postprocessing') # Changed
    postProcessStatic(compAdcirc2dem, floodDepth, pathOut, gs, garray, leveesFile, ras2vec)
    t5 = time.time()
    logger.info(f'    Ready with postprocess: {(t5 - t4)/60:0.3f} min') # Changed
    logger.info(f'Ready with static downscaling: {(t5 - t1)/60:0.3f} min') # Changed
 
    if finalOutToLatLon == True:
        finalOut = os.path.join(pathaux, os.path.basename(pathOut)[:-4] + '_level_downscaled.tif') 
        reprojectRas(finalOut, pathaux, epsgOut = 4326)
    logger.info(f'Kalpana finished sucsesfully after: {(t5 - t0)/60:0.3f} min') # Changed
    logger.info(f'Output files saved on: {pathaux}') # Changed

def meshRepLen2raster(fort14, epsgIn, epsgOut, pathOut, grassVer, pathRasFiles, rasterFiles, subDomain=None, 
                                                        nameGrassLocation=None, createGrassLocation=True, createLocMethod='from_raster', 
                                                        exportDEM=True):
    ''' Function to rasterize mesh shapefile created from the fort.14 file
        Parameters
            fort14: str
                full path of the fort.14 file
            epsgIn: int
                coordinate system of the adcirc input
            epsgOut: int
                coordinate system of the output shapefile
            grassVer: float
                Version of the grass software (The code was writen for v8.0).
            pathRasters: str
                path of the raster files
            rasterFiles: list or str
                name(s) of the raster file(s). If 'all' is input, all files in pathRasters are used
            subDomain: str or list. Default None
                complete path of the subdomain polygon kml or shapelfile, or list with the
                uper-left x, upper-left y, lower-right x and lower-right y coordinates. The crs must be the same of the
                adcirc input file.
            nameGrassLocation: str. DEFAULT None
                path and name of the grass location. If None the grass location will be called 'grassLoc' and save in the 
                same path of the extracted shape file (pathout).
            createGrassLocation: boolean. DEFAULT True
                True for creating a new location and loading DEMs, false to use an existing location with DEMs already imported
            createLocMethod: str. DEFAULT 'from_raster'
                Two options "from_epsg" or "from_raster" otherwise an error will be thrown.
        Returns
            None
    '''
    ## create gdf from fort14 file with elements as geometries
    t0 = time.time()
    if fort14.endswith('.nc'):
        gdfMesh = fort14togdf(fort14, epsgIn, epsgOut) ## default netcdf
    else:
        gdfMesh = fort14togdf(fort14, epsgIn, epsgOut, fileintype = 'fort.14')
    logger.info(f'fort14 to mesh: {(time.time() - t0)/60:0.3f} min') # Changed
    
    ## clip contours if requested
    if subDomain is not None:
        t0 = time.time()
        subDom = readSubDomain(subDomain, epsgIn)
        gdfMesh = gpd.clip(gdfMesh, subDom)
        logger.info(f'Clip mesh using subfomain: {(time.time() - t0)/60:0.3f} min') # Changed

    ## export gdf as shapefile
    t0 = time.time()
    gdfMesh.to_file(pathOut)
    logger.info(f'Export mesh gdf as shapefile: {(time.time() - t0)/60:0.3f} min') # Changed
    ## path where the grass loc will be created
    pathaux = os.path.dirname(pathOut)
    ## add grass to the environment variables
    grassEnvVar(grassVer)
    ## import grass
    import grass.script as gs
    import grass.script.setup as gsetup
    ## grass location path
    if nameGrassLocation == None:
        pathGrassLocation = os.path.join(pathaux, 'grassLoc')
    else:
        pathGrassLocation = os.path.join(pathaux, nameGrassLocation)

    logger.info(f'    Start Setup grass environment') # Changed
    t11 = time.time()
 
    if rasterFiles == 'all':
        rasterFiles = os.listdir(pathRasFiles)
 
    ## setup grass env
    setGrassEnv(grassVer, pathGrassLocation, createGrassLocation, gs, gsetup,
                pathRasFiles, rasterFiles, createLocMethod, epsgOut)
    
    if exportDEM == True:
        gs.run_command('r.out.gdal', input = 'dem', flags = 'cm', format = 'GTiff', nodata = -9999,
               output = os.path.join(os.path.dirname(pathOut), 'downscaling_dem.tif'))
    
    ## get minimum area
    minArea = gdfMesh.elemArea.min()
    t0 = time.time()
    ## load mesh into grass
    gs.run_command('v.in.ogr', input = pathOut, overwrite = True,
                    quiet = True, min_area = int(minArea/100)*100, flags = 'o', snap = 0.1)
    logger.info(f'        Import mesh shapefile: {(time.time() - t0)/60:0.2f} min') # Changed
    ## mesh shapefile to raster
    t0 = time.time()
    gs.run_command('v.to.rast', input = os.path.splitext(os.path.basename(pathOut))[0], 
                    type = 'area', use = 'attr', quiet = True, 
                    attribute_column = 'repLen', overwrite = True,
                    output = os.path.splitext(os.path.basename(pathOut))[0])
    logger.info(f'        Mesh shape to raster: {(time.time() - t0)/60:0.2f} min') # Changed
    ## export raster
    t0 = time.time()
    gs.run_command('r.out.gdal', input = os.path.splitext(os.path.basename(pathOut))[0], 
                    flags = 'mc', format = 'GTiff', nodata = -9999, overwrite = True,
                    output = os.path.splitext(pathOut)[0] + '.tif')
    logger.info(f'        Mesh exported to raster: {(time.time() - t0)/60:0.2f} min') # Changed
    
def reprojectRas(filein, pathout, epsgOut=None, res='same'):
    ''' Reproject and change resolution of rasters
        Parameters
            filein: str
                full path of the input raster
            pathout: str
                path where the new file will be stored, same filename
                will be used with new epsg or resolution if changed
            epsgOut: int. Default None
                coordinate system of the output raster
            res: int or float. Default None
                desired resolution
        Returns
            rasOut: rioxarray raster object
                updated raster
    '''
    ## open raster
    rasIn = rxr.open_rasterio(filein)
    bname = os.path.splitext(os.path.basename(filein))[0]
    ## reproject if raster is in wgs84 (lat/lon)
    if res == 'same':
        logger.info('Reproject '+bname+' to EPSG 4326')
        rasOut = rasIn.rio.reproject(epsgOut)
        logger.info('Reprojected '+bname+' to EPSG 4326') # Changed
        #logger.info('Write '+bname+' to COG') # Changed 
        #rasOut.rio.to_raster(os.path.join(pathout, bname + f'_epsg{epsgOut}.tif'), driver="COG") # Changed
        #logger.info('Wrote '+bname+' to COG') # Changed 
        logger.info('Write '+bname+' to TIFF') # Changed 
        rasOut.rio.to_raster(os.path.join(pathout, bname + f'_epsg{epsgOut}.tif')) # Changed
        logger.info('Wrote '+bname+' to TIFF') # Changed 
        program_list = [['rio', 'cogeo', 'create', os.path.join(pathout, bname + f'_epsg{epsgOut}.tif'), os.path.join(pathout, bname + f'_epsg{epsgOut}_cog.tif', '--web-optimized')]] # Changed 
        # Run list of program commands using subprocess                                                                                                                                                                            
        for program in program_list:                                                                                                                                                                                               
            logger.info('Write '+bname+' to COG')                                                  
            output = subprocess.run(program, shell=False, check=True)                      
            logger.info('Wrote '+bname+' to COG') # Changed       
   
    ## change resolution
    else:
        ## same crs
        if epsgOut == None:
            scaleFactor = rasIn.rio.resolution()[0] / res
            newWidth = int(rasIn.rio.width * scaleFactor)
            newHeight = int(rasIn.rio.height * scaleFactor)

            logger.info('Reproject '+bname+' Resampling.bilinear 1') # Changed
            rasOut = rasIn.rio.reproject(rasIn.rio.crs, shape = (newHeight, newWidth),
                                        resampling = Resampling.bilinear)
            logger.info('Reprojected '+bname+' Resampling.bilinear 1') # Changed
            logger.info('Write '+bname+' to TIFF') # Changed
            #rasOut.rio.to_raster(os.path.join(pathout, bname + f'_res{res}.tif'), driver="COG") # Changed
            rasOut.rio.to_raster(os.path.join(pathout, bname + f'_res{res}.tif')) # Changed
            logger.info('Wrote '+bname+' to TIFF') # Changed
            program_list = [['rio', 'cogeo', 'create', os.path.join(pathout, bname + f'_epsg{epsgOut}.tif'), os.path.join(pathout, bname + f'_epsg{epsgOut}_cog.tif', '--web-optimized')]] # Changed 
            # Run list of program commands using subprocess                                                                                                                                                                            
            for program in program_list:                                                                                                                                                                                               
                logger.info('Write '+bname+' to COG')                                                  
                output = subprocess.run(program, shell=False, check=True)                      
                logger.info('Wrote '+bname+' to COG') # Changed 
        else:
            rasOut = rasIn.rio.reproject(epsgOut)
            scaleFactor = rasOut.rio.resolution()[0] / res
            newWidth = int(rasOut.rio.width * scaleFactor)
            newHeight = int(rasOut.rio.height * scaleFactor)
            logger.info('Reproject '+bname+' Resampling.bilinear 2') # Changed
            rasOut = rasIn.rio.reproject(rasOut.rio.crs, shape = (newHeight, newWidth),
                                resampling = Resampling.bilinear)
            logger.info('Reprojected '+bname+' Resampling.bilinear 2') # Changed
            #logger.info('Write '+bname+' to COG') # Changed
            #rasOut.rio.to_raster(os.path.join(pathout, bname + f'_epsg{epsgOut}_res{res}.tif'), driver="COG")
            #logger.info('Wrote '+bname+' to COG') # Changed
            logger.info('Write '+bname+' to TIFF') # Changed                                                                                                                                                                           
            rasOut.rio.to_raster(os.path.join(pathout, bname + f'_epsg{epsgOut}.tif')) # Changed                                                                                                                                       
            logger.info('Wrote '+bname+' to TIFF') # Changed 
            program_list = [['rio', 'cogeo', 'create', os.path.join(pathout, bname + f'_epsg{epsgOut}.tif'), os.path.join(pathout, bname + f'_epsg{epsgOut}_cog.tif', '--web-optimized')]] # Changed 
            # Run list of program commands using subprocess                                                                                                                                                                            
            for program in program_list:                                                                                                                                                                                               
                logger.info('Write '+bname+' to COG') # Changed  
                output = subprocess.run(program, shell=False, check=True)                      
                logger.info('Wrote '+bname+' to COG') # Changed                    
 
    return rasOut
            
def mergeDEMs(grassVer, pathRasFiles, rasterFiles, pathOut, epsgOut,
              nameGrassLocation=None, createGrassLocation=True, createLocMethod='from_raster'):
    ''' Function to rasterize mesh shapefile created from the fort.14 file
        Parameters
            grassVer: float
                Version of the grass software (The code was writen for v8.0).
            pathRasters: str
                path of the raster files
            rasterFiles: list or str
                name(s) of the raster file(s). If 'all' is input, all files in pathRasters are used
            pathOut: str
                full path of the output DEM
            nameGrassLocation: str. DEFAULT None
                path and name of the grass location. If None the grass location will be called 'grassLoc' and save in the 
                same path of the extracted shape file (pathout).
            createGrassLocation: boolean. DEFAULT True
                True for creating a new location and loading DEMs, false to use an existing location with DEMs already imported
             createLocMethod: str. DEFAULT 'from_raster'
                 Two options "from_epsg" (default) or "from_raster" otherwipathOut, epsgOut,se an error will be thrown.
        Returns
            None
    '''
    pathaux = os.path.dirname(pathOut)
    ## add grass to the environment variables
    grassEnvVar(grassVer)
    ## import grass
    import grass.script as gs
    import grass.script.setup as gsetup
    ## grass location path
    if nameGrassLocation == None:
        pathGrassLocation = os.path.join(pathaux, 'grassLoc')
    else:
        pathGrassLocation = os.path.join(pathaux, nameGrassLocation)

    if rasterFiles == 'all':
        rasterFiles = os.listdir(pathRasFiles)
        
    ## setup grass env
    setGrassEnv(grassVer, pathGrassLocation, createGrassLocation, gs, gsetup,
                pathRasFiles, rasterFiles, createLocMethod, epsgOut)
        
    gs.run_command('r.out.gdal', input = 'dem', flags = 'cm', format = 'GTiff', nodata = -9999, 
           output = pathOut)
