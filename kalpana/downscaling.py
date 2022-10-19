import shutil
import os
import subprocess
import sys
import time
import geopandas as gpd
import rioxarray as rxr
from rasterio.crs import CRS
from rasterio.enums import Resampling
from export import nc2shp, mesh2gdf, fort14togdf

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
        # print("Warning: if DEM raster resolutions do not match, the aggregate DEM " \
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
            print(f'ERROR: {err}', file=sys.stderr)
            print(f'ERROR: Cannot find GRASS GIS {grassVer} start script: {startCmd}', file=sys.stderr)
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
        print('OS not known! only windows and linux are supported')
        
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
        print('OS not known! only windows and linux are supported')
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
        print(f'ERROR: {err}', file = sys.stderror)
        print(f'"ERROR: Cannot create location {startCmd}', file = sys.stderror)
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
        with TqdmCallback(desc = "Importing DEMs"):
            rasterOutList2 = dask.compute(tasks, scheduler = 'processes')
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
        print(f'        rasters to list: {(time.time() - ta)/60: 0.3f} min')
        if os.path.exists(rasFiles[0]):
            ta = time.time()
            createGrassLoc(grassVer, pathGrassLocation, createLocMethod, myepsg, rasFiles[0])
            print(f'        create location: {(time.time() - ta)/60: 0.3f} min')
        else:
            sys.exit(f'Raster file does not exist: {rasFiles[0]}')
        
        ta = time.time()
        initGrass(pathGrassLocation, pkg1)
        print(f'        init grass: {(time.time() - ta)/60: 0.3f} min')
        
        ta = time.time()
        grassRasList = importRasters_parallel(rasFiles, pkg0, myepsg) ## try rasters with different resolution? check if rasters are in different crs that location
        print(f'        import raster: {(time.time() - ta)/60: 0.3f} min')
        
        ta = time.time()
        setDownscalingDEM(grassRasList, pkg0)
        print(f'        set downscaling dem: {(time.time() - ta)/60: 0.3f} min')
    
    else:
        ta = time.time()
        initGrass(pathGrassLocation, pkg1)
        print(f'        init grass: {(time.time() - ta)/60: 0.3f} min')
        
def setupGrowing(kalpanaShp, attrCol, mesh2ras, meshFile, minArea, pkg, myepsg):
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
            minArea: int
                Minimum size of area to be imported (square meters) 
            myepsg: int
                Output coordinate reference system
                
    '''
    ## import shape file with max water level
    t0 = time.time()
    pkg.run_command('v.in.ogr', input = kalpanaShp, overwrite = True,
                    quiet = True, snap = 0.000001, min_area = 10,
                    flags = 'o')
    print(f'        Import kalpana shapefile: {(time.time() - t0)/60:0.2f} min')
    
    t0 = time.time()
    pkg.run_command('v.to.rast', input = os.path.basename(kalpanaShp[:-4]), 
                    type = 'area', output = 'kalpanaRast', use = 'attr', 
                    quiet = True, attribute_column = attrCol, overwrite = True)
    print(f'        Kalpana shape to raster: {(time.time() - t0)/60:0.2f} min')
    
    if mesh2ras == True: #exportMesh is True so meshFile is a shapefile
    
        t0 = time.time()
        pkg.run_command('v.in.ogr', input = meshFile, overwrite = True,
                        quiet = True, min_area = int(minArea/100)*100, flags = 'o', snap = 0.1)
        print(f'        Import mesh shapefile: {(time.time() - t0)/60:0.2f} min')
        
        t0 = time.time()
        pkg.run_command('v.to.rast', input = os.path.splitext(os.path.basename(meshFile))[0], 
                        type = 'area', use = 'attr', quiet = True, 
                        attribute_column = 'repLen', overwrite = True,
                        output = os.path.splitext(os.path.basename(meshFile))[0])
        print(f'        Mesh shape to raster: {(time.time() - t0)/60:0.2f} min')
        t0 = time.time()
        pkg.run_command('r.out.gdal', input = os.path.splitext(os.path.basename(meshFile))[0], 
                        flags = 'm', format = 'GTiff', nodata = -9999, overwrite = True,
                        output = os.path.splitext(meshFile)[0] + '.tif')
        print(f'        Mesh exported as raster: {(time.time() - t0)/60:0.2f} min')
        
    else: #exportMesh is True so meshFile is a raster
        importRasters_parallel([meshFile], pkg, myepsg)

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
    pkg.run_command('r.grow.distance', input = 'kalpanaRast', metric = 'squared', distance = 'grownRastDist',
                    value = 'grownRastVal', overwrite = True, quiet = True) ## flag m: distance in meters
    t1 = time.time()
    print(f'        Running r.grow algorithm: {(t1 - t0)/60:0.3f} min')
    repLenFactorSq = repLenFactor**2
    pkg.mapcalc("$output = if(!isnull($input), $input, if($dist <= $fac * $radius * $radius, $new, null()))",
          output = 'grownKalpanaRast0', input = 'kalpanaRast', radius = meshFile, base = 'dem', fac = repLenFactorSq,
          new = 'grownRastVal', dist = 'grownRastDist', quiet = True, overwrite = True)
    t2 = time.time()
    print(f'        Limit grown raster using adcirc mesh: {(t2 - t1)/60:0.3f} min')

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
    reclassList = ''.join([f"{i} = -1\n" for i in areas[::2]])
    
    pkg.write_command('r.reclass', input = 'temp2', output = 'temp3', rules = '-', 
                stdin = reclassList, quiet = True, overwrite = True)
    
    # Passes back grown ADCIRC cells if they coincide with the assigned value (-1).
    pkg.mapcalc("$output = if($A == -1, $B, null())", output = rasterNew, A = 'temp3', 
               B = rasterGrown, quiet = True, overwrite = True)

def postProcessStatic(compAdcirc2dem, floodDepth, kalpanaShp, clumpThreshold, pkg, ras2vec=False):
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
            clumpThreshold: int or float
                threshold to delete isolated cells. Groups of cells smaller than threshold will be removed 
                if are not connected to the main water area.
            ras2vec: boolean. Default False
                If False output raster are not saved as shapefiles
    '''

    # if compArcirc2dem is True means ADCIRC cells with values lower than the dem
    # elevation will be removed
    if compAdcirc2dem == True:
        ta = time.time()
        pkg.mapcalc("$output = if(!isnull($dem), if($dem > $adcirc, null(), $adcirc), $adcirc)",
                    output = 'grownKalpanaRast1', adcirc = 'grownKalpanaRast0', 
                    dem = 'dem', quiet = True, overwrite = True)
        print(f'        Delete ground level: {(time.time() - ta)/60:0.3f}')
    else:
        ta = time.time()
        pkg.run_command('g.rename', raster = ('grownKalpanaRast0', 'grownKalpanaRast1'), 
                      overwrite = True, quiet = True)
        print(f'        Rename: {(time.time() - ta)/60:0.3f}')
                      
    ta = time.time()
    clumping('grownKalpanaRast1', 'kalpanaRast', 'grownKalpanaRastLevel', clumpThreshold, pkg)
    print(f'        Delete unconnected clumps: {(time.time() - ta)/60:0.3f}')
    
    
    pathOut = os.path.dirname(kalpanaShp)
    fileOut = os.path.basename(kalpanaShp)[:-4]# remove extension
    
    ta = time.time()
    # export raster as tif
    pkg.run_command('r.out.gdal', input = 'grownKalpanaRastLevel', flags = 'm', format = 'GTiff', nodata = -9999, 
                   output = os.path.join(pathOut, f'{fileOut}_level_downscaled.tif'), 
                   overwrite = True)
    print(f'        export as tif level: {(time.time() - ta)/60:0.3f}')
    
    if ras2vec == True:
        ta = time.time()
        #Export to ESRI shapefile
        pkg.run_command('r.to.vect', input = 'grownKalpanaRastLevel', output = 'grownKalpanaVectLevel', type = 'area')
        print(f'        ras to vector: {(time.time() - ta)/60:0.3f}')
        
        ta = time.time()
        pkg.run_command('v.out.ogr', input = 'grownKalpanaVectLevel', 
                          output = os.path.join(pathOut, f'{fileOut}_level_downscaled'), 
                          type = 'area', format = 'ESRI_Shapefile', flags = 'se', quiet = True, overwrite = True)
        print(f'        export as shp level: {(time.time() - ta)/60:0.3f}')
    
    if floodDepth == True: ## export water depth in flooded cells
        ta = time.time()
        pkg.mapcalc("$output = if(!isnull($dem) && !isnull($grown), $grown - $dem, null())", grown = 'grownKalpanaRastLevel', 
                    dem = 'dem', output = 'grownKalpanaRastDepth', quiet = True, overwrite = True)
        print(f'        compute water depth: {(time.time() - ta)/60:0.3f}')
        
        ta = time.time()
        # export raster as tif
        pkg.run_command('r.out.gdal', input = 'grownKalpanaRastDepth', flags = 'm', format = 'GTiff', nodata = -9999, 
                       output = os.path.join(pathOut, f'{fileOut}_depth_downscaled.tif'), 
                       overwrite = True)
        print(f'        export as tif depth: {(time.time() - ta)/60:0.3f}')
        
        ta = time.time()
        if ras2vec == True:
            #Export to ESRI shapefile
            pkg.run_command('r.to.vect', input = 'grownKalpanaRastDepth', output = 'grownKalpanaVectDepth', type = 'area')
            print(f'        ras to shp depth: {(time.time() - ta)/60:0.3f}')
            
            ta = time.time()
            pkg.run_command('v.out.ogr', input = 'grownKalpanaVectDepth', 
                            output = os.path.join(pathOut, f'{fileOut}_depth_downscaled'), 
                            type = 'area', format = 'ESRI_Shapefile', flags = 'se', quiet = True, overwrite = True)
            print(f'        export as shp depth: {(time.time() - ta)/60:0.3f}')
        
def runStatic(ncFile, levels, epsgOut, pathOut,  grassVer, pathRasFiles, rasterFiles,
              epsgIn=4326, vUnitIn='m', vUnitOut='ft', var='zeta_max', conType ='polygon', 
              subDomain=None, exportMesh=False, dzFile=None, zeroDif=-20, 
              nameGrassLocation=None, createGrassLocation=True, createLocMethod='from_raster', attrCol='zMean', repLenGrowing=1.0, 
              meshFile=None, compAdcirc2dem=True, floodDepth=False, clumpThreshold='from_mesh', perMinElemArea=1, ras2vec=False):
    ''' Run static downscaling method and the nc2shp function of the kalpanaExport module.
        Parameters
        ********************************************************************************************************************
        ****************************** REQUIRED inputs of nc2shp function **************************************************
        ********************************************************************************************************************
            ncFile: string
                path of the adcirc output, must be a netcdf file
            levels: np.array
                Contour levels. The max value in the entire doman and over all timesteps is added to the requested levels.
            epsgOut: int
                coordinate system of the output shapefile
            pathout: string
                complete path of the output file (*.shp)
        ********************************************************************************************************************
        ***************************************** REQUIRED inputs of static method *****************************************
        ********************************************************************************************************************
            grassVer: float
                Version of the grass software (The code was writen for v8.0 but tested for 8.2 and 8.3 versions).
            pathRasters: str
                path of the raster files
            rasterFiles: list or str
                name(s) of the raster file(s). If equals to 'all', all dems inside the 'pathRasters' will be used.
                In case 'all' is used, the pathRasters folder must has only DEM files.
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
                complete path of the subdomain polygon kml or shapelfile, or list with the
                uper-left x, upper-left y, lower-right x and lower-right y coordinates. The crs must be the same of the
                adcirc input file.
            exportMesh: boolean. Default False
                True for export the mesh geodataframe and also save it as a shapefile
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
                Two options "from_epsg" (default) or "from_raster" otherwise an error will be thrown.
            attrCol: str. DEFAULT 'avgVal'
                name of the attribute column
            repLenFactor: float. Default 0.5
                factor of the representative length of each triangle used as a maximum
                growing distance. Adcirc results are expanded until the distance to the nearest non-null cel
                is equals to the repLenFactor times the representative length of the triangle of the grown cell is in.
            meshFile: string
                path of the raster file with the mesh elements. This file does not depend of the simulation it self, only on the mesh.
                So it is not necessary to export the adcirc mesh as a shapefile each time a simulation is downscaled. For speed up the
                downscaling process this input should be not None only the first time a mesh is used.
            compAdcirc2dem: boolean. DEFAULT True
                True for removing ADCIRC cells with values lower than the dem.
            floodDepth: boolean . DEFAULT True
                True for transform water levels to water depth. False for export water levelss.
            clumpThreshold: str or int or float. Default 'from_mesh'
                threshold to delete isolated cells. Groups of cells smaller than threshold will be removed if are not connected to the main water area. 
                If equals 'from_mesh' the minimum area of the triangles elements is used to define the threshold. If int or float, this value will be used as
                threshold, the unit of this number must be in the map units
            perMinElemArea: float. Default 1
                percentage of the smaller element's area used as a threshold to delete unconnected groups of raster cells. Only used if
                clumpThreshold is "from_mesh".
            ras2vec: boolean. Default False
                For speed up the process is recommended that raster files should not be converted to shapefiles (False).
    '''
    t0 = time.time()
    pathaux = os.path.dirname(pathOut)
    if not os.path.exists(pathaux):
        os.mkdir(pathaux)
    
    if exportMesh == True:
        gdf, mesh = nc2shp(ncFile, var, levels, conType, pathOut, epsgOut, 
                           vUnitOut, vUnitIn, epsgIn, subDomain, exportMesh,
                           os.path.splitext(os.path.basename(meshFile))[0], dzFile, zeroDif)
        meshFile = os.path.join(pathaux, os.path.splitext(os.path.basename(meshFile))[0] + '.shp')
    else:
        gdf = nc2shp(ncFile, var, levels, conType, pathOut, epsgOut, vUnitOut, 
                     vUnitIn, epsgIn, subDomain, dzFile = dzFile, zeroDif = zeroDif)
        mesh = gpd.read_file(os.path.splitext(meshFile)[0]+'.shp', ignore_geometry = True) # not fully sure if it is the best way
    
    if epsgOut == 4326:
        sys.exit('Downscaling can not be done for the lat lon crs.')
    
    if clumpThreshold == 'from_mesh':
        thres = mesh.elemArea.min() * perMinElemArea
    else:
        thres = clumpThreshold

    print(f'Static downscaling started')
    t1 = time.time()
    
    grassEnvVar(grassVer)
    ## import grass
    import grass.script as gs
    import grass.script.setup as gsetup
    
    if nameGrassLocation == None:
        pathGrassLocation = os.path.join(pathaux, 'grassLoc')
    else:
        pathGrassLocation = os.path.join(pathaux, nameGrassLocation)
    
    print(f'    Start Setup grass environment')
    t11 = time.time()
    
    if rasterFiles == 'all':
        rasterFiles = os.listdir(pathRasFiles)
    ## setup grass env
    setGrassEnv(grassVer, pathGrassLocation, createGrassLocation, gs, gsetup,
                pathRasFiles, rasterFiles, createLocMethod, epsgOut)
    
    t2 = time.time()
    print(f'    Setup grass environment: {(t2 - t11)/60:0.2f} min')
    
     ## setup growing
    print(f'    Start Downscaling preprocess')
    ## here the thres must be in square meters
    setupGrowing(pathOut, attrCol, exportMesh, meshFile, thres, gs, epsgOut)
    t3 = time.time()
    print(f'    Downscaling preprocess: {(t3 - t2)/60:0.3f} min')
    
    ## grow
    print(f'    Start growing')
    staticGrow(repLenGrowing, gs, os.path.splitext(os.path.basename(meshFile))[0])
    t4 = time.time()
    print(f'    Ready with static grow: {(t4 - t3)/60:0.3f} min')
    
   ## postprocess
    print(f'    Start postprocessing')
    postProcessStatic(compAdcirc2dem, floodDepth, pathOut, thres, gs, ras2vec)
    t5 = time.time()
    print(f'    Ready with postprocess: {(t5 - t4)/60:0.3f} min')
    print(f'Ready with static downscaling: {(t5 - t1)/60:0.3f} min')
    print(f'Kalpana finished sucsesfully after: {(t5 - t0)/60:0.3f} min')
    print(f'Output files saved on: {pathaux}')
    
def meshRepLen2raster(fort14, epsgIn, epsgOut, pathOut, grassVer, pathRasFiles, rasterFiles, subDomain=None, 
                      nameGrassLocation=None, createGrassLocation=True, createLocMethod='from_raster'):
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
                name(s) of the raster file(s).
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
                 Two options "from_epsg" (default) or "from_raster" otherwise an error will be thrown.
        Returns
            NOne
    '''
    ## create gdf from fort14 file with elements as geometries
    t0 = time.time()
    gdfMesh = fort14togdf(fort14, 4326, 6543)
    print(f'fort14 to mesh: {(time.time() - t0)/60:0.3f} min')
    
    ## clip contours if requested
    if subDomain is not None:
        t0 = time.time()
        subDom = readSubDomain(subDomain, epsgIn)
        gdfMesh = gpd.clip(gdfMesh, subDom)
        print(f'Clip mesh using subfomain: {(time.time() - t0)/60:0.3f} min')
        
    ## export gdf as shapefile
    t0 = time.time()
    gdfMesh.to_file(pathOut)
    print(f'Export mesh gdf as shapefile: {(time.time() - t0)/60:0.3f} min')
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

    print(f'    Start Setup grass environment')
    t11 = time.time()
    ## setup grass env
    setGrassEnv(grassVer, pathGrassLocation, createGrassLocation, gs, gsetup,
                pathRasFiles, rasterFiles, createLocMethod, epsgOut)
    ## get minimum area
    minArea = gdfMesh.elemArea.min()
    t0 = time.time()
    ## load mesh into grass
    gs.run_command('v.in.ogr', input = pathOut, overwrite = True,
                    quiet = True, min_area = int(minArea/100)*100, flags = 'o', snap = 0.1)
    print(f'        Import mesh shapefile: {(time.time() - t0)/60:0.2f} min')
    ## mesh shapefile to raster
    t0 = time.time()
    gs.run_command('v.to.rast', input = os.path.splitext(os.path.basename(pathOut))[0], 
                    type = 'area', use = 'attr', quiet = True, 
                    attribute_column = 'repLen', overwrite = True,
                    output = os.path.splitext(os.path.basename(pathOut))[0])
    print(f'        Mesh shape to raster: {(time.time() - t0)/60:0.2f} min')
    ## export raster
    t0 = time.time()
    gs.run_command('r.out.gdal', input = os.path.splitext(os.path.basename(pathOut))[0], 
                    flags = 'm', format = 'GTiff', nodata = -9999, overwrite = True,
                    output = os.path.splitext(pathOut)[0] + '.tif')
    print(f'        Mesh exported as raster: {(time.time() - t0)/60:0.2f} min')
    
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
            aux: int
                1 if the raster file was reproject, 0 otherwise
    '''
    ## open raster
    rasIn = rxr.open_rasterio(filein)
    bname = os.path.splitext(os.path.basename(filein))[0]
    ## reproject if raster is in wgs84 (lat/lon)
    if res == 'same':
        rasOut = rasIn.rio.reproject(epsgOut)
        rasOut.rio.to_raster(os.path.join(pathout, bname + f'_epsg{epsgOut}.tif'))
    ## change resolution
    else:
        scaleFactor = rasIn.rio.resolution()[0] / res
        newWidth = int(rasIn.rio.width * scaleFactor)
        newHeight = int(rasIn.rio.height * scaleFactor)
        
        ## same crs
        if epsgOut == None:
            rasOut = rasIn.rio.reproject(rasIn.rio.crs, shape = (newHeight, newWidth),
                                        resampling = Resampling.bilinear)
            rasOut.rio.to_raster(os.path.join(pathout, bname + f'_res{res}.tif'))
        
        ## change crs
        else:
            ## if raster in epsg 4326 reprojection and change of res
            ## doesn't work. First reprojection and then change of res
            if rasIn.rio.crs.to_string() == 'EPSG:4326':
                rasOut = rasIn.rio.reproject(epsgOut)
                scaleFactor = rasOut.rio.resolution()[0] / res
                newWidth = int(rasOut.rio.width * scaleFactor)
                newHeight = int(rasOut.rio.height * scaleFactor)
                rasOut = rasIn.rio.reproject(rasOut.rio.crs, shape = (newHeight, newWidth),
                                            resampling = Resampling.bilinear)                
            else:
                rasOut = rasIn.rio.reproject(epsgOut, shape = (newHeight, newWidth),
                                            resampling = Resampling.bilinear)
            rasOut.rio.to_raster(os.path.join(pathout, bname + f'_epsg{epsgOut}_res{res}.tif'))
