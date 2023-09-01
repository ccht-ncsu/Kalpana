import os
import sys
import time
import numpy as np
import subprocess
from loguru import logger

sys.path.append(r'/home/tacuevas/github/Kalpana/kalpana')
from downscaling import importRasters_parallel, grassEnvVar, setGrassEnv, clumpingV2
from export import nc2shp

def setManning(manningRasPath, manningLandCover, pkg, URConstant=1, k=1):
    '''
    Sets Manning's n values and calculates unit head loss based on provided parameters.

    This function imports a land cover classes raster, reclassifies the land cover
    classification data to Manning's n values multiplied by 10000 (due to integer
    constraints of the reclassification process), converts the Manning's n values
    to decimal form, and then calculates the unit head loss for each cell within
    the specified domain using the Manning's equation.

    Parameters:
    manningRasPath (str): Path to the input raster containing land cover classes.
    manningLandCover (str): Path to the rules file for reclassifying land cover classes to Manning's n values.
    pkg: The package/module used for executing GRASS GIS commands.
    URConstant (float, optional): Unit Runoff Constant for the Manning's equation. Default is 1.
    k (float, optional): 1 for SI and 1.49 for imperial. Default is 1.

    Returns:
    None: The function performs necessary operations to set Manning's n values and calculate unit head loss.

    Note:
    - The input package `pkg` is expected to have a run_command method for executing GRASS GIS commands.

    Example:
    setManning("path/to/land_cover_raster", "path/to/reclass_rules", grass_package,
               URConstant=0.8, k=0.03)
    '''
    ## import land cover classes raster
    pkg.run_command('r.import', input = manningRasPath, output="landCoverClass",
                    resolution="region", resample="nearest", extent="region",
                    overwrite=True, quiet=True)
    
    ## reclassify land cover classification data to manning value * 10000
    ## multiplication by 10000 is necessary because r.reclass only accepts integers
    pkg.run_command('r.reclass', rules = manningLandCover, input="landCoverClass",
                   output="manningInteger", overwrite=True, quiet=True)
    
    ## divide by 10000.0 to convert manning data to decimal values
    pkg.mapcalc("$output=if(!isnull($manningInteger),$manningInteger/10000.0,null())",
                output="manning", manningInteger="manningInteger",
                overwrite=True, quiet=True)

    ## now calculate the unit head loss for each cell within the domain
    pkg.mapcalc("$output=if(!isnull($dem),($manning*$URConstant/$k)^2,null())",
                output="unitHeadLoss", dem="dem", manning="manning",
                URConstant=URConstant, k=k,
                overwrite=True, quiet=True)

def setMSLras(pkg, waterClass=11, minArea=20000000):
    '''
    Sets a mean (MSL) raster based on water clumps and specified parameters.

    This function takes a package/module used for executing GRASS GIS commands (`pkg`)
    and generates a mean sea level (MSL) raster. The MSL raster is created by
    identifying water cells, growing them by two cells to avoid disconnections,
    clumping the grown water cells, sorting the clumps by area, and then reclassifying
    the larger clumps as water areas. The resulting MSL raster is produced after
    shrinking the water extents back to the original state.

    Parameters:
    pkg: The package/module used for executing GRASS GIS commands.
    waterClass (int, optional): Land cover class value representing water bodies. Default is 11.
    minArea (int, optional): Minimum area (in square meters) for a water clump to be retained in the MSL raster. Default is 20000000.

    Returns:
    None: The function performs necessary operations to generate the modified sea level (MSL) raster.

    Note:
    - The input package `pkg` is expected to have methods for executing GRASS GIS commands.
    - The resulting MSL raster will be stored in the GRASS GIS environment.

    Example:
    setMSLras(grass_package, waterClass=10, minArea=1000000)
    '''
    ## define a raster with the MSL, all water cells are set to 0 othersiwe are set to null
    pkg.mapcalc(f"$output=if(landCoverClass == 0 | landCoverClass == {waterClass}, 0, null())",
                output = "water", dem = "dem", 
                overwrite=True, quiet=True)
    ## grow by two cells to avoid unnecesary disconnections
    pkg.run_command('r.grow', input = "water", output = "waterGrown1", radius=2.01,
                    overwrite=True, quiet=True)
    ## find clumps
    pkg.run_command('r.clump', input = "waterGrown1", output = "clumpmap",
                    overwrite=True, quiet=True)
    ## sort clumps based on area
    areas = pkg.read_command('r.stats', input="clumpmap", sort='desc',
                            flags = 'a', quiet=True).split()
    
    #Make a list of all areas larger than set minimum
    largeAreas = []
    minCells = minArea #Minimum area (meters^2) needed to create clump of water
    
    ## iterate through areas to get the the clumps above the minArea threshold
    for _ in areas:
        clumpID = areas.pop(0)
        clumpArea = areas.pop(0)
        if float(clumpArea) >= minCells and clumpID != "*":
            largeAreas.append(clumpID)
    
    ## format the large area IDs for use with r.reclass
    reclasslist=''
    for i in largeAreas:
        reclasslist += ("{0} = 0\n".format(i))

    ## reclassify all larger clumps as 0 for waterFinal raster
    pkg.write_command('r.reclass', input='clumpmap', output='waterReclass', rules='-', stdin=reclasslist,
                        overwrite=True, quiet=True)
    
    ## shrink water extents two cells to return to original pre-grown state
    pkg.mapcalc("$output=if(!isnull(waterReclass) && water == 0, 0, null())",
                output="waterFinal", overwrite=True, quiet=True)

def costSurface(res, pkg, slopeFactor=-0.2125, walkCoeefs=[0, 1, -1, -1], URConstant=1):
    '''
    Pre-computes a cost surface for pathfinding based on specified parameters.

    This function generates a pre-computed cost surface for pathfinding purposes using the specified parameters.
    It takes a resolution (`res`), a package/module for executing GRASS GIS commands (`pkg`), a slope factor,
    and walk coefficients as inputs. The cost surface is computed using the r.walk algorithm between each cell
    and specified starting points. The final cost values are updated by iteratively comparing new values to
    previously computed cost maps.

    Parameters:
    res (int): Resolution of the grid of points to run r.walk on.
    pkg: The package/module used for executing GRASS GIS commands.
    slopeFactor (float, optional): Slope factor for r.walk algorithm. Default is -0.2125.
    walkCoeefs (list of float, optional): Walk coefficients for r.walk algorithm. Default is [0, 1, -1, -1].
    URConstant (float, optional): Unit Runoff Constant for the Manning's equation. Default is 1.

    Returns:
    None: The function performs necessary operations to pre-compute the cost surface.

    Note:
    - The input package `pkg` is expected to have methods for executing GRASS GIS commands.
    - The resulting cost surface will be stored in the GRASS GIS environment.

    Example:
    preCompCostSurface(5, grass_package, slopeFactor=-0.2125, walkCoeefs=[0, 1, -1, -1])
    '''
    ## get region extent
    reg_extents = pkg.read_command('g.region', flags="t").split("/")
    xmin = float(reg_extents[0])
    xmax = float(reg_extents[1])
    ymin = float(reg_extents[2])
    ymax = float(reg_extents[3])

    ## get grid of points to run r.walk on
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    xgr, ygr = np.meshgrid(xs, ys)

    pkg.mapcalc("totalCost=50000",
                overwrite=True, quiet=True)
        
    for xVal, yVal in zip(xgr.ravel(), ygr.ravel()):
        
        ## set previous loop results to oldCostMap
        pkg.run_command('g.rename', raster="totalCost,oldCostMap",
                        overwrite=True, quiet=True)
        
        ## create cost surface between water and each set of xyCoords
        pkg.run_command('r.walk', elevation="dem", friction="unitHeadLoss",
                        output="walkValsFromLoop", start_raster="waterFinal",
                        stop_coordinates=f"{xVal},{yVal}", slope_factor=f"{slopeFactor}",
                        walk_coeff=f"{walkCoeefs[0]},{walkCoeefs[1]},{walkCoeefs[2]},{walkCoeefs[3]}",
                        overwrite=True, quiet=True)
            
        ## Where r.walk values exist from the current iteration, keep each cell value which is smaller than 
        ## all previous iterations.
        pkg.mapcalc("$newVal=if(!isnull($valFromLoop)&&&$valFromLoop<$oldVal,$valFromLoop,$oldVal)",
                    newVal="totalCost", valFromLoop="walkValsFromLoop", 
                    oldVal="oldCostMap", overwrite=True, quiet=True)
    
    pkg.mapcalc("$output=if(!isnull(dem), $totalCost, ($totalCost-$dem)/($URConstant^2))",
                output = "rawCost", totalCost = "totalCost",
                dem = "dem", URConstant=URConstant,
                overwrite=True, quiet=True)

def preCompCostSurface(grassVer, createGrassLocation, pathGrassLocation, pathRasFiles, rasterFiles, myepsg, manningRasPath, manningLandCover, 
                                    pathOutRawCostRas, pathOutTotalCostRas, nameGrassLocation = None, createLocMethod = 'from_raster', URConstant=1, 
                                    k=1, waterClass=11, minArea=20000000, res=5, slopeFactor=-0.2125, walkCoeefs=[0, 1, -1, -1]):
    '''
    Compute a cost surface for base on frictions (mannings) and elevation change.

    This function orchestrates various steps of the to generate
    a pre-computed cost surface for pathfinding purposes. It handles setting up the GRASS GIS
    environment, importing required raster files, setting Manning's raster, generating the
    mean sea level (MSL) raster, and computing the cost surface using specified parameters.

    Parameters:
    grassVer (str): Version of GRASS GIS to be used.
    createGrassLocation (bool): Whether to create a new GRASS GIS location.
    pathGrassLocation (str): Path to the GRASS GIS location.
    pathRasFiles (str): Path to the directory containing raster files.
    rasFiles (list of str): List of raster filenames, or raster file name in case of using 1.
    epsg (int): EPSG code for the coordinate reference system (CRS) of the location.
    manningRasPath (str): Path to the input raster containing Manning's n values.
    manningLandCover (str): Path to the rules file for reclassifying land cover classes to Manning's n values.
    pathOutRawCostRas (str): full path of the output raw cost raster 
    pathOutTotalCostRas (str): full path of the output total cost raster
    nameGrassLocation (str): Name of the GRASS GIS location. Default is None
    createLocMethod (str, optional): Two options "from_epsg" or "from_raster" (default) otherwise an error will be thrown.
    URConstant (float, optional): Unit Runoff Constant for the Manning's equation. Default is 1.
    k (float, optional): Manning's roughness coefficient. Default is 1.
    waterClass (int, optional): Land cover class value representing water bodies. Default is 11.
    minArea (int, optional): Minimum area (in square meters) for a water clump to be retained in the MSL raster. Default is 20000000.
    res (int, optional): Resolution of the grid of points for cost surface computation. Default is 5.
    slopeFactor (float, optional): Slope factor for cost surface computation. Default is -0.2125.
    walkCoeefs (list of float, optional): Walk coefficients for cost surface computation. Default is [0, 1, -1, -1].

    Returns:
    None: The function performs necessary operations to pre-compute the cost surface.

    Note:
    - The function assumes the logger is available for logging progress and information.
    - This function makes use of other utility functions: `grassEnvVar`, `setGrassEnv`, `setManning`,
      `setMSLras`, and `costSurface`.

    '''

    logger.info(f'Static downscaling started') # Changed
    t1 = time.time()
    
    grassEnvVar(grassVer)
   
    ## import grass
    import grass.script as gs
    import grass.script.setup as gsetup
    from grass.script import array as garray
    
    ## set path of grass location
    if nameGrassLocation == None:
        pathGrassLocation = os.path.join(pathGrassLocation, 'grassLoc')
    else:
        pathGrassLocation = os.path.join(pathGrassLocation, nameGrassLocation)

    logger.info(f'    Start Setup grass environment')
    
    if rasterFiles == 'all':
        rasterFiles = os.listdir(pathRasFiles)

    ## set grass environment
    setGrassEnv(grassVer, pathGrassLocation, createGrassLocation, gs, gsetup,
                pathRasFiles, rasterFiles, createLocMethod, myepsg)
    t2 = time.time()
    logger.info(f'    Setup grass environment: {(t2 - t1)/60:0.2f} min')
    
    ## set Manning's raster
    setManning(manningRasPath, manningLandCover, gs, URConstant, k)
    t3 = time.time()
    logger.info(f"    Set Manning's raster {(t3 - t2)/60:0.2f} min")

    ## set MSL raster
    setMSLras(gs, waterClass, minArea)
    t4 = time.time()
    logger.info(f"    Set MSL raster {(t4 - t3)/60:0.2f} min")

    logger.info(f"    Started the cost surface computation, may take a considerable time")
    costSurface(res, gs, slopeFactor, walkCoeefs, URConstant)
    gs.run_command('r.out.gdal', input = 'rawCost', flags = 'cm', format = 'GTiff', nodata = -9999, 
               output = pathOutRawCostRas, overwrite = True, quiet = True)
    gs.run_command('r.out.gdal', input = 'totalCost', flags = 'cm', format = 'GTiff', nodata = -9999, 
               output = pathOutTotalCostRas, overwrite = True, quiet = True)
    
    t5 = time.time()
    logger.info(f"    Compute cost surface {(t5 - t4)/60:0.2f} min")

def setupHeadLoss(kalpanaShp, attrCol, myepsg, rawCostRas, totalCostRas, pkg, exportOrg):
    '''
    Preprocesses Kalpana shape file and imports cost rasters for head loss analysis.

    This function preprocesses a Kalpana shape file containing ADCIRC results and imports associated
    cost rasters for head loss analysis. It converts the shape file to a raster, imports cost rasters,
    renames the imported rasters, and optionally exports the original ADCIRC raster without growing as a
    GeoTIFF for debugging purposes.

    Parameters:
    kalpanaShp (str): Path of the shape file with ADCIRC results.
    attrCol (str): Name of the attribute column in the shape file.
    myepsg (int): Output coordinate reference system (EPSG code).
    rawCostRas (str): Path of the raw cost raster.
    totalCostRas (str): Path of the total cost raster.
    pkg: The package/module used for executing GRASS GIS commands.
    exportOrg (bool, optional): True to export the raw ADCIRC outputs (without growing) as a GeoTIFF. Default is False.

    Returns:
    None: The function performs necessary preprocessing and imports for head loss analysis.

    Note:
    - The input package `pkg` is expected to have methods for executing GRASS GIS commands.
    - This function utilizes the utility function `importRasters_parallel`.
                
    '''
    ## import shape file with max water level
    t0 = time.time()
    pkg.run_command('v.in.ogr', input = kalpanaShp, overwrite = True,
                    quiet = True, snap = 0.000001, min_area = 10,
                    flags = 'o', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info(f'        Import kalpana shapefile: {(time.time() - t0)/60:0.2f} min')

    t0 = time.time()
    pkg.run_command('v.to.rast', input = os.path.basename(kalpanaShp[:-4]), 
                    type = 'area', output = 'kalpanaRast', use = 'attr', 
                    quiet = True, attribute_column = attrCol, overwrite = True)
    logger.info(f'        Kalpana shape to raster: {(time.time() - t0)/60:0.2f} min')

    ## import cost rasters
    t1 = time.time()
    importRasters_parallel([rawCostRas, totalCostRas], pkg, myepsg)
    logger.info(f'        Import cost rasters: {(time.time() - t1)/60:0.2f} min')

    ## rename files
    t2 = time.time()
    pkg.run_command('g.rename', raster = (os.path.splitext(os.path.basename(rawCostRas))[0], 'rawCost'), 
                        overwrite = True, quiet = True)
    pkg.run_command('g.rename', raster = (os.path.splitext(os.path.basename(totalCostRas))[0], 'totalCost'), 
                    overwrite = True, quiet = True)
    logger.info(f'        Rename cost rasters: {(time.time() - t2)/60:0.2f} min')
    
    ## update raw cost
    t3 = time.time()
    pkg.mapcalc("rawCost=if(isnull(rawCost), totalCost, rawCost)",
                overwrite=True, quiet=True)
    logger.info(f'        Update cost raster: {(time.time() - t3)/60:0.2f} min')

    if exportOrg == True: #export adcirc output without growing as tif
        t4 = time.time()
        pkg.run_command('r.out.gdal', input = 'kalpanaRast', flags = 'cm', format = 'GTiff', 
                        nodata = -9999, output = f'{kalpanaShp[:-4]}.tif', overwrite = True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f'        Export raw ADCIRC raster: {(time.time() - t4)/60:0.2f} min')

def headLossGrow(exagVal, floodDepth, pkg):
    '''
    Performs head loss analysis by extrapolating ADCIRC results and computing cost values.

    This function carries out head loss analysis by extrapolating ADCIRC results, creating cumulative raw cost rasters,
    and computing cost values based on the difference between extrapolated ADCIRC values and cost values. The function
    also handles scenarios with and without flood depth consideration.

    Parameters:
    exagVal (float): Exaggeration factor for hydraulic radius calculation.
    floodDepth (bool): True to consider flood depth, False otherwise.
    pkg: The package/module used for executing GRASS GIS commands.

    Returns:
    None: The function performs necessary head loss analysis operations.

    Note:
    - The input package `pkg` is expected to have methods for executing GRASS GIS commands.

    Example:
    headLossGrow(1.5, True, grass_package)
    '''
    ## extrapolate ADCIRC maxele output
    t0 = time.time()
    pkg.run_command("r.grow.distance", input='kalpanaRast', value=f"kalpanaRastGrownVal",
                    overwrite=True, quiet=True)
    logger.info(f'        Grown ADCIRC raw raster: {(time.time() - t0)/60:0.2f} min')

    t1 = time.time()
    ## creates raster containing the cumulative raw costs within the extents of ADCIRC results
    pkg.mapcalc("$output=if(!isnull($ADCIRC_WL),$rawCost,null())",
                output="rawCostADCIRC", ADCIRC_WL="kalpanaRast", rawCost="rawCost",
                overwrite=True, quiet=True)
    logger.info(f'        Create cum raw cost within ADCIRC extent: {(time.time() - t1)/60:0.2f} min')

    ## extrapolate cumulative raw cost values at edge of ADCIRC extent
    t2 = time.time()
    pkg.run_command("r.grow.distance", input="rawCostADCIRC", value="rawCostADCIRCVal",
                    overwrite=True, quiet=True)
    logger.info(f'        Grow ADCIRC raw raster: {(time.time() - t2)/60:0.2f} min')

    if floodDepth == False:
        ## if ADCIRC water levels are above the dem, calculate the cost of reaching each cell based on the difference
        ## between current window and ADCIRC raw costs using the average hydraulic radius.
        t3 = time.time()
        pkg.mapcalc('$output=if($ADCIRC_WLVal>$dem&&$ADCIRC_WLVal>($dem+$exag*($rawCost-$rawCostADCIRCVal)*(1/($ADCIRC_WLVal-0.5*$dem)^(2/3))^2),$ADCIRC_WLVal,null())',
                    output="forecastWLnotClumped", ADCIRC_WLVal="kalpanaRastGrownVal", dem="dem", exag=exagVal,
                    rawCost="rawCost", rawCostADCIRCVal="rawCostADCIRCVal",
                    overwrite=True, quiet=True)
        logger.info(f'        Compare extrapolated ADCIRC to cost : {(time.time() - t3)/60:0.2f} min')

    else:
        ## set the water levels to the difference between extrapolated ADCIRC values and cost values
        t4 = time.time()
        pkg.mapcalc('''$output=if($ADCIRC_WLVal>$dem&&$dem>0&&$ADCIRC_WLVal>($dem+$exag*($rawCost-$rawCostADCIRCVal)*
                    (1/($ADCIRC_WLVal-0.5*$dem)^(2/3))^2),$ADCIRC_WLVal-($dem+$exag*($rawCost-$rawCostADCIRCVal)*
                    (1/($ADCIRC_WLVal-0.5*$dem)^(2/3))^2),null())''',
                    output="forecastWLnotClumped_temp", ADCIRC_WLVal="kalpanaRastGrownVal", dem="dem",
                    exag=exagVal, rawCost="rawCost", rawCostADCIRCVal="rawCostADCIRCVal",
                    quiet=True, overwrite=True)

        ## temporary fix for depths where rawCost < rawCostADCIRCVal
        pkg.mapcalc('''$output=if(!isnull($forecastWLnotClumped_temp)&&$rawCostADCIRCVal>$rawCost,$ADCIRC_WLVal-$dem,
                    $forecastWLnotClumped_temp)''',
                    output="forecastWLnotClumped", forecastWLnotClumped_temp="forecastWLnotClumped_temp", dem="dem", 
                    rawCostADCIRCVal="rawCostADCIRCVal", rawCost="rawCost", ADCIRC_WLVal="kalpanaRastGrownVal", 
                    quiet=True, overwrite=True)
        logger.info(f'        Compare extrapolated ADCIRC to cost : {(time.time() - t4)/60:0.2f} min')
    
    t5 = time.time()
    pkg.mapcalc("$output=if(!isnull(forecastWLnotClumped), forecastWLnotClumped, kalpanaRast)",
            output="forecastWLnotClumpedCorr", quiet=True, overwrite=True)
    logger.info(f'        Combined raw ADCIRC with grown result : {(time.time() - t5)/60:0.2f} min')

def postProcessHeadLoss(floodDepth, kalpanaShp, pkg0, pkg1, ras2vec):
    '''
    Post-processes head loss results and exports them to files.

    This function performs post-processing on the head loss results, including correcting hydraulic connectivity,
    exporting results as GeoTIFF or ESRI Shapefiles, and optionally exporting water depth in flooded cells.

    Parameters:
    floodDepth (bool): True to consider flood depth, False otherwise.
    kalpanaShp (str): Path of the original Kalpana shape file.
    pkg0: The package/module used for executing GRASS GIS commands for raster operations.
    pkg1: The package/module used for executing GRASS GIS commands for vector operations.
    ras2vec (bool, optional): True to export results as ESRI Shapefiles, False otherwise. Default is True.

    Returns:
    None: The function performs necessary post-processing and exports.

    Note:
    - The input packages `pkg0` and `pkg1` are expected to have methods for executing GRASS GIS commands.

    Example:
    postProcessHeadLoss(True, "path/to/kalpana.shp", grass_package_raster, grass_package_vector, ras2vec=True)
    '''
    
    t0 = time.time()
    clumpingV2('kalpanaRast', 'forecastWLnotClumpedCorr', 'grownKalpanaRastLevel', pkg0, pkg1)
    logger.info(f'        correcting hydraulic connectivity: {(time.time() - t0)/60:0.3f} min')

    pathOut = os.path.dirname(kalpanaShp)
    fileOut = os.path.basename(kalpanaShp)[:-4]# remove extension

    t1 = time.time()
    pkg0.run_command('r.out.gdal', input = 'grownKalpanaRastLevel', flags = 'mc', format = 'GTiff', nodata = -9999, 
                    output = os.path.join(pathOut, f'{fileOut}_level_downscaled_headLoss.tif'), 
                    overwrite = True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info(f'        export as tif level: {(time.time() - t1)/60:0.3f} min')

    if ras2vec == True:
        t2 = time.time()
        ## export to ESRI shapefile
        pkg0.run_command('r.to.vect', input = 'grownKalpanaRastLevel', output = 'grownKalpanaVectLevel', type = 'area')
        logger.info(f'        ras to vector: {(time.time() - t2)/60:0.3f} min')
 
        t3 = time.time()
        pkg0.run_command('v.out.ogr', input = 'grownKalpanaVectLevel', 
                          output = os.path.join(pathOut, f'{fileOut}_level_downscaled'), 
                          type = 'area', format = 'ESRI_Shapefile', flags = 'se', quiet = True, overwrite = True)
        logger.info(f'        export as shp level: {(time.time() - t3)/60:0.3f} min')

    if floodDepth == True: ## export water depth in flooded cells
        t4 = time.time()
        pkg0.mapcalc("$output = if(!isnull($dem) && !isnull($grown), $grown - $dem, null())", 
                        grown = 'grownKalpanaRastLevel', dem = 'dem', output = 'grownKalpanaRastDepth', 
                        quiet = True, overwrite = True)
        logger.info(f'        compute water depth: {(time.time() - t4)/60:0.3f} min')
 
        t5 = time.time()
        ## export raster as tif
        pkg0.run_command('r.out.gdal', input = 'grownKalpanaRastDepth', flags = 'cm', format = 'GTiff', nodata = -9999, 
                       output = os.path.join(pathOut, f'{fileOut}_depth_downscaled_headLoss.tif'), 
                       overwrite = True)
        logger.info(f'        export as tif depth: {(time.time() - t5)/60:0.3f} min')
 
        if ras2vec == True:
            ## export to ESRI shapefile
            t6 = time.time()
            pkg0.run_command('r.to.vect', input = 'grownKalpanaRastDepth', output = 'grownKalpanaVectDepth', type = 'area')
            logger.info(f'        ras to shp depth: {(time.time() - t6)/60:0.3f} min')
 
            t7 = time.time()
            pkg0.run_command('v.out.ogr', input = 'grownKalpanaVectDepth', 
                                output = os.path.join(pathOut, f'{fileOut}_depth_downscaled_headLoss'), 
                                type = 'area', format = 'ESRI_Shapefile', flags = 'se', quiet = True, overwrite = True)
            logger.info(f'        export as shp depth: {(time.time() - t7)/60:0.3f} min') # 

def runHeadLoss(ncFile, levels, epsgOut, vUnitOut, pathOut, grassVer, pathRasFiles, rasterFiles,
                rawCostRas, totalCostRas, epsgIn=4326, vUnitIn='m', var='zeta_max', conType ='polygon', 
                subDomain=None, epsgSubDom=None, dzFile=None, zeroDif=-20, exagVal=1, nameGrassLocation=None, 
                createGrassLocation=True, createLocMethod='from_raster', attrCol='zMean', floodDepth=False, 
                ras2vec=False, exportOrg=False, leveesFile = None, finalOutToLatLon=True):
    '''
    Executes head loss downscaling process on ADCIRC results and generates downscaled maps.

    This function orchestrates the entire head loss downscaling process, starting from preprocessing
    ADCIRC results, running the downscaling procedure, post-processing the downscaled maps, and optionally
    exporting the final output to a different coordinate reference system (CRS).

    Parameters:
    ncFile (str): Path to the ADCIRC NetCDF file.
    levels (list of float): List of levels to process from the NetCDF file.
    epsgOut (int): Output EPSG code for the coordinate reference system (CRS) of the generated maps.
    vUnitOut (str): Output vertical unit of the generated maps.
    pathOut (str): Path to the output directory for saving results.
    grassVer (str): Version of GRASS GIS to be used.
    pathRasFiles (str): Path to the directory containing raster files.
    rasterFiles (list of str): List of raster filenames.
    rawCostRas (str): Path of the raw cost raster.
    totalCostRas (str): Path of the total cost raster.
    epsgIn (int, optional): Input EPSG code for the NetCDF file. Default is 4326.
    vUnitIn (str, optional): Input vertical unit of the NetCDF file. Default is 'm'.
    var (str, optional): Variable name in the NetCDF file. Default is 'zeta_max'.
    conType (str, optional): Type of connectivity for shapefile conversion. Default is 'polygon'.
    subDomain (str, optional): Path to the shapefile defining the sub-domain of interest. Default is None.
    epsgSubDom (int, optional): EPSG code of the sub-domain shapefile. Default is None.
    dzFile (str, optional): Path to the digital elevation model (DEM) file for corrections. Default is None.
    zeroDif (float, optional): Value for zero difference in digital elevation. Default is -20.
    exagVal (float, optional): Exaggeration factor for hydraulic radius calculation. Default is 1.
    nameGrassLocation (str, optional): Name of the GRASS GIS location. Default is None.
    createGrassLocation (bool, optional): Whether to create a new GRASS GIS location. Default is True.
    createLocMethod (str, optional): Method to create the GRASS GIS location. Default is 'from_raster'.
    attrCol (str, optional): Name of the attribute column in the shape file. Default is 'zMean'.
    floodDepth (bool, optional): True to consider flood depth, False otherwise. Default is False.
    ras2vec (bool, optional): True to export results as ESRI Shapefiles, False otherwise. Default is False.
    exportOrg (bool, optional): True to export the original ADCIRC raster without growing as a GeoTIFF. Default is False.
    leveesFile (str, optional): Path to the levees shapefile for additional analysis. Default is None.
    finalOutToLatLon (bool, optional): True to reproject the final output back to lat/lon CRS. Default is True.

    Returns:
    None: The function performs the entire head loss downscaling process.

    Note:
    - The function makes use of other utility functions: `nc2shp`, `grassEnvVar`, `setGrassEnv`,
      `setupHeadLoss`, `headLossGrow`, `postProcessHeadLoss`, `reprojectRas`.
    - The input package/module `grass_package_raster` and `grass_package_vector` are expected to have methods for executing GRASS GIS commands.
    '''

    t0 = time.time()
    pathaux = os.path.dirname(pathOut)
    
    if not os.path.exists(pathaux):
        os.mkdir(pathaux)

    gdf = nc2shp(ncFile, var, levels, conType, pathOut, epsgOut, vUnitOut, 
                    vUnitIn, epsgIn, subDomain, epsgSubDom, dzFile = dzFile, 
                    zeroDif = zeroDif)
    
    logger.info(f'Head loss downscaling started')

    grassEnvVar(grassVer)
    ## import grass
    import grass.script as gs
    import grass.script.setup as gsetup
    from grass.script import array as garray
    
    if nameGrassLocation == None:
        pathGrassLocation = os.path.join(pathaux, 'grassLoc')
    else:
        pathGrassLocation = os.path.join(pathaux, nameGrassLocation)

    
    t1 = time.time()

    if rasterFiles == 'all':
        rasterFiles = os.listdir(pathRasFiles)
    
    ## setup grass env
    setGrassEnv(grassVer, pathGrassLocation, createGrassLocation, gs, gsetup,
                pathRasFiles, rasterFiles, createLocMethod, epsgOut)
    logger.info(f'   Setup grass environment: {(time.time() - t1)/60:0.3f} min')

    ## preprocess for downscaling
    t2 = time.time()
    setupHeadLoss(pathOut, attrCol, epsgOut, rawCostRas, totalCostRas, gs, exportOrg)
    logger.info(f'    Downscaling preprocess: {(time.time() - t2)/60:0.3f} min')

    ## run downscaling
    t3 = time.time()
    headLossGrow(exagVal, floodDepth, gs)
    logger.info(f'    Running downscaling: {(time.time() - t3)/60:0.3f} min')

    ## posptrocess downscaled map
    t4 = time.time()
    postProcessHeadLoss(floodDepth, pathOut, gs, garray, ras2vec)
    logger.info(f'    Downscaling postprocessing: {(time.time() - t4)/60:0.3f} min')

    if finalOutToLatLon == True:
        t5 = time.time()
        finalOut = os.path.join(pathaux, os.path.basename(pathOut)[:-4] + '_level_downscaled_headLoss.tif')
        reprojectRas(finalOut, pathaux, epsgOut = 4326) 
        logger.info(f'    Downscaled level map reprojected back to lat/lon: {(time.time() - t5)/60:0.3f} min')

    logger.info(f'Kalpana finished sucsesfully after: {(time.time() - t0)/60:0.3f} min')
    logger.info(f'Output files saved on: {pathaux}')