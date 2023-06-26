import sys
import os
import glob
import shutil
import argparse 
import psycopg
import pandas as pd
from downscaling import meshRepLen2raster, runStatic
from loguru import logger

def getADCIRCFileNameVariables(modelRunID):
    ''' Returns DataFrame containing a list of variables (track_raw_fst, downloadurl, ADCIRCgrid, RunStartTime, advisory),
        extracted from table ASGS_Mon_config_item, in the asgs_dashboard DB, using the public.get_adcirc_filename_variables
        SQL function with modelRunID as input. These variables are used to construct filenames.
        Parameters
            modelRunID: string
                Unique identifier of a model run. It combines the instance_id, and uid from asgs_dashboard db
        Returns
            DataFrame
    '''

    try:
        # Create connection to database, set autocommit, and get cursor
        with psycopg.connect(dbname=os.environ['ASGS_DB_DATABASE'], user=os.environ['ASGS_DB_USERNAME'],
                             host=os.environ['ASGS_DB_HOST'], port=os.environ['ASGS_DB_PORT'],
                             password=os.environ['ASGS_DB_PASSWORD']) as conn:
            cur = conn.cursor()

            # Set enviromnent
            cur.execute("""SET CLIENT_ENCODING TO UTF8""")
            cur.execute("""SET STANDARD_CONFORMING_STRINGS TO ON""")

            # Run query
            cur.execute("""SELECT * FROM public.get_adcirc_filename_variables(_run_id := %(modelRunID)s);""",
                        {'modelRunID':modelRunID})

            # convert query output to Pandas dataframe
            df = pd.DataFrame.from_dict(cur.fetchall()[0], orient='columns')

            # Close cursor and database connection
            cur.close()
            conn.close()

            # Return Pandas dataframe
            return(df)

    # If exception log error
    except (Exception, psycopg.DatabaseError) as error:
        logger.info(error)

# This was added to run in k8s environment
@logger.catch
def main(args):
    ''' Takes argparse inputs and passes theme to the main function
        Parameters
            args: dictionary
                contains the parameters listed below.
            runScript: string
                Kalpana process to run: 
                    meshRepLen2raster, creates a grass location importing the DEM for downscaling and also creates a new DEM with
                    same resolution and extend with the size of the mesh triangles. This step is key for the downscaling and can
                    be run in advance, since only depends on the mesh.
                        # arguments common to both the meshRepLen2raster, and runStatic processes
                        epsgIn: string 
                            input epsg number
                        epsgOut: string 
                            output espg numnber
                        pathOut: string 
                            output directory path for shape file
                        grassVer: string 
                            grass version number
                        pathRasFiles: string
                            directory path to input raster files
                        rasterFiles: string
                            file name of input raster file
                        # arguments specific to the meshRepLen2raster process.
                        fort14: string
                              directory path and name of fort14 file. This variable is specific to the meshRepLen2raster process.
                        sbFile: string 
                              directory path and name of sbFile file.  This variable is specific to the meshRepLen2raster process.
                runStaticShort, runs static downscaling creating a grass location, and importing the DEM with the mesh elements size.
                    Both inputs were created by the meshRepLen2raster process. This process only requires modelRunID as input. All 
                    other input variables are defined in main. The modelRunID is used to get the grid name, using the function 
                    getADCIRCFileNameVariables
                        modelRunID: string
                            unique identifier of a model run. It combines the instance_id, and uid from asgs_dashboard db.
                runStatic, runs static downscaling creating a grass location, and importing the DEM with the mesh elements size.
                    Both inputs were created by the meshRepLen2raster process.
                        # arguments common to both the meshRepLen2raster, and runStatic processes
                        epsgIn: string   
                            input epsg number
                        epsgOut: string   
                            output espg numnber
                        pathOut: string   
                            output directory path for shape file
                        grassVer: string   
                            grass version number
                        pathRasFiles: string   
                            directory path to input raster files
                        rasterFiles: string   
                            file name of input raster file
                        # arguments specifid to the runStatic process
                        ncFile: string   
                            full path of the netCDF maxele file
                        conLevels: string   
                            contour levels to use in the downscaling
                        meshFile: string   
                            full path of the raster with the mesh element size
                        vUnitIn: string   
                            vertical unit of the maxele
                        vUnitOut: string   
                            vertical unit of the downscaled water levels
                        adcircVar: string   
                            name of the maxele variable to downscale. Always 'zeta_max' for downscaling
                        conType: string   
                            contours type. Always 'polygon' for downscaling
                        subDomain: string   
                            full path of file (kml, kmz, shp, gpkg or tif) to crop the domain
                        epsgSubDom: string   
                            epsg code or crs of the subDomain
                        exportMesh: string   
                            boolean for exporting the mesh as a shape file from maxel
                        dzFile: string   
                            full path of pickle file with vertical datum differences for all mesh nodes
                        zeroDif: string   
                            threshold to do apply the vertical datum difference, below -20 vyperdatum gives weird
                        nameGrassLocation: string   
                            full path of the grass location if a existing one will be used
                        createGrassLocation: string   
                            boolean for creating grass location
                        createLocMethod: string   
                            method for assigning the crs to the grass location
                        attrCol: string   
                            variable to downscale, can be 'zMax', 'zMean' and 'zMin'. With 'zMean', the mean value
                        repLenGrowing: string   
                            how many times the representative length the results are grown in the downscaling
                        compAdcirc2dem: string   
                            remove wet cells with water level below the ground surface
                        floodDepth: string   
                            transform the water level to water depth
                        clumpThreshold: string   
                            define clumpling threshold from mesh
                        perMinElemArea: string   
                            percentage of the minimum element area to scale the clumping threshold
                        ras2vec: string   
                            export downscaled results as shape files
                        exportOrg: string   
                            boolean for exporing raw maxele as a DEM
        Returns
            None
    '''         

    # Remove old logger and start new one
    logger.remove()
    log_path = os.path.join(os.getenv('LOG_PATH', os.path.join(os.path.dirname(__file__), 'logs')), '')
    logger.add(log_path+'kalpana.log', backtrace=True, diagnose=True)
    logger.add(sys.stdout, level="DEBUG")
    logger.add(sys.stderr, level="ERROR")

    # get input variables common to both meshRepLen2raster and runStatic from args
    runScript = args.runScript

    # check if runScript is meshRepLen2raster or runStatic
    if runScript == 'meshRepLen2raster':
        # variables not specific the meshRepLen2raster
        epsgIn = args.epsgIn
        epsgOut = args.epsgOut
        pathOut = args.pathOut
        grassVer = args.grassVer
        pathRasFiles = args.pathRasFiles
        rasterFiles = args.rasterFiles

        # if runScript is meshRepLen2raster get arguments specific to that process
        fort14 = args.fort14
        sbFile = args.sbFile

        #log start of meshRepLen2raster run 
        logger.info('Start meshRepLen2raster with the following inputs: '+runScript+', '+fort14+', '+epsgIn+', '+epsgOut+', '+pathOut+', '+grassVer+', '+pathRasFiles+', '+rasterFiles+", "+sbFile)

        # start meshRepLen2raster run
        meshRepLen2raster(fort14, epsgIn, epsgOut, pathOut, grassVer, pathRasFiles, rasterFiles,
                  subDomain=sbFile, nameGrassLocation=None, createGrassLocation=True,
                  createLocMethod='from_raster')

    elif args.runScript == 'runStaticShort':
        # define runStatic input variables
        modelRunID = args.modelRunID
        pathOut = '/data/'+modelRunID+'/kalpana/maxele.shp'
        ncFile = '/data/'+modelRunID+'/input/maxele.63.nc'
        conLevels = [float(i) for i in args.conLevels.split(',')]

        # Get ADCIRC filename variables
        df = getADCIRCFileNameVariables(modelRunID)
        grid = df['ADCIRCgrid'].values[0]
    
        if grid == 'NCSC_SAB_v1.23': 
            meshFile = '/data/kalpana/north_carolina/inputs/'+grid+'/NCSC123.tif'
            dzFile = '/data/kalpana/north_carolina/inputs/'+grid+'/NCSC_SAB_123_msl2navd88.pkl'
        elif grid  == 'hsofs':
            meshFile = '/data/kalpana/north_carolina/inputs/'+grid+'/HSOFS.tif'
            dzFile = '/data/kalpana/north_carolina/inputs/'+grid+'/HSOFS_msl2navd88.pkl'

        epsgIn = 4326
        epsgOut = 6543
        grassVer = '8.2'
        pathRasFiles = '/data/kalpana/north_carolina/inputs/'+grid+'/'
        rasterFiles = 'ncDEMs_epsg6543'
        conLevelsLog = "-".join(map(str, conLevels))
        vUnitIn = 'm'
        vUnitOut = 'm'
        adcircVar = 'zeta_max'
        conType = 'polygon'
        subDomain = '/data/kalpana/north_carolina/inputs/'+grid+'/ncDEMs_epsg6543'
        epsgSubDom = 6543
        exportMesh = False
        zeroDif = -20.0
        nameGrassLocation = 'grassLoc'
        createGrassLocation = True 
        createLocMethod = 'from_raster'
        attrCol = 'zMean'
        repLenGrowing = 1.0
        compAdcirc2dem = True
        floodDepth = False
        clumpThreshold = 'from_mesh'
        perMinElemArea = 1
        ras2vec = False
        exportOrg = False

        if grid == 'NCSC_SAB_v1.23' or grid == 'hsofs':
            logger.info('ncFile '+ncFile+' does use the '+grid+' grid, so begin processing')

            # Create outputs directory for second process shape, and tiff files
            outputDir = "/".join(pathOut.split('/')[0:-1])+'/'
            finalDir = "/".join(outputDir.split('/')[0:-2])+'/final/kalpana/'
            if not os.path.exists(outputDir):
                mode = 0o777
                os.makedirs(outputDir, mode, exist_ok=True)
                logger.info('Made directory '+outputDir)
            else:
                logger.info('Directory '+outputDir+' already made.')

            if not os.path.exists(finalDir):
                mode = 0o777
                os.makedirs(finalDir, mode, exist_ok=True)
                logger.info('Made directory '+finalDir)
            else:
                logger.info('Directory '+finalDir+' already made.')

            # log start of runStatic run
            logger.info('Start runScript with the following inputs: '+runScript+', '+str(epsgIn)+', '+str(epsgOut)+', '+pathOut+', '+grassVer+', '+ncFile+', '+meshFile+', '+conLevelsLog+', '+vUnitIn+', '+vUnitOut+', '+adcircVar+', '+conType+', '+str(subDomain)+', '+str(epsgSubDom)+', '+str(exportMesh)+', '+dzFile+', '+str(zeroDif)+', '+nameGrassLocation+', '+str(createGrassLocation)+', '+createLocMethod+', '+attrCol+', '+str(repLenGrowing)+', '+str(compAdcirc2dem)+', '+str(floodDepth)+', '+clumpThreshold+', '+str(perMinElemArea)+', '+str(ras2vec)+', '+str(exportOrg))

            # start runStatic run
            runStatic(ncFile, conLevels, epsgOut, pathOut,  grassVer, pathRasFiles, rasterFiles, meshFile,
                 epsgIn, vUnitIn, vUnitOut, adcircVar, conType, subDomain, epsgSubDom, exportMesh, dzFile, zeroDif,
                 nameGrassLocation, createGrassLocation, createLocMethod, attrCol, repLenGrowing,
                 compAdcirc2dem, floodDepth, clumpThreshold, perMinElemArea, ras2vec, exportOrg)

            # move cog tiff to final directory
            finalPathFile = glob.glob(outputDir+'*_epsg4326.tif')[0]
            logger.info('The length of the finalPathFile is: '+str(len(finalPathFile)))
            try:
                shutil.move(finalPathFile, finalDir)
                logger.info('Moved cog file '+finalPathFile+' to '+finalDir+' directory.')
                shutil.rmtree('/data/'+modelRunID+'/kalpana', ignore_errors=True)
                logger.info('Removed TIFF file /data/'+modelRunID+'/kalpana.') 
            except OSError as err:
                logger.error(err)
                sys.exit(1)
        else:
            logger.info('ncFile '+ncFile+' does not use the NCSC_SAB_v1.22 grid or the hsofs grid, it uses the '+grid+' grid, so do not process')

    elif args.runScript == 'runStatic':
        # variables not specific the runStatic
        epsgIn = args.epsgIn
        epsgOut = args.epsgOut
        pathOut = args.pathOut
        grassVer = args.grassVer
        pathRasFiles = args.pathRasFiles
        rasterFiles = args.rasterFiles

        # if runScript is runStatic get arguments specific to that process
        ncFile = args.ncFile
        conLevels = args.conLevels
        conLevelsLog = "-".join(map(str, args.conLevels))
        meshFile = args.meshFile
        vUnitIn = args.vUnitIn
        vUnitOut = args.vUnitOut
        adcircVar = args.adcircVar
        conType = args.conType

        # convert from string to bool
        if args.subDomain == 'None':
            subDomain = None
        else:
            subDomain = args.subDomain

        # convert from string to bool
        if args.epsgSubDom == 'None':
            epsgSubDom = None
        else:
            epsgSubDom = args.epsgSubDom

        # convert from string to bool
        if args.exportMesh == 'True':
            exportMesh = True
        elif args.exportMesh == 'False':
            exportMesh = False
        else:
            logger.info('exportMesh value has to be True or False')

        dzFile = args.dzFile
        zeroDif = float(args.zeroDif)
        nameGrassLocation = args.nameGrassLocation

        # convert from string to bool
        if args.createGrassLocation == 'True':
            createGrassLocation = True
        elif args.createGrassLocation == 'False':
            createGrassLocation = False
        else:
            logger.info('createGrassLocation value has to be True or False')

        createLocMethod = args.createLocMethod
        attrCol = args.attrCol
        repLenGrowing = float(args.repLenGrowing)

        # convert from string to bool
        if args.compAdcirc2dem == 'True':
            compAdcirc2dem = True
        elif args.compAdcirc2dem == 'False':
            compAdcirc2dem = False
        else:
            logger.info('compAdcirc2dem value has to be True or False')

        # convert from string to bool
        if args.floodDepth == 'True':
            floodDepth = True
        elif args.floodDepth == 'False':
            floodDepth = False
        else:
            logger.info('floodDepth value has to be True or False')

        clumpThreshold = args.clumpThreshold
        perMinElemArea = int(args.perMinElemArea)

        # convert from string to bool
        if args.ras2vec == 'True':
            ras2vec = True
        elif args.ras2vec == 'False':
            ras2vec = False
        else:
            logger.info('ras2vec value has to be True or False')

        # convert from string to bool
        if args.exportOrg == 'True':
            exportOrg = True
        elif args.exportOrg == 'False':
            exportOrg = False
        else:
            logger.info('exportOrg value has to be True or False')

        # Create outputs directory for second process shape, and tiff files
        outputDir = "/".join(pathOut.split('/')[0:-1])+'/'
        finalDir = "/".join(outputDir.split('/')[0:-2])+'/final/kalpana/'
        if not os.path.exists(outputDir):
            mode = 0o777
            os.makedirs(outputDir, mode, exist_ok=True)
            os.makedirs(finalDir, mode, exist_ok=True)
            logger.info('Made directories '+outputDir+ ' and '+finalDir+'.')
        else:
            logger.info('Directories '+outputDir+' and '+finalDir+' already made.')

        # log start of runStatic run
        logger.info('Start runScript with the following inputs: '+runScript+', '+epsgIn+', '+epsgOut+', '+pathOut+', '+grassVer+', '+ncFile+', '+meshFile+', '+conLevelsLog+', '+vUnitIn+', '+vUnitOut+', '+adcircVar+', '+conType+', '+str(subDomain)+', '+str(epsgSubDom)+', '+str(exportMesh)+', '+dzFile+', '+str(zeroDif)+', '+nameGrassLocation+', '+str(createGrassLocation)+', '+createLocMethod+', '+attrCol+', '+str(repLenGrowing)+', '+str(compAdcirc2dem)+', '+str(floodDepth)+', '+clumpThreshold+', '+str(perMinElemArea)+', '+str(ras2vec)+', '+str(exportOrg))

        # start runStatic run
        runStatic(ncFile, conLevels, epsgOut, pathOut,  grassVer, pathRasFiles, rasterFiles, meshFile,
             epsgIn, vUnitIn, vUnitOut, adcircVar, conType, subDomain, epsgSubDom, exportMesh, dzFile, zeroDif,
             nameGrassLocation, createGrassLocation, createLocMethod, attrCol, repLenGrowing,
             compAdcirc2dem, floodDepth, clumpThreshold, perMinElemArea, ras2vec, exportOrg)

        # create cog 
        # move cog tiff to final directory
        finalPathFile = glob.glob(outputDir+'*_epsg4326.tif')[0]
        try:
            shutil.move(finalPathFile, finalDir)
            logger.info('Created cog file '+finalPathFile+' and move to '+finalDir+' directory.')
        except OSError as err:
            logger.error(err)
            sys.exit(1)

if __name__ == "__main__":
    ''' Takes argparse inputs and passes theme to the main function
        Parameters
            runScript: string
                Kalpana process to run: 
                    meshRepLen2raster, creates a grass location importing the DEM for downscaling and also creates a new DEM with
                    same resolution and extend with the size of the mesh triangles. This step is key for the downscaling and can
                    be run in advance, since only depends on the mesh.
                        # arguments common to both the meshRepLen2raster, and runStatic processes
                        epsgIn: string 
                            input epsg number
                        epsgOut: string 
                            output espg numnber
                        pathOut: string 
                            output directory path for shape file
                        grassVer: string 
                            grass version number
                        pathRasFiles: string
                            directory path to input raster files
                        rasterFiles: string
                            file name of input raster file
                        # arguments specific to the meshRepLen2raster process.
                        fort14: string
                              directory path and name of fort14 file. This variable is specific to the meshRepLen2raster process.
                        sbFile: string 
                              directory path and name of sbFile file.  This variable is specific to the meshRepLen2raster process.
                runStaticShort, runs static downscaling creating a grass location, and importing the DEM with the mesh elements size.
                    Both inputs were created by the meshRepLen2raster process. This process only requires modelRunID as input. All 
                    other input variables are defined in main. The modelRunID is used to get the grid name, using the function 
                    getADCIRCFileNameVariables
                        modelRunID: string
                            unique identifier of a model run. It combines the instance_id, and uid from asgs_dashboard db.
                runStatic, runs static downscaling creating a grass location, and importing the DEM with the mesh elements size.
                    Both inputs were created by the meshRepLen2raster process.
                        # arguments common to both the meshRepLen2raster, and runStatic processes
                        epsgIn: string   
                            input epsg number
                        epsgOut: string   
                            output espg numnber
                        pathOut: string   
                            output directory path for shape file
                        grassVer: string   
                            grass version number
                        pathRasFiles: string   
                            directory path to input raster files
                        rasterFiles: string   
                            file name of input raster file
                        # arguments specifid to the runStatic process
                        ncFile: string   
                            full path of the netCDF maxele file
                        conLevels: string   
                            contour levels to use in the downscaling
                        meshFile: string   
                            full path of the raster with the mesh element size
                        vUnitIn: string   
                            vertical unit of the maxele
                        vUnitOut: string   
                            vertical unit of the downscaled water levels
                        adcircVar: string   
                            name of the maxele variable to downscale. Always 'zeta_max' for downscaling
                        conType: string   
                            contours type. Always 'polygon' for downscaling
                        subDomain: string   
                            full path of file (kml, kmz, shp, gpkg or tif) to crop the domain
                        epsgSubDom: string   
                            epsg code or crs of the subDomain
                        exportMesh: string   
                            boolean for exporting the mesh as a shape file from maxel
                        dzFile: string   
                            full path of pickle file with vertical datum differences for all mesh nodes
                        zeroDif: string   
                            threshold to do apply the vertical datum difference, below -20 vyperdatum gives weird
                        nameGrassLocation: string   
                            full path of the grass location if a existing one will be used
                        createGrassLocation: string   
                            boolean for creating grass location
                        createLocMethod: string   
                            method for assigning the crs to the grass location
                        attrCol: string   
                            variable to downscale, can be 'zMax', 'zMean' and 'zMin'. With 'zMean', the mean value
                        repLenGrowing: string   
                            how many times the representative length the results are grown in the downscaling
                        compAdcirc2dem: string   
                            remove wet cells with water level below the ground surface
                        floodDepth: string   
                            transform the water level to water depth
                        clumpThreshold: string   
                            define clumpling threshold from mesh
                        perMinElemArea: string   
                            percentage of the minimum element area to scale the clumping threshold
                        ras2vec: string   
                            export downscaled results as shape files
                        exportOrg: string   
                            boolean for exporing raw maxele as a DEM
        Returns
            None
    '''         

    # create argument parser from argparse
    parser = argparse.ArgumentParser()

    # arguments used by both meshRepLen2raster and runStatic processes
    parser.add_argument("--runScript", help="run script", action="store", dest="runScript", choices=['meshRepLen2raster','runStaticShort','runStatic'], required=True)

    # get runScript argument to use in if statement
    args = parser.parse_known_args()[0]

    if args.runScript == 'meshRepLen2raster':
        # arguments not specific to the meshRepLen2raster process
        parser.add_argument("--epsgIn", help="input epsg number", action="store", dest="epsgIn", required=True)
        parser.add_argument("--epsgOut", help="output espg numnber", action="store", dest="epsgOut", required=True)
        parser.add_argument("--pathOut", help="output directory path for shape file", action="store", dest="pathOut", required=True)
        parser.add_argument("--grassVer", help="grass version number", action="store", dest="grassVer", required=True)
        parser.add_argument("--pathRasFiles", help="directory path to input raster files", action="store", dest="pathRasFiles", required=True)
        parser.add_argument("--rasterFiles", help="file name of input raster file", action="store", dest="rasterFiles", required=True)

        # arguments specific to the meshRepLen2raster process
        parser.add_argument("--fort14", help="directory path and name of fort14 file", action="store", dest="fort14", required=True)
        parser.add_argument("--sbFile", help="directory path and name of sbFile file", action="store", dest="sbFile", required=True)
    elif args.runScript == 'runStaticShort':
        parser.add_argument("--modelRunID", help="the modelRunID, which is a combination of the instance_id and uid", action="store", dest="modelRunID", required=True)
        parser.add_argument("--conLevels", help="contour levels to use in the downscaling", action="store", dest="conLevels", required=True)
    elif args.runScript == 'runStatic':
        # arguments not specific to the meshRepLen2raster process
        parser.add_argument("--epsgIn", help="input epsg number", action="store", dest="epsgIn", required=True)
        parser.add_argument("--epsgOut", help="output espg numnber", action="store", dest="epsgOut", required=True)
        parser.add_argument("--pathOut", help="output directory path for shape file", action="store", dest="pathOut", required=True)
        parser.add_argument("--grassVer", help="grass version number", action="store", dest="grassVer", required=True)
        parser.add_argument("--pathRasFiles", help="directory path to input raster files", action="store", dest="pathRasFiles", required=True)
        parser.add_argument("--rasterFiles", help="file name of input raster file", action="store", dest="rasterFiles", required=True)

        # arguments specifid to the runStatic process
        parser.add_argument("--ncFile", help="full path of the netCDF maxele file", action="store", dest="ncFile", required=True)
        parser.add_argument("--conLevels", help="contour levels to use in the downscaling", type=lambda s: [int(item) for item in s.split(',')], action="store", dest="conLevels", required=True)
        parser.add_argument("--meshFile", help="full path of the raster with the mesh element size", action="store", dest="meshFile", required=True)
        parser.add_argument("--vUnitIn", help="vertical unit of the maxele", action="store", dest="vUnitIn", choices=['m'], required=True)
        parser.add_argument("--vUnitOut", help="vertical unit of the downscaled water levels", action="store", dest="vUnitOut", choices=['m', 'ft'], required=True)
        parser.add_argument("--adcircVar", help="name of the maxele variable to downscale. Always 'zeta_max' for downscaling", action="store", dest="adcircVar", choices=['zeta_max'], required=True)
        parser.add_argument("--conType", help="contours type. Always 'polygon' for downscaling", action="store", dest="conType", choices=['polygon'], required=True)
        parser.add_argument("--subDomain", help="full path of file (kml, kmz, shp, gpkg or tif) to crop the domain", action="store", dest="subDomain", required=False)
        parser.add_argument("--epsgSubDom", help="epsg code or crs of the subDomain", action="store", dest="epsgSubDom", required=False)
        parser.add_argument("--exportMesh", help="boolean for exporting the mesh as a shape file from maxel", action="store", dest="exportMesh", choices=['True','False'], required=True)
        parser.add_argument("--dzFile", help="full path of pickle file with vertical datum differences for all mesh nodes", action="store", dest="dzFile", required=True)
        parser.add_argument("--zeroDif", help="threshold to do apply the vertical datum difference, below -20 vyperdatum gives weird", action="store", dest="zeroDif", required=True)
        parser.add_argument("--nameGrassLocation", help="full path of the grass location if a existing one will be used", action="store", dest="nameGrassLocation", required=True)
        parser.add_argument("--createGrassLocation", help="Boolean for creating grass location", action="store", dest="createGrassLocation", choices=['True','False'], required=True)
        parser.add_argument("--createLocMethod", help="Method for assigning the crs to the grass location", action="store", dest="createLocMethod", required=True)
        parser.add_argument("--attrCol", help="variable to downscale, can be 'zMax', 'zMean' and 'zMin'. With 'zMean', the mean value", action="store", dest="attrCol", choices=['zMax','zMean','zMin'], required=True)
        parser.add_argument("--repLenGrowing", help="how many times the representative length the results are grown in the downscaling", action="store", dest="repLenGrowing", required=True)
        parser.add_argument("--compAdcirc2dem", help="remove wet cells with water level below the ground surface", action="store", dest="compAdcirc2dem", choices=['True','False'], required=True)
        parser.add_argument("--floodDepth", help="transform the water level to water depth", action="store", dest="floodDepth", choices=['True','False'], required=True)
        parser.add_argument("--clumpThreshold", help="define clumpling threshold from mesh", action="store", dest="clumpThreshold", required=True)
        parser.add_argument("--perMinElemArea", help="percentage of the minimum element area to scale the clumping threshold", action="store", dest="perMinElemArea", required=True)
        parser.add_argument("--ras2vec", help="export downscaled results as shape files", action="store", dest="ras2vec", choices=['True','False'], required=True)
        parser.add_argument("--exportOrg", help="boolean for exporing raw maxele as a DEM", action="store", dest="exportOrg", choices=['True','False'], required=True)

    # parse arguments and run main
    args = parser.parse_args()
    main(args)

