import fiona
import os
import sys
import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, LineString
from shapely.ops import split
from tqdm import tqdm
import itertools
import netCDF4 as netcdf
import simplekml
import warnings
warnings.filterwarnings("ignore")
import time
import dask
#import dask.array as da
#from tqdm.dask import TqdmCallback # Changed
#from vyperdatum.points import VyperPoints
from scipy.spatial import KDTree, distance
from itertools import islice
from pathlib import Path
import rioxarray as rxr
from loguru import logger # Changed

'''
    EXPLAIN WORKFLOW
'''

def gdfChangeVerUnit(gdf, ini, out):
    ''' Change vertical units of the columns of a GeoDataFrame which
        contains "z" on the name
        Parameters
            gdf: GeoDataFrame
                out of the runExtractContours function
            ini: string
                initial unit eg: 'm' or 'ft'
            out: string
                final unit eg: 'm' or 'ft'
        Returns
            gdf2: GeoDataFrame
                updated geodataframe with the new unit added on the name 
                to the columns with "z" in it.
    '''
    gdf2 = gdf.copy()
    ## select only columns with z in the name
    cols = [x for x in gdf2.columns if 'z' in x]
    ## from m to ft
    if ini == 'm' and out == 'ft':
        for col in cols:
            gdf2[col] = gdf2[col] * 3.2808399
    ## from ft to m
    elif ini == 'ft' and out == 'm':
        for col in cols:
            gdf2[col] = gdf2[col] / 3.2808399
    else:
        sys.exit('Only m to ft or vice versa')
    
    return gdf2

def dzDatum(vdatum_directory, x, y, pathout, vdatumIn='tss', 
            vdatumOut='navd88', epsg1=6319, epsg2=7912,
            areaFile="xGeoid20B_R1.gpkg", pkg=None):
    ''' Get the vertical difference of two datums on a group of locations
        Parameters
            vdatum_directory: string
                full path of the instalation folder of vdatum (https://vdatum.noaa.gov/)
            x, y: numpy array
                x and y-coordinate of the group of points
            vdatumIn, vdatumOut: string. Defaults, 'tss' and 'navd88'
                name of the input and output vertical datums. Mean sea level is "tss"
                For checking the available datums:
                from vyperdatum.pipeline import datum_definition
                list(datum_definition.keys())
            epsg1, epsg2: int. Defaults 6319 and 7912
                coordinate system code of the data. Points inside Chesapeake/Delaware bay needs to be
                converted using a different epsgs code (7912). 
            areaFile: str
                complete path of if a polygon geopackage file with the area of the Chesapeake/Delaware bay
                where the epgs2 code is needed.
        Returns
            df: dataframe
                dataframe with the vertical difference between datums of the requested points.     
    '''
    ## load area where the epsg 7912 is needed
    aux0 =__file__
    if sys.platform == 'win32':
        aux1 = aux0.split('\\')
        aux2 = '\\'.join(aux1[:-2])
    else:
        aux1 = aux0.split('/')
        aux2 = '/'.join(aux1[:-2])
    areaFile = os.path.join(aux2, 'adds', 'vyperDatum', 'chesapeake_delaware_bay_area', areaFile)
    
    pol = gpd.read_file(areaFile)
    aux = pointsInsidePoly(list(zip(x, y)), list(pol.geometry[0].exterior.coords))
    ## define vector with zeros to only get the difference between datums
    zcero = np.zeros(len(x))
    ## load vyperdatum
    ### Valid tidal area
    vp0 = pkg(vdatum_directory = vdatum_directory, silent = True)
    ## get the difference between vertical datums on the requested points
    vp0.transform_points((epsg1, vdatumIn), (epsg1, vdatumOut), x[~aux], y[~aux], z = zcero[~aux])
    ## call class
    vp1 = pkg(silent = True)
    ## get the difference between vertical datums on the requested points
    vp1.transform_points((epsg2, vdatumIn), (epsg1, vdatumOut), x[aux], y[aux], z = zcero[aux])   
    ## define dataframe
    df0 = pd.DataFrame({'x': x[~aux], 'y': y[~aux], 'dz': vp0.z, 'area': 0})
    df1 = pd.DataFrame({'x': x[aux], 'y': y[aux], 'dz': vp1.z, 'area': 1})
    df = pd.concat([df0, df1], axis = 0)
    df = df.dropna()
    df.index = range(len(df))
    df.to_pickle(pathout)
    
    return df
    
def changeDatum(x, y, z, var, dzFile, zeroDif=-20):
    ''' Change the vertical datum
        Parameters
            x, y, z: arrays
                coordinates of the points to change the datum
            var: array
                values to be transformed
            dzFile: str
                full path of the pickle file with the vertical difference between datums
                for each mesh node
            zeroDif: int
                threshold for using nearest neighbor interpolation to change datum. Points below
                this value won't be changed.
        Returns
            dfout: dataframe
                data transformed to the new datum
    '''
    dfdz = pd.read_pickle(dzFile)
    dfout = pd.DataFrame({'x': x, 'y': y, 'z': -1*z, 'var': var, 'dz': 0})
    tree = KDTree(list(zip(dfdz['x'], dfdz['y'])))
    query = tree.query(list(zip(dfout[dfout['z'] > zeroDif]['x'], dfout[dfout['z'] > zeroDif]['y'])))[1]
    dfout.loc[dfout[dfout['z'] > zeroDif].index, 'dz'] = dfdz['dz'].values[query]
    dfout['newVar'] = dfout['var'] + dfout['dz']
    
    return dfout

def classifyPolygons(polys):
    ''' Classify polygons based on signed area to get inner and outer polygons.
        Used in filledContours2gpd.
        Parameters
            polys: list
                list with pair of coordinates of the polygon vertices
        Returns
            outer, inner: list
                list with vertices of each type of polygon
    '''
    ## compute area using signedArea function
    areas = np.array([signedArea(p) for p in polys])
    ## get outer polygons where signed area is positive
    outer = [polys[x] for x in list(np.where(areas >= 0)[0])]
    ## get inner polygons where signed area is negative
    inner = [polys[x] for x in list(np.where(areas < 0)[0])]
    
    return outer, inner

def signedArea(ring):
    ''' Return the signed area enclosed by a ring in linear time using the algorithm at
        https://web.archive.org/web/20080209143651/http://cgafaq.info:80/wiki/Polygon_Area.
        Used in classifyPolygons.
        Parameters
            ring: list
                list with pairs of coordinates as tuples
        Returns
            signedArea: float
                area calculated using the vertices coordinates
    '''
    
    v2 = np.roll(ring, -1, axis=0)
    sArea = np.cross(ring, v2).sum() / 2.0
    
    return sArea

def pointsInsidePoly(points, polygon):
    ''' Get the subset of a cloud of points which are inside a polygon
        Used in filledContours2gpd.
        Parameters
            points: list
                list of pair of coordinates as tuples
            polygon: list
                list of coordinates of the polygon vertices as tuples
        Returns
            cont: array
                array with booleans
    '''
    p = mpl.path.Path(polygon)
    cont = p.contains_points(points)
    return cont
    
def checkTimeVarying(ncObj):
    ''' Check if an adcirc input is time-varying or not.
        Parameters
            ncObj: netCDF4._netCDF4.Dataset
                Adcirc input file
        Returns
            timeVar: int
                1 if time-varying, 0 if not
    '''
    if ncObj['time'].shape[0] <= 1:
        ## not time-varying
        timeVar = 0
    elif (ncObj['time'][-1].data - ncObj['time'][0].data).astype(int) == 0:
        ## time variable has lenght 2 but dates are the same --> not time-varying
        timeVar = 0
    else:
        ## time-varying file
        timeVar = 1
    
    return timeVar

def filledContours2gpd(tri, data, levels, epsg, step, orgMax, pbar=False):
    ''' Dataset to GeoDataFrame with filled contours as shapely polygons.
        A dask delayed python decorator is used to handle the code's parallelization.
        Parameters
            tri: mpl.tri.Triangulation
                triangulation of the adcirc mesh
            data: numpy array
                1D array with the data of the adcirc output variable
            levels: np.array
                Contour levels. The max value in the entire doman and over all timesteps is added to the requested levels.
            epsg: int
                coordinate system
            step: int or float
                step size of the levels requested
            orgMax: int or float
                max level requested
            pbar: boolean. Default False
                False for not displaying a progress bar with tqdm
        Returns
            gdf: GeoDataFrame
                Polygons as geometry and contours min, average, max values as columns. 
                The name of the variable, the date and other info are also included in the columns
    '''
    ## compute filled contours using matplotlib
    contoursf = plt.tricontourf(tri, data, levels = levels, extend = 'max')
    ## close the generated plot
    plt.close()
    
    ## define dask delayed function to transform the matplotlib object to shapely polygons
    @dask.delayed
    def getGeom(icoll, coll):
        ## get min and max value of the edge contours
        vmin, vmax = contoursf.levels[icoll:icoll+2]
        ## mean value of the polygon
        vmean = np.mean([vmin, vmax])
        geoms = []
        
        ## loop over polygon paths
        for p in coll.get_paths():
            p.simplify_threshold = 0.0
            # Removing polygons with less than 3 vertices
            polys = [g for g in p.to_polygons() if g.shape[0] >= 3] 
            # classify polygons based on the signed area
            outer, inner = classifyPolygons(polys)

            if len(inner) > 0:
                ## get first point of the inner polygon
                inner_points = [pts[0] for pts in inner]
            ## array of booleans
            overall_inout = np.zeros((len(inner),), dtype = bool)
            ## iteration through outer boundary
            for out in outer:
                if len(inner) > 0:
                    ## check which inner polygons are inside out
                    inout = pointsInsidePoly(inner_points, out)
                    ## update boolean array
                    overall_inout = np.logical_or(overall_inout, inout)
                    ## use boolean array to discard inner polygons
                    out_inner = [g for f, g in enumerate(inner) if inout[f]]
                    ## remove holes or inner polygons
                    poly = Polygon(out, out_inner)
                else: ## no inner polygons
                    poly = Polygon(out)

                # clean-up polygons (remove intersections)
                if not poly.is_valid:
                    poly = poly.buffer(0.0)
                if poly.is_empty:
                    continue

                geoms.append((poly, vmin, vmax, vmean))
            
            # collect all interiors which do not belong to any of the exteriors
            outer_interiors = [interior for s, interior in enumerate(inner) if not overall_inout[s]]
            for k in outer_interiors:
                poly = Polygon(k[::-1]) ## reverse geometrty
                # clean-up polygons (remove intersections)
                if not poly.is_valid:
                    poly = poly.buffer(0.0)
                if poly.is_empty:
                    continue
                geoms.append((poly, vmin, vmax, vmean))
        
        return geoms
    ## define tasks to call the delayed function
    tasks = [getGeom(icoll, coll) for icoll, coll in enumerate(contoursf.collections[:-1])]
    ## call dask without progress bar
    if pbar == False:
        daskGeoms = dask.compute(tasks, scheduler = 'threads')
    ## call dask with progress bar
    else:
        #with TqdmCallback(desc = "Compute contours using Dask"): # Changed
        logger.info('Begin computing contours using Dask') # Changed
        daskGeoms = dask.compute(tasks, scheduler = 'threads') # Changed
        logger.info('Finnished computing contours using Dask') # Changed
    ## dask output to list
    geoms = list(itertools.chain(*daskGeoms[0]))
    
    ## define geodataframe
    data = list(zip(*geoms))
    gdf = gpd.GeoDataFrame(crs = epsg, geometry = list(data[0]), 
                           index = range(len(data[0])))
    ## add extra columns
    gdf['zMin'] = np.clip(data[1], None, orgMax - step/2)
    gdf['zMean'] = np.clip(data[3], None, orgMax)
    gdf['zMax'] = np.clip(data[2], None, orgMax + step/2)
    
    return gdf

def contours2gpd(tri, data, levels, epsg, pbar=False):
    ''' Dataset to GeoDataFrame with contours as shapely LineStrings
        A dask delayed python decorator is used to handle the code's parallelization.
        Parameters
            tri: mpl.tri.Triangulation
                triangulation of the adcirc mesh
            data: numpy array
                1D array with the data of the adcirc output variable
            levels: np.array
                Contour levels. The max value in the entire doman and over all timesteps is added to the requested levels.
            epsg: int
                coordinate system
            pbar: boolean. Default False
                False for not displaying a progress bar with tqdm
        Returns
            gdf: GeoDataFrame
                Linestrings as geometry and contour value in the "value" column.
                The name of the variable, the date and other info are also included in the columns
    '''
    ## compute non-filled contours using matplotlib
    contours = plt.tricontour(tri, data, levels = levels, extend = 'max')
    plt.close()
    ## define dask delayed function to transform the matplotlib object to shapely polygons
    @dask.delayed
    def getGeom(ic, c):
        
        ## get value of contour
        val = contours.levels[ic]
        ## get contour
        paths = c.get_paths()
    
        ## Path to linestring. Dismiss lines with less than 3 vertices
        aux0 = [(LineString(path.vertices), val) for path in paths if len(path.vertices) > 2]
        
        return aux0
    
    ## define tasks to call the delayed function
    tasks = [getGeom(icoll, coll) for icoll, coll in enumerate(contours.collections)]
    ## call dask without progress bar
    if pbar == False:
        daskGeoms = dask.compute(tasks, scheduler = 'threads')
    ## call dask with progress bar
    else:
        #with TqdmCallback(desc = "Compute contours using Dask"): # Changed
        logger.info('Begin computing contours using Dask') # Changed
        daskGeoms = dask.compute(tasks, scheduler = 'threads') # Changed
        logger.info('Finish computing contours using Dask') # Changed

    geoms = list(itertools.chain(*daskGeoms[0]))
    
    ## define geopandas
    data = list(zip(*geoms))
    gdf = gpd.GeoDataFrame(crs = epsg, geometry = list(data[0]), 
                           index = range(len(data[0])))
   
   ## add extra columns
    gdf['z'] = data[1]#np.clip(data[1], None, orgMax)
    
    return gdf

def runExtractContours(ncObj, var, levels, conType, epsg, stepLevel, orgMaxLevel, dzFile=None, zeroDif=-20, timesteps=None):
    ''' Run "contours2gpd" or "filledContours2gpd" if npro = 1 or "contours2gpd_mp" or "filledContours2gpd_mp" if npro > 1.
        Parameters
            ncObj: netCDF4._netCDF4.Dataset
                Adcirc input file
            var: string
                Name of the variable to export
            levels: np.array
                Contour levels. The max value in the entire doman and over all timesteps is added to the requested levels.
            conType: string
                'polyline' or 'polygon'
            epsg: int
                coordinate system
            stepLevel: int or float
                step size of the levels requested
            orgMaxLevel: int or float
                max level requested
            dzFile: str
                full path of the pickle file with the vertical difference between datums
                for each mesh node
            zeroDif: int
                threshold for using nearest neighbor interpolation to change datum. Points below
                this value won't be changed.
            timesteps: numpy array. Default None
                timesteps to extract if the ncObj is a time-varying ADCIRC output file. If None, all time steps are exported
        Returns
            gdf: GeoDataFrame
                Polygons or polylines as geometry columns. If the requested file is time-varying the GeoDataFrame will include all timesteps.
            
    '''
    ## get triangles and nodes coordinates
    nv = ncObj['element'][:,:] - 1 ## triangles starts from 1
    x = ncObj['x'][:].data
    y = ncObj['y'][:].data
    z = ncObj['depth'][:].data
    
    ## get extra info: variable name, variable long-name and unit name
    vname = ncObj[var].name
    lname = ncObj[var].long_name
    #u = ncObj[var].units

    ## matplotlib triangulation
    tri = mpl.tri.Triangulation(x, y, nv)
    
    ## check if the time is time-varying
    timeVar = checkTimeVarying(ncObj)
    ## if the variable requested is the bathymetry, values are inverted (times -1) for plotting
    if var == 'depth':
        timeVar = 0
        auxMult = -1
    else:
        auxMult = 1
    
    ## time constant
    if timeVar == 0:
        aux = ncObj[var][:].data
        if dzFile != None: ## change datum
            dfNewDatum = changeDatum(x, y, z, aux, dzFile, zeroDif)
            ## change nan to -99999 and transform it to a 1D vector
            aux = np.nan_to_num(dfNewDatum['newVar'].values, nan = -99999.0).reshape(-1)*auxMult
        else: ## original datum remains constant
            ## change nan to -99999 and transform it to a 1D vector
            aux = np.nan_to_num(aux, nan = -99999.0).reshape(-1)*auxMult
        ## non-filled contours
        if conType == 'polyline':
            labelCol = 'z'
            gdf = contours2gpd(tri, aux, levels, epsg, True)
        ## filled contours
        elif conType == 'polygon':
            labelCol = 'zMean'
            gdf = filledContours2gpd(tri, aux, levels, epsg, stepLevel, orgMaxLevel, True)
        ## error message
        else:
            logger.info('only "polyline" and "polygon" types are supported!') # Changed
            sys.exit(-1)
        ## add more info to the geodataframe
        gdf['variable'] = [vname]*len(gdf)
        gdf['name'] = [lname]*len(gdf)
        #gdf['zLabelCol'] = [f'{x:0.2f} {unit}' for x in gdf[labelCol]]
    ## time varying
    else:
        ## get epoch
        if timesteps is None:
            timesteps = range(ncObj['time'].shape[0])
        t0 = pd.to_datetime(ncObj['time'].units.split('since ')[1])
        listGdf = []
        ## non-filled contours
        if conType == 'polyline':
            labelCol = 'z'
            ## time loop
            for t in tqdm(timesteps):
                ## data to 1D non masked array
                aux = ncObj[var][t, :].data
                if dzFile != None: ## change datum
                    dfNewDatum = changeDatum(x, y, z, aux, dzFile, zeroDif)
                    ## change nan to -99999 and transform it to a 1D vector
                    aux = np.nan_to_num(dfNewDatum['newVar'].values, nan = -99999.0).reshape(-1)*auxMult
                else: ## original datum remains constant
                    ## change nan to -99999 and transform it to a 1D vector
                    aux = np.nan_to_num(aux, nan = -99999.0).reshape(-1)*auxMult
                gdfi = contours2gpd(tri, aux, levels, epsg, False)
                ## add extra info to the gdf
                ti = pd.Timedelta(seconds = int(ncObj['time'][t]))
                gdfi['nTimeStep'] = [t]*len(gdfi)
                gdfi['date'] = [str(t0 + ti)]*len(gdfi)
                gdfi['nHours'] = [ti.total_seconds()/3600]*len(gdfi)
                listGdf.append(gdfi)
        ## filled contours
        elif conType == 'polygon':
            labelCol = 'zMean'
            ## time loop
            for t in tqdm(timesteps):
                ## data to 1D non masked array
                aux = ncObj[var][t, :].data
                if dzFile != None: ## change datum
                    dfNewDatum = changeDatum(x, y, z, aux, dzFile, zeroDif)
                    ## change nan to -99999 and transform it to a 1D vector
                    aux = np.nan_to_num(dfNewDatum['newVar'].values, nan = -99999.0).reshape(-1)*auxMult
                else: ## original datum remains constant
                    ## change nan to -99999 and transform it to a 1D vector
                    aux = np.nan_to_num(aux, nan = -99999.0).reshape(-1)*auxMult
                gdfi = filledContours2gpd(tri, aux, levels, epsg, stepLevel, orgMaxLevel, False)
                ## add extra info to the gdf
                ti = pd.Timedelta(seconds = int(ncObj['time'][t]))
                gdfi['nTimeStep'] = [t]*len(gdfi)
                gdfi['date'] = [str(t0 + ti)]*len(gdfi)
                gdfi['nHours'] = [ti.total_seconds()/3600]*len(gdfi)
                listGdf.append(gdfi)
        ## error
        else:
            logger.info('only "polyline" and "polygon" types are supported!') # Changed
            sys.exit(-1)
        
        ## define output geodataframe
        gdf = gpd.GeoDataFrame(pd.concat(listGdf, axis=0, ignore_index=True), crs=epsg)
        gdf['variable'] = [vname]*len(gdf)
        gdf['name'] = [lname]*len(gdf)
        #gdf['zlabelCol'] = [f'{x:0.2f} {uinit}' for x in gdf[labelCol]]
        
    return gdf

def mesh2gdf(ncObj, epsgIn, epsgOut):
    ''' Write adcirc mesh from netcdf as GeoDataFrame and extract centroid of each element. Used to create submesh
        and for the downscaling process
        Parameters:
            ncObj: netCDF4._netCDF4.Dataset
                adcirc file
            epsgIn: int
                coordinate system of the adcirc input
            epsgOut: int
                coordinate system of the output shapefile
        Returns
            gdf: GeoDataFrame
                GeoDataFrame with polygons as geometry and centroid coordinates in columns
        
    '''
    ## get triangles and nodes coordinates
    nv = ncObj['element'][:,:] - 1 ## triangles starts from 1
    x = ncObj['x'][:].data
    y = ncObj['y'][:].data
    ## matplotlib triangulation
    tri = mpl.tri.Triangulation(x, y, nv)
    ## get the x and y coordinate of the triangle elements in the right order
    xvertices = x[tri.triangles[:]]
    yvertices = y[tri.triangles[:]]
    ## add x and y togheter
    listElem = np.stack((xvertices, yvertices), axis = 2)
    ## define polygons
    pols = [Polygon(x) for x in listElem]
    ## define geodataframe
    gdf = gpd.GeoDataFrame(geometry = pols, crs = epsgIn)
    ## change crs if epsgIn and epsgOut are different
    if epsgIn == epsgOut:
        pass
    else:
        gdf = gdf.to_crs(epsgOut)
    ## add extra information to the mesh gdf, coordinates of the centroids and ID of each vertex
    gdf['centX'] = xvertices.mean(axis = 1)
    gdf['centY'] = yvertices.mean(axis = 1)
    gdf['v1'] = nv.data[:, 0]
    gdf['v2'] = nv.data[:, 1]
    gdf['v3'] = nv.data[:, 2]
    
    ## area and representative length are not computed if latlon epsg is requested
    if epsgOut == 4326: 
        pass
    else:
        ## representative length is 1/3 of the perimeter
        gdf['repLen'] = [np.round(geom.length/3, 3) for geom in gdf.geometry]
        gdf['elemArea'] = [np.round(geom.area, 3) for geom in gdf.geometry]
    
    return gdf
    
def readSubDomain(subDomain, epsg):
    ''' Read a user specified subdomain and transform it to GeoDataFrame
        Parameters
            subDomain: str or list
                complete path of the subdomain polygon kml or shapelfile, or list with the
                uper-left x, upper-left y, lower-right x and lower-right y coordinates
            epsg: int
                epsg code of the used system of coordinates
        Returns
            gdfSubDomain: GeoDataFrame
                user specified subdomain as geodataframe
    '''
    if type(subDomain) == str:
        ## kml or shp file
        if subDomain.endswith('.shp') or subDomain.endswith('.gpkg'):
            gdfSubDomain = gpd.read_file(subDomain)
            ## get exterior coordinates of the polygon
            xAux, yAux = gdfSubDomain.geometry[0].exterior.coords.xy
            extCoords = list(zip(xAux, yAux))
            poly = Polygon(extCoords)
            gdfSubDomain = gpd.GeoDataFrame(geometry = [poly], crs = epsg)
        
        elif subDomain.endswith('.kml'):
            gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
            gdfSubDomain = gpd.read_file(subDomain, driver = 'KML')
            ## get exterior coordinates of the polygon
            xAux, yAux = gdfSubDomain.geometry[0].exterior.coords.xy
            extCoords = list(zip(xAux, yAux))
            poly = Polygon(extCoords)
            gdfSubDomain = gpd.GeoDataFrame(geometry = [poly], crs = epsg)
        
        else: ## try read DEMs
            try:
                r = rxr.open_rasterio(subDomain)
                bbox = r.rio.bounds()
                ulLon, ulLat, lrLon, lrLat = bbox[0], bbox[3], bbox[2], bbox[1]
                extCoords = [(ulLon, ulLat), (ulLon, lrLat), (lrLon, lrLat), (lrLon, ulLat), (ulLon, ulLat)]
                poly = Polygon(extCoords)
                gdfSubDomain = gpd.GeoDataFrame(geometry = [poly], crs = epsg)

            except:
                logger.info('Only shape, geopackage, kml formats and rasters are suported for sub domain generation!') # Changed
                sys.exit(-1)            
    
    elif type(subDomain) == list and len(subDomain) == 4:
        ## only UL lon, UL lat, LR lon LR lat
        ulLon, ulLat, lrLon, lrLat = subDomain
        ## define list with exterior coordinates
        extCoords = [(ulLon, ulLat), (ulLon, lrLat), (lrLon, lrLat), (lrLon, ulLat), (ulLon, ulLat)]
        ## define shapely polygon
        poly = Polygon(extCoords)
        ## define gdf
        gdfSubDomain = gpd.GeoDataFrame(geometry = [poly], crs = epsg)
    else:
        logger.info('subDomain must be the path of a kml or shapefile, or a list with the coordinates of the upper left and lower right corners of a box') # Changed
        sys.exit(-1)
    
    return gdfSubDomain
    
def daNcSubset(ncObj, epsg, subDom):
    ''' Extract subset of netCDF file using a GeoDataFrame as subdomain
        Parameters
            ncObj: netCDF4._netCDF4.Dataset
                Adcirc input file
            epsg: int
                coordinate system
            subDom: str or list
                complete path of the subdomain polygon kml or shapelfile, or list with the
                uper-left x, upper-left y, lower-right x and lower-right y coordinates
        Returns
            mind: dask.array.core.Array
                matrix with triangles
            nodesInside: array
                index of the ncObj nodes inside subDom
    '''
    ## read subdomain
    subDomain = readSubDomain(subDom, epsg)
    ## read mesh as GeoDataFrame
    mesh = mesh2gdf(ncObj, epsg)
    ## exterior coordinates of the subdomain polygon
    xAux, yAux = subDomain.geometry[0].exterior.coords.xy
    extCoords = list(zip(xAux, yAux))
    ## centroid of each mesh triangle
    centroids = list(zip(mesh.centX, mesh.centY))
    ## find centroids inside subdomain
    inside = pointsInsidePoly(centroids, extCoords)
    centInside = np.where(inside)[0]
    
    ## mesh subset
    meshSub = mesh.iloc[centInside, :]
    ## select nodes inside subdomain
    nodesInside = meshSub.loc[:, ['v1', 'v2', 'v3']].values.reshape(-1)
    nodesInside = np.unique(nodesInside)
    aux = list(nodesInside)
#     vertices = list(meshSub[['v1', 'v2', 'v3']].to_records(index=False))
    ## aux to matrix to avoid for loop
    auxm = np.broadcast_to(aux, (len(meshSub), len(aux)))
    ## numpy to dask array to allow parallel computing
    dauxm = da.array(auxm)
    ## iteration over each node of all triangles
    indlist = []
    for iv, v in enumerate([meshSub.v1.values, meshSub.v2.values, meshSub.v3.values]):
        ## vertices as matrix
        vm =  np.broadcast_to(v, (len(aux), len(v))).T
        ## numpy to dask array to allow parallel computing
        dvm = da.array(vm)
        ## absolute value of the difference between the node id and the list 
        ## of all nodes id
        vind = da.fabs(da.subtract(dvm, dauxm))
        ## argmin to find the index of each node in the list of all nodes to make the list
        ## of triangles
        vind = da.argmin(vind, axis = 1)
        indlist.append(vind)
    ## merge as one matrix
    mind = da.concatenate(indlist)
    mind = mind.reshape((3, len(v)))
#     xNodesInside = ncObj['x'][nodesInside].data
#     yNodesInside = ncObj['y'][nodesInside].data
#     tri = mpl.tri.Triangulation(xNodesInside, yNodesInside, mind.T) 
    return mind, nodesInside
    
def nc2xr(ncFile, var):
    ''' Write netcdf as xarray dataset. May be redundant but I had some problems reading adcirc files with xarray. NOT USED
        Xarray is faster and easier to work with than netCDF4 object.
        Parameters
            ncFile: string
                complete path of the adcirc input
            var: string
                name of the variable to work with
        Returns
            ds: xarray dataset
                netCDF4 object casted as xarray dataset
    '''
    with netcdf.Dataset(ncFile, 'r') as ncObj:
    
        epoch = pd.to_datetime(ncObj['time'].units.split('since ')[1])
        dates = [epoch + pd.Timedelta(seconds = float(x)) for x in ncObj['time'][:]]
    
        if checkTimeVarying(ncObj) == 1:
            dummy1 = ncObj[var][:,:].data
            nnodes = range(ncObj[var].shape[1])
            dims = ['time', 'node']
            # coords = {'time': ncObj['time'][:].data, 'node': nnodes}
            coords = {'time': dates, 'node': nnodes}
        else:
            dummy1 = ncObj[var][:].data
            dummy1 = dummy1.reshape((1, len(dummy1)))
            nnodes = range(ncObj[var].shape[0])
            dims = ['node']
            coords = {'time': [0], 'node': nnodes}            
        
        dummy2 = np.ma.masked_invalid(dummy1)
        dummy3 = dummy2.filled(fill_value = -99999.0)
        
        ds = xr.Dataset({
                        var:xr.DataArray(data = dummy3,
                                          dims = ['time', 'node'],
                                          coords = coords),
                       'element': xr.DataArray(data = ncObj['element'][:, :].data,
                                              dims = ['nele', 'nvertex']),
                       'x': xr.DataArray(data = ncObj['x'][:].data,
                                         dims = ['node'],
                                         coords = {'node': nnodes}),
                       'y': xr.DataArray(data = ncObj['y'][:].data,
                                         dims = ['node'],
                                         coords = {'node': nnodes})                                     
                        })
        
        ds[var].attrs['long_name'] = ncObj[var].long_name.capitalize()
        ds[var].attrs['units'] = ncObj[var].units
        # ds['time'].attrs['base_date'] = pd.to_datetime(ncObj['time'].units.split('since ')[1])
        # ds['time'].attrs['units'] = f'Seconds since {epoch}'
        
    return ds
    
def nc2shp(ncFile, var, levels, conType, pathOut, epsgOut, vUnitOut='ft', vUnitIn='m', epsgIn=4326,
           subDomain=None, epsgSubDom=None, exportMesh=False, meshName=None, dzFile=None, zeroDif=-20, timesteps=None):
    ''' Run all necesary functions to export adcirc outputs as shapefiles.
        Parameters
            ncFile: string
                path of the adcirc output, must be a netcdf file
            var: string
                Name of the variable to export
            levels:list
                Contour levels. Min, Max and Step. Max IS included as in np.arange method.
                Values must be in vUnitOut vertical unit.
            conType: string
                'polyline' or 'polygon'
            pathout: string
                complete path of the output file (*.shp or *.gpkg)
            epsgOut: int
                coordinate system of the output shapefile
            vUnitIn, vUnitOut: string. Default for vUnitIn is 'm' and 'ft' for vUnitOut
                input and output vertical units. For the momment only supported 'm' and 'ft'
            epsgIn: int. Default 4326.
                coordinate system of the adcirc input
            subDomain: str or list. Default None
                complete path of the subdomain polygon kml or shapelfile, or list with the
                uper-left x, upper-left y, lower-right x and lower-right y coordinates. The crs must be the same of the
                adcirc input file.
            exportMesh: boolean. Default False
                True for export the mesh geodataframe and also save it as a shapefile
            meshName: str
                file name of the output mesh shapefile
            dzFile: str
                full path of the pickle file with the vertical difference between datums
                for each mesh node
            zeroDif: int
                threshold for using nearest neighbor interpolation to change datum. Points below
                this value won't be changed.
            timesteps: numpy array. Default None
                timesteps to extract if the ncObj is a time-varying ADCIRC output file. If None, all time steps are exported
        Returns
            gdf: GeoDataFrame
                gdf with contours
            mesh: GeoDataFrame, only if exportMesh is True
                gdf with mesh elements, representative length and area of each triangle
    '''
    
    logger.info('Start exporting adcirc to shape') # Changed
    ## read adcirc file
    nc = netcdf.Dataset(ncFile, 'r')
    ## change units of the requested levels
    if vUnitIn == 'm' and vUnitOut == 'ft':
        levels = [l / 3.2808399 for l in levels]
    elif vUnitIn == 'ft' and vUnitOut == 'm':
        levels = [l * 3.2808399 for l in levels]
    if conType == 'polygon':
        maxmax = np.max(nc[var][:].data)
        orgMaxLevel = levels[1]

        if maxmax < orgMaxLevel:
            maxmax = orgMaxLevel
 
        stepLevel = levels[2]
        ## list of levels to array
        levels_aux = np.arange(levels[0], np.ceil(maxmax) + 2*stepLevel, stepLevel)
        ## given levels will now match the avarege value of each interval    
        levels_aux = levels_aux - stepLevel/2
        levels = levels_aux.copy()
    else:
        orgMaxLevel = levels[1]
        stepLevel = levels[2]
        levels = np.arange(levels[0], orgMaxLevel + stepLevel, stepLevel)
    
    t00 = time.time()
    gdf = runExtractContours(nc, var, levels, conType, epsgIn, stepLevel, orgMaxLevel, 
                            dzFile, zeroDif, timesteps)
    logger.info(f'    Ready with the contours extraction: {(time.time() - t00)/60:0.3f} min') # Changed
 
    ## clip contours if requested
    if subDomain is not None:
        t0 = time.time()
        subDom = readSubDomain(subDomain, epsgSubDom)
        gdf = gpd.clip(gdf, subDom.to_crs(epsgIn))
        logger.info(f'    Cliping contours based on mask: {(time.time() - t0)/60:0.3f} min') # Changed
 
    ## change vertical units if requested
    if vUnitIn == vUnitOut:
        pass
    else:
        t0 = time.time()
        gdf = gdfChangeVerUnit(gdf, vUnitIn, vUnitOut)
        logger.info(f'    Vertical units changed: {(time.time() - t0)/60:0.3f} min') # Changed
    ## change CRS if requested
    if epsgIn == epsgOut:
        pass
    else:
        t0 = time.time()
        gdf = gdf.to_crs(epsgOut)
        logger.info(f'    Changing CRS: {(time.time() - t0)/60:0.3f} min') # Changed
    ## save output shape file
    t0 = time.time()
    if pathOut.endswith('.shp'):
        gdf.to_file(pathOut)
    elif pathOut.endswith('.gpkg'):
        gdf.to_file(pathOut, driver = 'GPKG')
    elif pathOut.endswith('.wkt'):
        gdf.to_csv(pathOut)
    logger.info(f'    Saving file: {(time.time() - t0)/60:0.3f} min') # Changed
 
    ## export mesh if requested
    if exportMesh == True:
        logger.info('    Exporting mesh') # Changed
        t0 = time.time()
        mesh = mesh2gdf(nc, epsgIn, epsgOut)
        
        if subDomain is not None:
            mesh = gpd.clip(mesh, subDom.to_crs(epsgOut))
        
        mesh.to_file(os.path.join(os.path.dirname(pathOut), f'{meshName}.shp'))
        logger.info(f'    Mesh exported: {(time.time() - t0)/60:0.3f} min') # Changed
        logger.info(f'Ready with exporting code after: {(time.time() - t00)/60:0.3f} min') # Changed
        return gdf, mesh
    
    else:
        logger.info(f'Ready with exporting code after: {(time.time() - t00)/60:0.3f} min') # Changed
        return gdf
    
def fort14togdf(filein, epsgIn, epsgOut, fileintype='netcdf'):
    ''' Write adcirc mesh from fort.14 file as GeoDataFrame and extract centroid of each element. 
        Used in the downscaling process
        Parameters:
            filein: str
                full path of the fort.14 file
            epsgIn: int
                coordinate system of the adcirc input
            epsgOut: int
                coordinate system of the output shapefile
            fileintype: str
                file type used to get nodes and elements. Default netcdf
        Returns
            gdf: GeoDataFrame
                GeoDataFrame with polygons as geometry and more info such as: area, representative
                element size, centroids coordinates, and vertices
    '''
    ## read only the two first lines of the file to get the number of elements and nodes
    if fileintype == 'fort.14':
        with open(filein) as fin:
            head = list(islice(fin, 2))
            data = [int(x) for x in head[1].split()]
        ## read nodes
        nodes = np.loadtxt(filein, skiprows = 2, max_rows = data[1], usecols = (1, 2, 3))
        ## read elements
        elem = np.loadtxt(filein, skiprows = 2 + data[1], max_rows = data[0], usecols = (2, 3, 4)) - 1
        x = nodes[:, 0]
        y = nodes[:, 1]
        z = nodes[:, 2]
    elif fileintype == 'netcdf':
        with netcdf.Dataset(filein) as nc:
            x = nc['x'][:].data              
            y = nc['y'][:].data         
            z = nc['depth'][:].data        
            elem = nc['element'][:].data - 1
    ## matplotlib triangulation
    tri = mpl.tri.Triangulation(x, y, elem)
    ## select the coordinate of each vertex
    xvertices = x[tri.triangles[:]]
    yvertices = y[tri.triangles[:]]
    zvertices = z[tri.triangles[:]]
    listElem = np.stack((xvertices, yvertices), axis = 2)
    ## define polygons and GeoDataFrame
    pols = [Polygon(x) for x in listElem]
    gdf = gpd.GeoDataFrame(geometry = pols, crs = 4326)
    
    ## change crs
    if epsgIn == epsgOut:
        pass
    else:
        gdf = gdf.to_crs(epsgOut)
 
    ## get centroids and vertices coordinatess
    gdf['zmean'] = -1*zvertices.mean(axis = 1)
    gdf['centX'] = xvertices.mean(axis = 1)
    gdf['centY'] = yvertices.mean(axis = 1)
    gdf['v1'] = elem[:, 0]
    gdf['v2'] = elem[:, 1]
    gdf['v3'] = elem[:, 2]
    gdf['id'] = range(len(gdf))

    ## compute area and presentative length if the output crs is not lat/lon
    if epsgOut == 4326:
        pass
    else:
        gdf['repLen'] = [np.round(geom.length/3, 3) for geom in gdf.geometry]
        gdf['minLen'] = [np.min([distance.euclidean(pi, pj) for pi, pj in zip(geom.boundary.coords[:-1], geom.boundary.coords[1:])]) for geom in gdf.geometry]
        gdf['elemArea'] = [np.round(geom.area, 3) for geom in gdf.geometry]

    return gdf
 
def getDates(ncFile):
    ''' Function to get dates of ADCIRC output file
        Parameters
            ncFile: string
                full path of the ADCIRC output file
        Returns
            dfDates: pandas dataframe
                df with file dates
    '''
    ncObj = netcdf.Dataset(ncFile, 'r')
    t0 = pd.to_datetime(ncObj['time'].units.split('since ')[1])
    dates = [t0 + pd.Timedelta(seconds = x) for x in ncObj['time'][:]]
    dfDates = pd.DataFrame({'dates': dates})
    return dfDates

####################################### GOOGLE EARTH ################################################

def createColorbar(levels, varName, units, cmap='viridis', fileName='tempColorbar.jpg', filePath='.'):
    ''' Create colorbar image to be imported in the kml file.
        Parameters
            levels: np.array
                contour levels
            varName: string
                name of the variable to export, just used for the legend.
            units: string
                units of the variable to export, just used for the legend.
            cmap: string
                colormap name. If varName is depth, the topo colormap from cmocean will be used.
            fileName: string
                name of the colorbar img. Default tempColorbar.jpg.
            filePath: string
                path where the colorbar will be saved.
        Return
            None. The image will be dumped in the same path where the script is executed.
                
    '''
    ## define norm for the colrobar
    norm = mpl.colors.Normalize(vmin = levels[0], vmax = levels[-1])
    ## get colorbar name if the requested variable is depth, the topo colormap from cmocean will be used
    if cmap == 'topo':
        cmap = cmocean.tools.crop(cmocean.cm.topo, levels[0], levels[-1], 0)
    else:
        cmap = mpl.cm.get_cmap(cmap)
    ## define figure to export the colorbar image
    fig, ax = plt.subplots(figsize=(1, 8))
    ## define colorbar
    cb = mpl.colorbar.ColorbarBase(ax, cmap = cmap, norm = norm, extend = 'neither',
                                    extendfrac = 'auto', ticks = levels, spacing = 'uniform',
                                    orientation = 'vertical')
    ## set label
    cb.set_label(f'{varName} [{units}]')
    ## export image
    fig.savefig(os.path.join(filePath, fileName), dpi = 300, bbox_inches = 'tight')
    plt.close()
    
def kmlScreenOverlays(kml, colorbar=True, colorbarFile='tempColorbar.jpg', logo=True, 
                        logoFile='logo.png', logoUnits='fraction', logoDims=None):
    ''' Create screen overlay for the kml file with the colorbar and the project logo.
        TODO: check logo file path when calling the code from github repo
        Parameters
            kml: simplekml object.
                output of lines2kml or poly2kml functions.
            colorbar: bolean, default True.
                True for including the colorbar.
            logo: bolean, default True.
                True for including the logo.
            logoFile: string, default "logo.png"
                name of the file to be included as logo
            logoUnits: string, default fraction
                the other available option is pixel.
            logoDims: list, default None
                x and y dimensions of the logo overlay
        Returns
            None    
    '''
    if colorbar == True:
        ## define kml overlay to add the colorbar image
        screen1 = kml.newscreenoverlay(name='Colorbar')
        ## add file
        screen1.icon.href = colorbarFile
        ## define overlay size
        screen1.overlayxy = simplekml.OverlayXY(x= 0 , y = 0, xunits = simplekml.Units.fraction,
                                         yunits = simplekml.Units.fraction)
        screen1.screenxy = simplekml.ScreenXY(x = 0, y = 0.1, xunits = simplekml.Units.fraction,
                                         yunits = simplekml.Units.fraction)
        screen1.size.x = 0.08
        screen1.size.y = 0.55
        screen1.size.xunits = simplekml.Units.fraction
        screen1.size.yunits = simplekml.Units.fraction
    ## add overlay for the 
    if logo == True:
        ## look logo in the github repository
        if logoFile == 'logo.png':
            aux0 = Path(__file__)
            #aux1 = aux0.split('\\')
            #aux2 = '\\'.join(aux1[:-2])
            #print(aux2)
            #logoFile = os.path.join(aux2, 'documentation', 'logoForKmz', logoFile)
            logoFile = aux0.parents[1]/'adds'/'logoForKmz'/logoFile
        ## define new overlay
        screen2 = kml.newscreenoverlay(name = 'logo')
        screen2.icon.href = logoFile
        screen2.overlayxy = simplekml.OverlayXY(x = 0, y = 1, xunits = simplekml.Units.fraction,
                                       yunits = simplekml.Units.fraction)
        screen2.screenxy = simplekml.ScreenXY(x = 0, y = 1, xunits = simplekml.Units.fraction,
                                        yunits = simplekml.Units.fraction)
        if logoUnits == 'fraction':     
            screen2.size.xunits = simplekml.Units.fraction
            screen2.size.yunits = simplekml.Units.fraction
        elif logoUnits == 'pixel':
            screen2.size.xunits = simplekml.Units.pixel
            screen2.size.yunits = simplekml.Units.pixel
        
        if logoDims == None:
            screen2.size.x = 0.85
            screen2.size.y = 0.08
        else:
            screen2.size.x = logoDims[0]
            screen2.size.y = logoDims[1]
            
def lines2kml(gdf, levels, cmap='viridis'):
    ''' Write GeoDataFrame with shapely linestrings as kml
        Parameters
            gdf: GeoDataFrame
                output of runExtractContours function
            levels: np.array
                contour levels
            cmap: str, default "viridis"
                colormap
        Returns
            kml: simplekml object
    '''
    ## define norm for the colrobar
    norm = mpl.colors.Normalize(vmin = levels[0], vmax = levels[-1])
    ## get colorbar name if the requested variable is depth, the topo colormap from cmocean will be used
    if cmap == 'topo':
        cmap = cmocean.tools.crop(cmocean.cm.topo, levels[0], levels[-1], 0)
    else:
        cmap = mpl.cm.get_cmap(cmap)
    ## get colormap to then transform the colors to simple kml
    m = mpl.cm.ScalarMappable(norm = norm, cmap = cmap)
    ## define kml
    kml = simplekml.Kml()
    ## loop through geometries
    for i in gdf.index:
        ## new linestring object, column name hardcoded
        ls = kml.newlinestring(name = gdf.loc[i, 'zMean'])
        try:
            ## one line
            coords = list(gdf.loc[i, 'geometry'].coords)
        except:
            ## more than one sub lines line
            aux = []
            for line in gdf.loc[i, 'geometry'].geoms:
                aux.extend(list(line.coords))
            coords = aux
        ## asign coordinates to the linestring object
        ls.coords = coords
        ## add vertical value
        value = gdf.loc[i, 'z']
        ## style
        r, g, b, a = m.to_rgba(value)
        ls.style.linestyle.color = simplekml.Color.rgb(int(255*r), int(255*g), int(255*b))
        ls.style.linestyle.width = 2
        ls.description = gdf.loc[i, 'zMean']
    
    return kml
    
def polys2kml(gdf, levels, cmap='viridis'):
    ''' Write GeoDataFrame with shapely polygons as kml
        Parameters
            gdf: GeoDataFrame
                output of runExtractContours function
            levels: np.array
                contour levels
            cmap: str, default "viridis"
                colormap
        Returns
            kml: simplekml object
    '''
    ## define norm for the colrobar
    norm = mpl.colors.Normalize(vmin = levels[0], vmax = levels[-1])
    ## get colorbar name if the requested variable is depth, the topo colormap from cmocean will be used
    if cmap == 'topo':
        cmap = cmocean.tools.crop(cmocean.cm.topo, levels[0], levels[-1], 0)
    else:
        cmap = mpl.cm.get_cmap(cmap)
    ## get colormap to then transform the colors to simple kml
    m = mpl.cm.ScalarMappable(norm = norm, cmap = cmap)
    ## define kml
    kml = simplekml.Kml()
    ## loop through geometries
    for i in gdf.index:
        ## the geometry has only one polygon
        try:
            ## outer coordinates
            outerCoords = list(zip(gdf.loc[i, 'geometry'].exterior.coords.xy[0], 
                                        gdf.loc[i, 'geometry'].exterior.coords.xy[1]))
            ## inner coordinates
            innerCoords = list(gdf.loc[i, 'geometry'].interiors)
            if len(innerCoords) > 0:
                innerCoords = [list(interior.coords) for interior in innerCoords]
            ## define kml polygon
            pol = kml.newpolygon(name = gdf.loc[i, 'zMean'])
            ## assign outer and inner coordinates
            pol.outerboundaryis = outerCoords
            pol.innerboundaryis = innerCoords
            ## z value
            value = gdf.loc[i, 'zMean']
            ## style
            r, g, b, a = m.to_rgba(value)
            col = simplekml.Color.rgb(int(255*r), int(255*g), int(255*b))
            pol.style.linestyle.color = col
            pol.style.linestyle.width = 2
            pol.description = gdf.loc[i, 'zMean']
            pol.style.polystyle.color = simplekml.Color.changealphaint(100, col)
        
        except:
            ## loop through all polygons in each geometry
            for poly in gdf.loc[i, 'geometry'].geoms:
                outerCoords = list(zip(poly.exterior.coords.xy[0], 
                                        poly.exterior.coords.xy[1]))

                innerCoords = list(poly.interiors)    
                if len(innerCoords) > 0:
                    innerCoords = [list(interior.coords) for interior in innerCoords]
                
                pol = kml.newpolygon(name = gdf.loc[i, 'zMean'])
                pol.outerboundaryis = outerCoords
                pol.innerboundaryis = innerCoords
                value = gdf.loc[i, 'zMean']
                r, g, b, a = m.to_rgba(value)
                col = simplekml.Color.rgb(int(255*r), int(255*g), int(255*b))
                pol.style.linestyle.color = col
                pol.style.linestyle.width = 2
                pol.description = gdf.loc[i, 'zMean']
                pol.style.polystyle.color = simplekml.Color.changealphaint(100, col)

    return kml

def countVertices(gdf):
    ''' Count vertices of the polygons of a GeoDataFrame.
        Parameters:
            gdf: GeoDataFrame
                output of runExtractContours function
        Returns:
            nVertices: list
                number of vertices of each polygon
    '''
    nVertices = []
    for ix, x in enumerate(gdf.geometry):
        ## number of vertices
        a = 0
        try:
            ## the geometry has only one polygon or linestring
            a = a + len(list(x.exterior.coords))
        except:
            ## loop through all sub geometries in the geometry
            for pol in x:
                a = a + len(list(pol.exterior.coords))
        nVertices.append(a)
        
    return nVertices

def splitOneGeom(geom):
    ''' Splits a polygon using the diagonal of its bounding box (upper left to lower right corner).
        Parameters:
            geom: shapely polygon
                polygon to split
        Returns
            gdfSub: GeoDataFrame
                gdf with new polygons
    '''
    ## get geometry bounding box
    bounds = geom.bounds
    ## define bounding box diagonal
    ls = LineString([(bounds[0], bounds[3]), (bounds[2], bounds[1])])
    ## split polyhon using the diabonal
    newGeoms = split(geom, ls)
    ## define new geodataframe with the split geometries
    gdfSub = gpd.GeoDataFrame(geometry = list(newGeoms.geoms), crs = 4326)
    return gdfSub

def splitAllGeoms(gdf, thres = 20_000):
    ''' splitOneGeom is applied iteratively on a GeoDataFrame until all polygons have less than a 
        certain number of vertices.
        Parameters:
            gdf: GeoDataFrame
                Group of polygons to split
            thres: int
                max number of vertices for a polygon to have
        Returns
            gdfSubAll: GeoDataFrame
                new polygons split
    '''
    a = len(gdf)
    listGdf0 = []
    count = 0
    ## iterate until the threshold is matched
    while a > 0:
        listGdf1 = []
        for ig, g in enumerate(gdf.geometry):
            ## split the geometry
            gdfSub = splitOneGeom(g)
            ## extra columns
            colsExtra = gdf.columns[1:-1]
            ## resize one row of data of the original gdf to match the shape of the new geodataframe with the split geom
            values = np.tile(gdf.iloc[ig, 1:-1].values, len(gdfSub)).reshape((len(gdfSub), len(colsExtra)))
            ## auxiliary dataframe
            dfAux = pd.DataFrame(index = gdfSub.index, columns = colsExtra, data = values)
            ## add geom to the auxiliary df
            gdfSub = pd.concat([gdfSub, dfAux], axis = 1)
            ## add number of vertices
            gdfSub['nVertices'] = countVertices(gdfSub)
            ## split the gdf between geoms with more and less vertices than the threshold
            dummy0 = gdfSub[gdfSub['nVertices'] > thres]
            dummy1 = gdfSub[gdfSub['nVertices'] <= thres]
            
            if len(dummy1) > 0:
                listGdf0.append(dummy1) ## polygons with less than 20_000 vertices
            if len(dummy0) > 0:
                listGdf1.append(dummy0) ## polygons with more than 20_000 vertices
        ## gdf to be split again
        gdfB = pd.concat(listGdf1, axis = 0)
        gdfB.index = range(len(gdfB))
        if len(gdfB) == 1:
            break
        else:
            gdf = gdfB.copy()
        a = len(gdf)
        count = count + 1
    gdfSubAll = pd.concat(listGdf0, axis = 0)
    gdfSubAll.index = range(len(gdfSubAll))
    
    return gdfSubAll
    
def nc2kmz(ncFile, var, levels, conType, epsg, pathOut, vUnitIn='m', vUnitOut='m', vDatumIn='tss', vDatumOut='tss',
           subDomain=None, overlay=True, logoFile='logo.png', colorbarFile='tempColorbar.jpg', 
           cmap='viridis', thresVertices=20_000, dzFile=None, zeroDif=-20):
    ''' Run all necesary functions to export adcirc outputs as kmz.
        Parameters
            ncFile: string
                path of the adcirc output, must be a netcdf file
            var: string
                Name of the variable to export
            levels: list
                Contour levels, [min value, max value, step]
            conType: string
                'polyline' or 'polygon'
            epsg: int
                coordinate system
            pathout: string
                complete path of the output file (*.shp or *.gpkg)
            vUnitIn, vUnitOut: string. Default 'm'
                input and output vertical units. For the momment only supported 'm' and 'ft'
            vdatumIn, vdatumOut: string. Default 'tss'
                name of the input and output vertical datums. Mean sea level is "tss"
                For checking the available datums:
                from vyperdatum.pipeline import datum_definition
                list(datum_definition.keys())
            subDomain: str or list. Default None
                complete path of the subdomain polygon kml or shapelfile, or list with the
                uper-left x, upper-left y, lower-right x and lower-right y coordinates The crs must be the same of the
                adcirc input file.
            overlay: boolean. Default True
                If true overlay layer with logo and colorbar are added to the kmz file
            logoFile: string. Default 'logo.png'
                path of the logo image
            colorbarFile: string. Default 'tempColorbar.jpg'
                name of the colorbar img
            cmap: string. Default 'viridis'
                name of the colormap. If varName is depth, the topo colormap from cmocean will be used
            thresVertices: int. Default 20_000
                maximum number of vertices allowed per polygon. If a polygon has more vertices, the
                katana function will be used. Google Earth has problems displaying polygons with large
                amount of vertices, while smaller this number, better will be the vizualiation but the
                execution time of the extracting function will be longer.
            dzFile: str
                full path of the pickle file with the vertical difference between datums
                for each mesh node
            zeroDif: int
                threshold for using nearest neighbor interpolation to change datum. Points below
                this value won't be changed.
        Returns
            gdf: GeoDataFrame
                gdf with contours
    '''
    ## read netcdf
    nc = netcdf.Dataset(ncFile, 'r')
    
    ## check if the file is time-constant or time-varying
    if checkTimeVarying(nc) == 1:
        ## file can time-varying but if the depth is requested, it can be exported to google earth
        if var != 'depth':
            logger.info('Time-varying files can not be exported as kmz!') # Changed
            sys.exit(-1)
    ## colormap name for bathymetry
    if var == 'depth':
        cmap = 'topo'
        
    stepLevel = levels[2]
    orgMaxLevel = levels[1]
    ## arrray with levels
    levels = np.arange(levels[0], levels[1], levels[2])
    ## extract contours
    gdf = runExtractContours(nc, var, levels, conType, epsg, stepLevel, orgMaxLevel, dzFile, zeroDif)
    if conType == 'polygon':
        ## split polygons if necessary
        gdf['nVertices'] = countVertices(gdf)
        gdfA = gdf[gdf['nVertices'] > thresVertices]
        gdfB = gdf[gdf['nVertices'] <= thresVertices]
    
        if len(gdfA) > 0:
            gdfC = splitAllGeoms(gdfA, thresVertices)
            gdf = pd.concat([gdfB, gdfC], axis = 0)
            gdf.index = range(len(gdf))
    ## clip data if subdomain is provided
    if subDomain is not None:
        subDom = readSubDomain(subDomain, epsg)
        gdf = gpd.clip(gdf, subDom)
    ## change vertical datum if requested
    if vDatumIn == vDatumOut:
        pass
    else:
        gdf = gdfChangeVertDatum(vDatumPath, gdf, vDatumIn, vDatumOut)
    ## change vertical units if requested
    if vUnitIn == vUnitOut:
        pass
    else:
        gdf = gdfChangeVerUnit(gdf, vUnitIn, vUnitOut)
    ## export to google earth
    if conType == 'polygon':
        kml = polys2kml(gdf, levels, cmap)
    elif conType == 'polyline':
        kml = lines2kml(gdf, levels, cmap)
    else:
        logger.info('Only "polygon" o "polyline" formats are supported') # Changed
        sys.exit(-1)
    ## add overlay if requested
    if overlay == True:
        name = nc[var].long_name.capitalize()
        units = nc[var].units
        createColorbar(levels, name, units, cmap=cmap, fileName=colorbarFile, filePath='.')
        kmlScreenOverlays(kml, colorbar=True, colorbarFile=colorbarFile, logo=True, 
                    logoFile=logoFile, logoUnits='fraction', logoDims=None)
    ## save file and remove temp colorbar image
    kml.savekmz(pathOut, format = False)
    os.remove('tempColorbar.jpg')
        
    return gdf

