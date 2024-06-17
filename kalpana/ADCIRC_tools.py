import numpy as np
import pandas as pd
from itertools import islice
import matplotlib as mpl
from scipy import interpolate
from tqdm import tqdm
import datetime
from sklearn.neighbors import KDTree
import time
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point, LineString
from .export import fort14togdf, readSubDomain, pointsInsidePoly
import geopandas as gpd
import rtree

def readNodes_fort14(f14):
    ''' Fx to read the fort.14 nodes as a pandas dataframe
        Parameters
            f14: string
               full path of the fort.14 file
        Returns
            Nodes: pandas dataframe
    '''
    with open(f14) as fin:
        head = list(islice(fin, 2))
        data = [int(x) for x in head[1].split()]
    nodes = pd.read_csv(f14, skiprows = 2, nrows = data[1], names = ['x', 'y', 'z'], delim_whitespace = True)
    nodes.index = [x - 1 for x in nodes.index]
    return nodes

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

def extract_ts_from_nc(ncObj, pnts, n=5, variable='zeta', extractOut=False, closestIfDry=False):
    ''' Interpolate adcirc results from the 3 nodes that forms the triangle in which 
        a point lies in for all timesteps
        Parameters
            ncObj: etCDF4._netCDF4.Dataset
                Adcirc input file
            pnts: list
                list with ziped coordinates. Eg. [(x0, y0), (x1, y1), ....]
            n: int, default 3
                number of triangles considered to look in which one the point
                is contained.
            extractOut: boolean
                True for extract data if the points are outside the domain. Defalt False,
                nan will be returned in this case.
        Returns
            dfout: pandas dataframe
                df with of interpolated results
            rep: list
                strings with information about how the data was extracted
    '''
    ## triangles
    nv = ncObj['element'][:,:] - 1 ## triangles starts from 1
    ## x and y coordinates
    x = ncObj['x'][:].data
    y = ncObj['y'][:].data
    ## matplotlib triangulation
    tri = mpl.tri.Triangulation(x, y, nv)
    ## get the x and y coordinate of the triangle elements in the right order
    xvertices = x[tri.triangles[:]]
    yvertices = y[tri.triangles[:]]
    ## add x and y togheter
    listElem = np.stack((xvertices, yvertices), axis = 2)
    ## vertex number of each node
    v1 = nv.data[:, 0]
    v2 = nv.data[:, 1]
    v3 = nv.data[:, 2]
    v = np.array((v1, v2, v3)).T  
    ## get centroids
    centx = xvertices.mean(axis = 1)
    centy = yvertices.mean(axis = 1)
    ## compute distance from all centroids to the requested points
    mdist = cdist(list(zip(centx, centy)), pnts)
    ## iterate through each point to find in what triangle is contained
    t0 = pd.to_datetime(ncObj['time'].units.split('since ')[1])
    dates = [t0 + pd.Timedelta(seconds = float(x)) for x in ncObj['time'][:]]
    dfout = pd.DataFrame(columns = [f'{variable}_pnt{x:03d}' for x in range(len(pnts))], index = dates)
    ## check if nc file is time-varying
    tvar = checkTimeVarying(ncObj)
    if tvar == 1:
        z = ncObj[variable][:, :].data
    else:
        ## reshape to add an extra dimension
        z = ncObj[variable][:].data.reshape((1, ncObj[variable].size))

    z[z < -9999] = np.nan
    ## loop through points
    rep = []
    # for i in tqdm(range(len(pnts))):
    for i in range(len(pnts)):
        # print(pnts[i])
        ## get the nearest n centroid to point i
        # print(mdist.shape)
        a = np.where(mdist[:, i] < sorted(mdist[:, i])[n])[0]
        #print(a)
        ## iterate through each element to see is the point is inside
        for ni in range(n):
            lnewzti = []
            ## define polygon
            pol = Polygon(listElem[a[ni], :, :])
            ## find the polygon that contains the point
            if pol.contains(Point(pnts[i])):
                vs = v[a[ni]]
                break
        ## point is inside the mesh
        if 'vs' in locals():
            xs = ncObj['x'][vs].data
            ys = ncObj['y'][vs].data 
            ## variable to interpolate
            zs = z[:, vs]

            ## 3 nodes are dry for the entire simulation, extract values from the closest node with any wet time step
            if np.isnan(zs).all() and closestIfDry == True:
                mdist2 = cdist(list(zip(x, y)), np.reshape(pnts[i], (1, 2)))
                mdist2 = np.concatenate((mdist2, np.arange(len(mdist2)).reshape((-1, 1))), axis = 1)
                mdist2_sorted = np.take(mdist2, np.argsort(mdist2[:, 0]), axis = 0)

                for k in mdist2_sorted[:, 1]:
                    zsk = z[:, int(k)]
                    if np.isnan(zsk).all() == False:
                        newz = zsk
                        rep.append(f"Point {i:03d} is inside the domain! All element's nodes are dry, so data from the closest wet node ({int(k)}) was extracted.")
                        break
                try:
                    lnewzti.extend(newz)
                except:
                    lnewzti.append(newz)

            elif np.isnan(zs).any() and closestIfDry == True:
                ## get values from the closest node with wet time steps from the same element
                
                mdist2 = cdist(list(zip(x[v[a[ni]]], y[v[a[ni]]])), np.reshape(pnts[i], (1, 2)))
                mdist2 = np.concatenate((mdist2, np.arange(len(mdist2)).reshape((-1, 1))), axis = 1)
                mdist2_sorted = np.take(mdist2, np.argsort(mdist2[:, 0]), axis = 0)

                for k in mdist2_sorted[:, 1]:
                    zsk = zs[:, int(k)]
                    if np.isnan(zsk).all() == False:
                        newz = zsk
                        rep.append(f"Point {i:03d} is inside the domain! There are dry nodes in the element, so data from the closest wet node ({v[a[int(k)]]}) in the element was extracted.")
                        break
                try:
                    lnewzti.extend(newz)
                except:
                    lnewzti.append(newz)
            else:
                rep.append(f"Point {i:03d} is inside the domain! data was interpolated using the corresponding element's nodes.")
                ## all nodes are wet or, if one node of the element is dry output will be nan
                for zi in zs:
                    f = interpolate.LinearNDInterpolator(list(zip(xs, ys)), zi)
                    newz = float(f(pnts[i][0], pnts[i][1]))
                    lnewzti.append(newz)
                    
            dfout[f'{variable}_pnt{i:03d}'] = lnewzti
            del vs
                
        else:
            ## point is outside the domain
            if extractOut == True:
                ## find nearest node to the requested point
                mdist2 = cdist(list(zip(x[v[a[0]]], y[v[a[0]]])), np.reshape(pnts[i], (1, 2)))
                clnode = mdist2.argmin()
                rep.append(f'Point {i:03d} is outside the domain! data from  node {clnode} (closest) was exported.')
                newz = z[:, v[a[0]][clnode]]
                lnewzti = newz.copy()
                dfout[f'{variable}_pnt{i:03d}'] = lnewzti
            else:
                rep.append(f'Point {i:03d} is outside the domain! Returning nan.')
                dfout[f'{variable}_pnt{i:03d}'] = np.nan
    
    return dfout, rep

def NNfort13(fort14_old, fort14_new, fort13_old, fort13_new, attrs):
    ''' Function to interpolate the fort.13 from one mesh to another using
        nearest neighbor
        Parameters
            fort14_old: str
                full path of the original fort.14
            fort13_old: str
                full path of the original fort.13
            fort14_new: str
                full path of the new fort.14
            fort13_new: str
                full path of the new fort.13, a new file will be created
            attrs: dictionary
                attributes to consider in the new fort.13
                Currently the keys of the are **the exact same name** of the attributes, be careful
                with empty spaces (this will be fixed soon, WIP).
                The items are integers with the number of lines per attribute in the hader information.
                (WIP: As far as I know, all attrs always have three lines In case this is always
                true this input needs to be changed to a list). 
                E.g.
                attrs = {
                         'surface_directional_effective_roughness_length': 3,
                         'surface_canopy_coefficient': 3,
                         'mannings_n_at_sea_floor': 3,
                         'primitive_weighting_in_continuity_equation': 3,
                         'average_horizontal_eddy_viscosity_in_sea_water_wrt_depth': 3,
                         'elemental_slope_limiter': 3
                        }
        Return
            None
    '''
    ## open old fort.14 to get number of nodes and elements
    with open(fort14_old) as fin:
        head_old = list(islice(fin, 2))
        data_old = [int(x) for x in head_old[1].split()]
    ## read only nodes as an array of x,y
    nodes_old = np.loadtxt(fort14_old, skiprows = 2, max_rows = data_old[1], 
                           usecols = (1, 2))
    
    ## idem for new fort.14
    with open(fort14_new) as fin:
        head_new = list(islice(fin, 2))
        data_new = [int(x) for x in head_new[1].split()]
    nodes_new = np.loadtxt(fort14_new, skiprows = 2, max_rows = data_new[1], 
                           usecols = (1, 2))
    
    ## nearest neighbor interpolation --> the closest node of the new mesh is
    ## assign to each of the nodes of the old mesh.
    tree = KDTree(nodes_old)
    dist, ind = tree.query(nodes_new)
    ## dataframe with new nodes and the closest old node assigned to each one
    dfnew = pd.DataFrame({'x': nodes_new[:, 0], 'y': nodes_new[:, 1], 'old_id':ind.reshape(-1)})
    
    ## open the old fort.13 in read mode to read the data
    with open(fort13_old, 'r') as fin:
        ## open the new fort.13 in writing mode to dump the interpolated information
        with open(fort13_new, 'w') as fout:
            ## write header in to the new fort.13: titile, number of nodes and number of
            ## attributes.
            fout.write(f'Spatial attributes descrption. File generated with NNfort13 on {datetime.date.today()} using {fort14_old} and {fort13_old} as basefiles, and {fort14_new} as the new mesh\n')
            fout.write(f'{data_new[1]}\n')
            fout.write(f'{len(attrs.keys())}\n')## write
            
            ## Inside this for loop we are writing the name, default value and other
            ## parameters of each of the selected attributes
            lines = fin.readlines()[3:]
            for key in attrs.keys():
                ## index of the attribute
                ind = lines.index(key+'\n')
                ## write default values and extra info per attribute
                ## lines from index of attr to the int in the item of that specific
                ## key es written.
                fout.writelines(lines[ind:ind+1+attrs[key]])
            
            ## From this line the value of the attrs for each node es written.
            for key in tqdm(attrs.keys()):
                ## get index of 1st and 2nd time the attr key appears in the file
                inds = [i for i, n in enumerate(lines) if n == key+'\n'][:2]
                try:
                    ## read default value, try and except block is due to the 
                    ## surface_directional_effective_roughness_length, which is a typically
                    ## a list of 12 values
                    defval = lines[inds[0]+3].split()
                    ## convert from str to float
                    defval = [float(x) for x in defval]
                except:
                    ## in case the values are not a list
                    defval = [float(lines[inds[0]+3][:-1])]
                
                ## index where the values will be dumped
                indi = inds[1] + 2
                ## number of nodes with non default value
                nnondef = int(lines[int(inds[1] + 1)][:-1])
                ## index where the nodes of the attr finish
                indf = indi + nnondef - 1
                ## read data only if there are more than 0 non default vertices
                if nnondef > 0:
                    ## read the lines between previous defined indices as dataframes
                    olds = pd.read_csv(fort13_old, skiprows = indi + 3, nrows = indf - indi + 1, header = None, sep = ' ', index_col = 0)
                    ## problem when first column of the org fort 13 has whitespaces, the index is nan and an extra column
                    ## with the vertex id is added to the dataframe
                    if np.isnan(olds.index).all() == True:
                        olds.index = olds.iloc[:, 0]
                        olds = olds.drop([1], axis = 1)
                        olds.columns = range(1, len(olds.columns) + 1)
                    else:
                        pass
                    ## not sure why this dataframe is not writable: olds_all_aux.values.flags will show the array is not writable. Fixed with copy
                    ## array for store the value of all nodes, not only the non-default
                    olds_all_aux = pd.DataFrame(columns = olds.columns, index = range(1, data_old[1] + 1),
                                                data = np.broadcast_to(np.array(defval), (data_old[1], len(defval))))
                    olds_all = olds_all_aux.copy()
                    ## add info of nodes with default value
                    olds_all.loc[olds.index, :] = olds.values
                    ## dataframe with attr values for the nodes of the new mesh
                    ## this is done selecting the data of the old data for the closest old
                    ## node associated to each of the new nodes
                    news_all = olds_all.loc[dfnew['old_id'] + 1, :]
                    news_all.index = range(1, len(news_all) + 1)
                    ## get the nodes with default value
                    dfdef = news_all[news_all == defval].dropna()
                    ## get nodes with non-default value
                    dfnondef = news_all[news_all != defval].dropna()
                    dfnondef = dfnondef.sort_index()
                    dfnondef['dummy'] = '\n'
                    ## write attribute name
                    fout.write(key + '\n')
                    ## write number of non default nodes
                    fout.write(str(len(dfnondef)) + '\n')
                    ## format the data to write it to the new fort.13 file
                    dfaux = pd.DataFrame({'x': dfnondef.index}, index = dfnondef.index)
                    new_lines = pd.concat([dfaux, dfnondef], axis = 1)
                    new_lines2 = [' '.join([str(x) for x in new_lines.loc[i, :]]) for i in new_lines.index]
                    fout.writelines(new_lines2)
                else:
                    ## write attr with only default values
                    fout.write(key + '\n')
                    fout.write('0\n')

def subsetMeshGdf(gdf, nodes, mask):
    '''  Subsets a geodataframe based on a mask polygon and returns matplotlib triangulation,
         the nodes outside the mask, and a geodataframe with the mesh subset. It is important to note
         that the part of the mesh inside the mask is removed.
         
         Parameters
            gdf: geopandas GeoDataFrame
                mesh gdf, each element is an individual polygon. Output of fort14togdf function
            nodes: pandas dataframe
                mesh nodes, output of readNodes_fort14 function
            mask: geopandas GeoDataFrame
                mask gdf, output of readSubDomain function.
        Returns
            newMeshTri: matplotlib triangulation object
                subset of the mesh triangulation
            meshSub: geopandas GeoDataFrame
                mesh subset gdf, each element is an individual polygon.
            dfNodesOutside: pandas dataframe
                coordinates of the nodes outside the mask (new mesh nodes). Dataframe index
                starts from 0, has x, y and z values, and the index of the node in the original mesh
    '''
    # Extract exterior coordinates of the mask polygon
    xAux, yAux = mask.geometry[0].exterior.coords.xy
    extCoords = list(zip(xAux, yAux))
    # Get centroids of mesh elements
    centroids = list(zip(gdf['centX'], gdf['centY']))
    # Determine which centroids are inside the mask polygon
    inside = pointsInsidePoly(centroids, extCoords)
    # Get indices of elements to keep (outside the mask polygon)
    # starts from 0 and not from 1 as ADCIRC nodes
    centOutside = np.where(inside == False)[0]
    # Subset the geodataframe based on the selected elements, the index starts from 0
    # so iloc or loc gives same result
    meshSub = gdf.iloc[centOutside, :]
    # get nodes of the elements to keep, starts from 0
    nodesOutside = meshSub.loc[:, ['v1', 'v2', 'v3']].values.reshape(-1)
    # remove repeated nodes
    nodesOutside = np.unique(nodesOutside)
    # get coordinates of nodes to keep, the index of the series correspond to the
    # full mesh
    xNodesOutside = nodes.iloc[nodesOutside, 0]
    yNodesOutside = nodes.iloc[nodesOutside, 1]
    zNodesOutside = nodes.iloc[nodesOutside, 2]
    # Create a lookup table for renumering the mesh
    aux = {value: index for index, value in enumerate(list(nodesOutside))}
    # find index of each element vertex in the list nodes to keep
    newV = [(aux[x], aux[y], aux[z]) for x, y, z in zip(meshSub['v1'], meshSub['v2'], meshSub['v3'])]
    # Create a triangulation based on the subsetted nodes and elements
    newMeshTri = mpl.tri.Triangulation(xNodesOutside, yNodesOutside, newV)
    # add new element's nodes to the geodataframe
    meshSub[['v1u', 'v2u', 'v3u']] = newV
    meshSub.index = range(len(meshSub))
    ## dataframe with new mesh nodes
    dfNodesOutside = pd.DataFrame({'x': xNodesOutside.values,
                                    'y': yNodesOutside.values,
                                    'z': zNodesOutside.values,
                                    'orgIndex': xNodesOutside.index})

    return newMeshTri, meshSub, dfNodesOutside

def readOrgBC(f14, nodes, epsg = 4326):
    ''' Reads boundary condition information from a fort.14 file and returns a GeoDataFrame and a dictionary
        Parameters
            f14: string
                full path of the fort.14 file
            nodes: pandas dataframe
                mesh nodes, output of readNodes_fort14 function
            epsg: int. Default 4326 (lat/lon)
                coordinate reference system
        Returns
            gdfBC: geopandas GeoDataFrame
                geodataframe with fort.14 boundary conditions
            dctBC: dictionary
                boundary conditions nodes ID
    '''
    with open(f14) as fin:
        ## get header of the fort.14: number of elements and nodes
        head = list(islice(fin, 2))
        data = [int(x) for x in head[1].split()]
        ## read lines with BC information
        lines = fin.readlines()[data[0]+data[1]:]
        ## dictionary to store the data
        dctBC = {'n_open_bound': int(lines[0].split()[0]), 'total_nodes_open_bc': int(lines[1].split()[0])}
        lines = lines[2:]
        ob = 0 ## open boundary counter
        lb = 0 ## line boundary counter
        aux = 0
        
        while len(lines) > 0:
            ## ob is for open boundary
            if ob < dctBC['n_open_bound']:
               # Read open boundary information
                nn = int(lines[0].split()[0]) # Number of nodes in the boundary condition
                bc = [int(x)-1 for x in lines[1:nn+1]] # Node index of the boundary condition, starts from 0 as python indices
                dctBC[f'bc_open_bound_{ob}'] = bc # Store the boundary condition in the dictionary
                aux = 0 # flag that helps when reading land boundaries
                lines = lines[len(bc)+1:] # remove the lines with the BC info already stored
                ob += 1 # update counter
            else:
                if aux == 0:
                    # first time reading land BC
                    dctBC['n_land_bound'] = int(lines[0].split()[0]) # number of land boundaries
                    dctBC['total_nodes_land_bc'] = int(lines[1].split()[0]) # land boundaries total nodes
                    aux += 1 # update flag
                    lines = lines[2:] # remove analyzed lines
                else:
                    nn = int(lines[0].split()[0]) # number of nodes of the current land boundary
                    bc = [int(x.split()[0]) - 1 for x in lines[1:1+nn]] # Node index of the boundary condition, starts from 0 as python indices
                    dctBC[f'bc_land_bound_{lb}'] = bc # store the data
                    lb += 1 # update counter
                    lines = lines[nn+1:] # remove analyzed lines
    
    ## create geodataframe
    nBC, lBC, tBC = [], [], []
    for key in [x for x in dctBC.keys() if x.startswith('bc_')]:
        ## check if BC is closed or open
        bc = dctBC[key]
        if bc[0] == bc[-1]: ## bc is closed
            ## define shapely Polygon
            geom = Polygon(list(zip(nodes.loc[dctBC[key], 'x'],
                                    nodes.loc[dctBC[key], 'y'])))
            dummy = 1 ## flag for closed BC
        else: ## bc is open --> ocean or main land boundary
            geom = LineString(list(zip(nodes.loc[dctBC[key], 'x'],
                                    nodes.loc[dctBC[key], 'y'])))
            dummy = 0
        nBC.append(key) ## name
        lBC.append(geom) ## geometries
        tBC.append(dummy) ## type

    gdfBC = gpd.GeoDataFrame(geometry = lBC, crs = epsg)
    gdfBC['bc_name'] = nBC
    gdfBC['bc_closed'] = tBC

    return gdfBC, dctBC

def renumClosedBCs(gdf, mask, dfNodesNew):
    ''' Update the numbering of the closed land boundary conditions
        Parameters
            gdf: geopandas GeoDataFrame
                boundary conditions gdf, output of readBCfort14
            mask: geopandas GeoDataFrame
                mask gdf, output of readSubDomain function.
            dfNodesOutside: pandas dataframe
                coordinates of the nodes outside the mask (new mesh nodes). Dataframe index
                starts from 0, has x, y and z values, and the index of the node in the original mesh
        Returns
            dctBC_closed: dictionary
                updated closed boundary conditions nodes number
            gdfBC_closed: geopandas GeoDataFrame
                updated closed boundary conditions gdf
    '''
    ## iterate through closed BC to see if they are inside or outside the new boundary
    gdfBC_closed = gdf[gdf['bc_closed'] == 1]
    auxList = []
    for bc in gdfBC_closed['geometry']:
        ## subDom represents the part of the mesh I want
        ## to exclude from the fort.14
        within = bc.within(mask['geometry'][0])
        if within == True:
            auxList.append(False)
        else:
            auxList.append(True)

    gdfBC_closed['inNewDom'] = auxList

    ## get the node ID of the BCs, the node's ID are related to the new mesh
    ## create lookup table, starts from 0
    coordsNodesOutside = list(zip(dfNodesNew['x'], dfNodesNew['y']))
    lookup_table = {tuple_val: index for index, tuple_val in enumerate(coordsNodesOutside)}
    dctBC_closed = {}

    for i in gdfBC_closed[gdfBC_closed['inNewDom'] == True].index:
        coords = list(gdfBC_closed.loc[i, 'geometry'].exterior.coords)
        # Find the indices of the target tuples
        indices = [lookup_table.get(tuple_val) for tuple_val in coords]
        dctBC_closed[gdfBC_closed.loc[i, 'bc_name']] = indices

    return dctBC_closed, gdfBC_closed[gdfBC_closed['inNewDom'] == True]

def renumOceanBC(gdf, dfNodesNew, sortBy=1, rev=False):
    ''' Update the numbering of the ocean boundary condition. The ocean BC nodes are sorted depending
        on the mesh orientation. E.g. if the mesh is aligned with N-S and the BC is eastwards of the coast
        (like NA meshes), the nodes are sorted by latitude in ascending order.
        Parameters
            gdf: geopandas GeoDataFrame
                boundary conditions gdf, output of readBCfort14
            dfNodesOutside: pandas dataframe
                coordinates of the nodes outside the mask (new mesh nodes). Dataframe index
                starts from 0, has x, y and z values, and the index of the node in the original mesh
            sortBy: int. Default 1
                If 1, nodes are sorted by latitude since BC is aligned with N-S (vertical).
                If 0, nodes are sorted by longitude since BC is aligned with W-E (horizontal).
            rev: boolean. Default False
                If False, nodes are sorted in increasing order. This is neede 
        Returns
            dfOpen: pandas dataframe
                renumbered ocean boundary condition
            gdfBC_open: geopandas GeoDataFrame
                updated closed boundary conditions gdf

    '''
    ## for now the code will assume the ocean boundary is not modified and the mask does not overlay with it
    ## get only the open bcs
    gdfBC_open = gdf[gdf['bc_closed'] == 0]

    ## get ocean bc
    oceanOpen = gdfBC_open[gdfBC_open['bc_name'] == 'bc_open_bound_0']['geometry'][0]
    ## get list of coordinate tuples
    oceanOpenCoords = list(oceanOpen.coords)
    ## sort the nodes depending in mesh orientation
    oceanOpenCoords = sorted(oceanOpenCoords, key = lambda x: x[sortBy], reverse = rev)
    
    ## get the node ID of the BCs, the node's ID are related to the new mesh
    ## create lookup table
    coordsNodesOutside = list(zip(dfNodesNew['x'], dfNodesNew['y']))
    lookup_table = {tuple_val: index for index, tuple_val in enumerate(coordsNodesOutside)}
    ## find id of each bc node. ADCIRC numbering starts from 1
    indices = [lookup_table.get(tuple_val) for tuple_val in oceanOpenCoords]
    dfOpen = dfNodesNew.iloc[indices, :]
    
    return dfOpen, gdfBC_open.iloc[[0], :]

def renumMainlandBC(gdfNew, gdfOcean, dfOcean, dfNodesNew, epsg = 4326):
    ''' Update the numbering of the mainland boundary condition
        Parameters
            gdfNew: geopandas GeoDataFrame
                mesh subset gdf, each element is an individual polygon. Output of subsetMeshGdf function
            gdfOcean:geopandas GeoDataFrame
                updated closed boundary conditions gdf. Output of updateOceanBC
            dfNodesOutside: pandas dataframe
                coordinates of the nodes outside the mask (new mesh nodes). Dataframe index
                starts from 0, has x, y and z values, and the index of the node in the original mesh
            epsg: int. Default 4326
                coordinate reference system
        Returns
            dfMainlandBC: pandas DataFrame
                BC with the updated nodes id
    '''
    ## get outer polygon of the mesh geodataframe
    outerPolNewMesh = gdfNew['geometry'].unary_union
    gdfOuterPolNewMesh = gpd.GeoDataFrame(geometry = [outerPolNewMesh], crs = epsg).boundary

    ## here I assumed that the linestring with more nodes is the outer boundary (ocean + mainland)
    max_nodes = 0
    max_line = None
    for line in gdfOuterPolNewMesh[0].geoms:
        num_nodes = len(line.coords)
        # Check if the current LineString has more nodes than the previous maximum
        if num_nodes > max_nodes:
            max_nodes = num_nodes
            max_line = line

    ## geodataframe with mainland + ocean bc
    mainBoundary = gpd.GeoDataFrame(geometry = [max_line], crs = epsg)
    ## ocean boundary condition linestring
    lsOcean = gdfOcean.loc[0, 'geometry']
    ## outer polygon boundary as linestring
    lsBound = mainBoundary.loc[0, 'geometry']
    ## get coordinates, list with tuples [(lon0, lat0), (lon1, lat1), ...]
    lsOceanCoords = list(zip(dfOcean['x'], dfOcean['y']))
    lsBoundCoords = list(lsBound.coords)
    ## define list of linestrings for the mesh ordering
    linesAll = [LineString((x, y)) for x, y in zip(lsBoundCoords[:-1], list(lsBoundCoords)[1:])]
    linesOcean = [LineString((x, y)) for x, y in zip(list(lsOceanCoords)[:-1], list(lsOceanCoords)[1:])]

    ## geodataframe with ocean bc
    gdfOcean = gpd.GeoDataFrame(geometry = linesOcean, crs = epsg)
    ## geodataframe with full boundary
    gdfAll = gpd.GeoDataFrame(geometry = linesAll, crs = epsg)
    ## geodataframe with mainland (difference between full boundary and ocean bc)
    gdfLand = gpd.overlay(gdfAll, gdfOcean, how = 'difference')

    ## start the mainland with the last point of the ocean bc to satisfy the counter clockwise ordering
    mainlandCounter = [Point(lsOceanCoords[-1])]

    # Build a spatial index for the LineString geometries
    spatial_index = rtree.index.Index()
    for i, geometry in enumerate(gdfLand['geometry']):
        spatial_index.insert(i, geometry.bounds)

    i = 0
    while len(mainlandCounter) < len(gdfLand)+1:
        # Find the nearest LineString to the last point of the mainlandCounter using the spatial index
        nearest_idx = list(spatial_index.nearest(mainlandCounter[-1].bounds, 1))[0]
        nearestLine = gdfLand.iloc[nearest_idx]['geometry']
        
        # Add the first point of the nearest LineString to mainlandCounter
        mainlandCounter.append(Point(nearestLine.coords[0]))
        
        # Remove the analyzed LineString from the spatial index
        spatial_index.delete(nearest_idx, nearestLine.bounds)
        i+=1

    ## mainlandCounter has the bound of each lineString, but we only need the starting point of each linestring
    mainlandCounterCoords = [x.bounds[:2] for x in mainlandCounter]
    # Create a dictionary lookup table for the indices
    lookup_table = {tuple_val: index for index, tuple_val in enumerate(zip(dfNodesNew['x'], dfNodesNew['y']))}
    # Find the indices of the target tuples
    indicesMainland = [lookup_table.get(tuple_val) for tuple_val in mainlandCounterCoords]
    dfMainlandBC = dfNodesNew.iloc[indicesMainland, :]
    
    return dfMainlandBC

def writeFort14(f14in, f14out, gdfNew, dfNodesNew, dfOpen, dctClosed, mainlandBC):
    ''' Write the fort.14
        Parameters
            f14in: string
                full path of the original fort.14. It is used only to get the header
            f14out: string
                full path of the output fort.14
            gdfNew: geopandas GeoDataframe
                mesh subset gdf, each element is an individual polygon.
            dfNodesOutside: pandas dataframe
                coordinates of the nodes outside the mask (new mesh nodes). Dataframe index
                starts from 0, has x, y and z values, and the index of the node in the original mesh
            dfOpen: pandas dataframe
                renumbered ocean boundary condition
            dctClosed: dictionary
                updated closed boundary conditions nodes number
            mainlandBC: pandas DataFrame
                BC nodes with the updated ID
        Returns
            None
    '''

    ## get original fort.14 header
    with open(f14in, 'r') as fin:
        header = list(islice(fin, 1))[0][:-1]
    
    now = datetime.datetime.now()
    nowStr = now.strftime("%Y/%m/%d %H:%M:%S")
    
    ## start writing new fort.14
    with open(f14out, 'w') as fout:
        ## write new header
        fout.write(f'{header} modified with fort14Subset on {nowStr}\n')
        ## write number of elements and nodes
        fout.write(f'{len(gdfNew)} {len(dfNodesNew)}\n')
        ## write nodes
        for i in dfNodesNew.index:#i, (xi, yi, zi) in enumerate(zip(xNodes, yNodes, zNodes)):
            xi = dfNodesNew.loc[i, 'x']
            yi = dfNodesNew.loc[i, 'y']
            zi = dfNodesNew.loc[i, 'z']
            fout.write(f"   {i+1:7}  {xi:13.10f}  {yi:13.10f}  {zi:14.10f}\n")
        ## write triangles
        for i in gdfNew.index:
            v1 = gdfNew.loc[i, 'v1u'] + 1
            v2 = gdfNew.loc[i, 'v2u'] + 1
            v3 = gdfNew.loc[i, 'v3u'] + 1
            fout.write(f"{i+1:7} 3 {v1} {v2} {v3}\n")
    
        ## start BC section
        # write number of open boundaries and total nodes
        fout.write("1 = Number of open boundaries\n")
        # get total number of open boundary nodes
        # total_nodes_open_bc = sum(len(lst) for lst in dctOpen.values())
        fout.write(f"{len(dfOpen)} = Total number of open boundary nodes\n")
    
        # write ocean boundary condition
        fout.write(f"{len(dfOpen)} 20 = Number of nodes for open boundary 1\n")
        for n in dfOpen.index:
            fout.write(f'{n + 1}\n')
        
        # write number of land boundaries and total nodes
        fout.write(f"{1 + len(dctClosed.keys())} = Number of land boundaries\n")
        # get total number of land boundary nodes
        total_nodes_land_bc = sum(len(lst) for lst in dctClosed.values()) + len(mainlandBC)
        fout.write(f"{total_nodes_land_bc} = Total number of land boundary nodes\n")
        
        # write main land boundary
        fout.write(f"{len(mainlandBC)} 20 = Number of nodes for land boundary 1\n")
        for n in mainlandBC.index:
            fout.write(f'{n + 1}\n')
            # fout.write(f"{int(mainlandBC.loc[n, 'index'])+1}\n")
        
        ## closed land boundaries
        for ik, k in enumerate(dctClosed.keys()):
            fout.write(f"{len(dctClosed[k])} 21 = Number of nodes for land boundary {ik+2}\n")
            for n in dctClosed[k]:
                fout.write(f'{n + 1}\n')

def subsetMesh(f14in, subDomain, f14out, epsg=4326, sortBy=1, rev=False):
    ''' Create a subset of a fort.14 using a shapefile as mask to remove the elements. 
        The code has some limitations since it has been tested only with meshes of the entire North Atlantic.
            - It is assumed that the ocean BC goes first in the fort.14
            - Only one ocean BC
            - Mask should not intersects islands or closed boundary conditions (not tested).
            - It is assumed there are closed boundaries or islands in the domain.

        NNfort13 function can be used to create a fort.13 for the new fort.14. It uses nearest neighbor
        to interpolate the nodal attributes from the original mesh to the new one.

        Note that as the nodes are renumebered, the fort.15 tide constituents might be changed.

        Parameters
            f14in: string
                full path of the original fort.14
            subDomain: str or list
                complete path of the subdomain polygon kml, shapefile or geopackage, or list with the
                uper-left x, upper-left y, lower-right x and lower-right y coordinates
            fout: string
                full path of the output fort.14
            epsg: int. Default 4326 (lat/lon)
                coordinate reference system of the mesh and mask layer
            sortBy: int. Default 1
                If 1, nodes are sorted by latitude since BC is aligned with N-S (vertical).
                If 0, nodes are sorted by longitude since BC is aligned with W-E (horizontal).
            rev: boolean. Default False
                This is needed to sort counter clockwise the ocean boundary nodes.
                If False, nodes are sorted in increasing order. E.g. north atlantic meshes
                IF True, nodes are sorted in decreasing order. E.g. pacific ocean where the shoreline is eastwards the ocean boundary.
            
    '''
    time00 = time.time()
    ## read nodes
    print('Mesh subset process started')
    dfNodes = readNodes_fort14(f14in)
    time01 = time.time()
    print(f'  Mesh nodes as DataFrame: {(time01 - time00)/60:0.2f} min')
    ## convert fort.14 to gdf
    gdfMesh = fort14togdf(f14in, epsg, epsg, fileintype = 'fort.14')
    time02 = time.time()
    print(f'  Mesh to GeoDataFrame: {(time02 - time01)/60:0.2f} min')
    ## read sub domain
    subDom = readSubDomain(subDomain, epsg)
    time03 = time.time()
    print(f'  Read subdomain: {(time03 - time02)/60:0.2f} min')
    ## subset mesh using subDomain
    _, meshSub, dfNodesNew = subsetMeshGdf(gdfMesh, dfNodes, subDom)
    time04 = time.time()
    print(f'  Subset mesh: {(time04 - time03)/60:0.2f} min')
    ## read fort.14 boundary conditions
    gdfbc, _ = readOrgBC(f14in, dfNodes)
    time05 = time.time()
    print(f'  Read fort.14 boundary conditions: {(time05 - time04)/60:0.2f} min')
    ## update the ocean boundary condition to match numbering of the subset mesh
    dfOpen, gdfOpen = renumOceanBC(gdfbc, dfNodesNew, sortBy, rev)
    time06 = time.time()
    print(f'  Update numbering ocean boundary condition: {(time06 - time05)/60:0.2f} min')
    ## update the land closed boundary conditions to match numbering of the subset mesh (islands)
    dctClosed, _ = renumClosedBCs(gdfbc, subDom, dfNodesNew)
    time07 = time.time()
    print(f'  Update numbering closed land boundary conditions: {(time07 - time06)/60:0.2f} min')
    ## update the mainland open boundary condition to match numbering of the subset mesh
    ## if more than one are merged
    dfMainland = renumMainlandBC(meshSub, gdfOpen, dfOpen, dfNodesNew, epsg)
    time07 = time.time()
    print(f'  Update numbering open land boundary conditions: {(time07 - time06)/60:0.2f} min')
    ## write new fort.14
    writeFort14(f14in, f14out, meshSub, dfNodesNew, dfOpen, dctClosed, dfMainland)
    time08 = time.time()
    print(f'  Writing new fort.14: {(time08 - time07)/60:0.2f} min')
    print(f'Done with fort.14 subset: {(time08 - time00)/60:0.2f} min')
    