## Functions utilized in the 'Examples' branch of the Kalpana repository
## Created by Brandon Tucker in 2023
## Mod by Tomas Cuevas in 2024

## General Plotting Functions: 
    # vis_netcdf: Plots contours one variable of a netcdf file.
    # vis_pgons: Plots polygon objects from a GeoDataframe object.
    # vis_plines: Plots polyline objects from a GeoDataframe object.
    # vis_mesh: Plots a mesh contained in a GeoDataframe object.
    # merge_cmap: Combines two matplotlib colormaps into one continuous colormap.


import geopandas as gpd
import numpy as np
from numpy import linspace
import netCDF4 as netcdf
from shapely.geometry import Point, LineString
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm
from sklearn.neighbors import KDTree

def res_nodes_for_plot_vectors(ncObj, dx, dy):
    ''' Resample nodes to plot vectors within the plot2D function
        Parameters
            ncObj: netcdf object. Default None
                adcirc file already loaded to memory to plot as vectors. E.g: wind or current vectors
            dx, dy: float. Default None
                spacing of the vectors
        Return:
            ind: list
                indices of the selected nodes
    '''
    maxx = np.max(ncObj['x'][:].data)
    minx = np.min(ncObj['x'][:].data)
    maxy = np.max(ncObj['y'][:].data)
    miny = np.min(ncObj['y'][:].data)
    xs = np.arange(minx, maxx + dx, dx)
    ys = np.arange(miny, maxy + dy, dy)
    X,Y = np.meshgrid(xs, ys)
    xravel = X.ravel()
    yravel = Y.ravel()
    
    nodes = list(zip(ncObj['x'][:].data, ncObj['y'][:].data))
    tree = KDTree(nodes)
    dist, ind = tree.query(list(zip(xravel, yravel)))
    ind = ind.reshape(-1)
    
    return ind


def plot_nc(nc, var, levels, ncvec = None, dxvec = None, dyvec = None, 
           vecsc = None, veccolor = 'k', xlims = None, ylims = None, 
           cbar = False, ts = None, ax = None, fig = None, 
           cmap = 'viridis', fsize = (8, 6), cb_shrink = 1, cb_label = None, 
           background_map = False, extend = 'neither'):
    
    ''' Funtion to create 2D plots from netcdf files
        Parameters
            nc: netcdf object
                adcirc file already loaded to memory to plot as contours
            var: string
                name of the variable to plot. E.g. 'zeta', 'zeta_max'
            levels: numpy array
                contours to plot
            ncvec: netcdf object. Default None
                adcirc file already loaded to memory to plot as vectors. E.g: wind or current vectors
            dxvec, dyvec: float. Default None
                spacing of the vectors
            vecsc: int or float. Default None
                scale of the vectors.
            veccolor: string. Default black ('k')
                color of the vectors
            xlims, ylims: list
                limits of the plot
            cbar: boolean. Default False
                True for adding colorbar
            gdf: GeoDataFrame
                usefull to overlay info, check if this not enough for ploting the track? REVIEW
            ts: int
                timestep for plotting time-varying adcirc outputs
            ax: matplotlib axis
            fig: matplotlib figure
            cmap: string
                name of the cmap. For maxele viridis is recommended, but for fort.63 seismic works well
            fsize: tuple
                figure of the output size if fig and ax are not specified
            cb_shrink: float
                useful to define size of colorbar
            cb_label: string
                colorbar label
            background_map: boolean
                True for using cartopy to plot a background map, doesn't work on the HPC
            extend: str
                'neither', 'both', 'min', 'max'
        Returns
            ax: matplotlib ax
    '''
    
    tri = mpl.tri.Triangulation(nc['x'][:].data, nc['y'][:].data, nc['element'][:,:] - 1)
    if ts == None:
        aux = nc[var][:].data
    else:
        aux = nc[var][ts, :].data
    
    aux = np.nan_to_num(aux, nan = -99999.0).reshape(-1)
    if ax == None and background_map == False:
        fig, ax = plt.subplots(figsize = fsize)
    elif ax == None and background_map == True:
        fig, ax = plt.subplots(figsize = fsize, subplot_kw={'projection': ccrs.PlateCarree()}, 
                            constrained_layout=True)

    contours = ax.tricontourf(tri, aux, levels = levels, cmap = cmap, extend = extend)
    
    if ncvec is not None:
        ## plot vectors
        if dxvec is None and dyvec is None:
            ax.quiver(ncvec['x'], ncvec['y'], ncvec['windx'][ts, :], ncvec['windy'][ts, :], scale=vecsc)
        else:
            nodes_to_plot = res_nodes_for_plot_vectors(ncvec, dxvec, dyvec)
            ax.quiver(ncvec['x'][nodes_to_plot], ncvec['y'][nodes_to_plot], 
                      ncvec['windx'][ts, nodes_to_plot], ncvec['windy'][ts, nodes_to_plot], scale=vecsc, color = veccolor)
    
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    if cbar == True:
        cb = fig.colorbar(contours, shrink = cb_shrink, ax = ax)
        cb.set_label(cb_label)
    
    if background_map == True:
        # show coordinates and grid
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--') 
        gl.top_labels = False
        gl.right_labels = False
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE,lw=0.25)
        ax.add_feature(cfeature.LAKES)

    return ax

def plot_kalpana_pgons(gdf, levels, var = 'zMean', xlims = None, ylims = None,
              ax = None, fig = None, fsize = (8,6), 
              cbar = True, cmap = 'viridis', cbar_label = None, 
              background_map = True):
    '''Plots polygon objects from a GeoDataframe file.
    Parameters
        gdf: GeoDataframe object
            usually created from nc2shp() in Kalpana
        levels: numpy array
            contours to plot
        var: str
            name of the column to plot
        xlims, ylims: list
            limits of the plot
        ax: matplotlib axis
        fig: matplotlib figure
        fsize: tuple
            figure of the output size if fig and ax are not specified
        cbar: boolean
            True for adding colorbar
        cmap: string
            name of the cmap. recommended None to use custom colormap built into function
        cbar_label: string
            colorbar label
        background_map: boolean
            True for using cartopy to plot a background map, doesn't work on the HPC
    Returns
        ax: matplotlib axes subplots
    '''

    if ax is None and background_map == False:
        fig, ax = plt.subplots(figsize = fsize)
    
    elif ax is None and background_map == True:
        fig, ax = plt.subplots(figsize = fsize, subplot_kw={'projection': ccrs.PlateCarree()}, 
                            constrained_layout=True)
        
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    if background_map == True:
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE,lw=0.25)
        ax.add_feature(cfeature.LAKES)

    step = levels[1] - levels[0]
    mycmap = plt.cm.get_cmap(cmap, int(levels[-1]/step))    

    #plot polygons
    if cbar == False:
        gdf.plot(column = var, legend = False, ax = ax, aspect = 'equal')
    
    else:
        gdf.plot(column = var, legend = True, cmap = mycmap, vmin = levels[0], vmax = levels[-1],
            legend_kwds={'label': cbar_label, 'orientation': 'vertical', 'fraction': 0.046, 'pad': 0.04, 'ticks': levels}, 
            ax = ax, aspect = 'equal')
    
    return ax

def vis_plines(gdf, levels, xlims = None, ylims = None,
              ax = None, fig = None, fsize = (8,6), 
              cbar = True, cmap = cm.viridis, cbar_label = None, outline = False, 
              ticks = None, background_map = True, point_circle = None):
    '''Plots polyline objects from a GeoDataframe file.
    Parameters
        gdf: GeoDataframe object
            usually created from nc2shp() in Kalpana
        levels: list to define levels
            [min, max, step]
            must be length 3
        xlims, ylims: list
            limits of the plot
        ax: matplotlib axis
        fig: matplotlib figure
        fsize: tuple
            figure of the output size if fig and ax are not specified
        cbar: boolean
            True for adding colorbar
        cmap: string
            name of the cmap. recommended None to use custom colormap built into function
        cb_label: string
            colorbar label
        outline: boolean. default False
            True to make polylines thin black lines
        ticks: list
            colorbar ticks
        background_map: boolean
            True for using cartopy to plot a background map, doesn't work on the HPC
        point_circle: shapely Point
                draws a circle around the point
    Returns
        ax: matplotlib axes subplots
    '''


    if ax == None and background_map == False:
        fig, ax = plt.subplots(figsize = fsize)
    elif ax == None and background_map == True:
        fig, ax = plt.subplots(figsize = fsize, subplot_kw={'projection': ccrs.PlateCarree()}, 
                            constrained_layout=True)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    if point_circle is not None:
        box = gpd.GeoDataFrame(geometry = [point_circle.buffer(6)])
        box.boundary.plot(ax = ax, color = 'k', ls = '--')

    if background_map == True:
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE,lw=0.25)
        ax.add_feature(cfeature.LAKES)
    
    if ticks == None:
        ticks = np.arange(levels[0], levels[1], levels[2])

    #plot polylines
    colors = [cmap(x) for x in linspace(0, 1, int(levels[1]/levels[2])+1)]
    for index, row in gdf.iterrows():
        polyline = row['geometry']
        x, y = polyline.coords.xy
        x = x.tolist()
        y = y.tolist()
        ax.plot(x, y, row['z'], color = 'k' if outline else colors[int(row['z']/levels[2])], linewidth = 0.8 if outline else 1.5)

    #colorbar  
    if outline:
        ax.legend([Line2D([0], [0], color='k', lw=1.1)], ['Polyline Contour'], loc = 'lower right')
    elif cbar:
        clist = []
        for i, c in enumerate(colors):
            clist.append(mpatches.Patch(color = c, label = str(levels[0] + i*0.5)))
        ax.legend(handles = clist, loc='center left', bbox_to_anchor=(1, 0.5), title='Max water level \n      [m MSL]', alignment='center')
    
    return ax

def plot_mesh(gdf, var = 'zmean', bbox = None, ax = None, fig = None, fig_size = (8, 6), cbar = True, cbar_label = 'Elevation [mMSL]', background = True,
              cmap = cmocean.cm.speed, lw = 0.01, elem_color = 'k', vmax = None, vmin = None, vcenter = None, xlabel = None, ylabel = None, alpha = 1):
    ''' Function to plot the ADCIRC mesh as GeoDataFrame (output of fort14togdf())
        Paramters
            gdf: GeoDataFrame
                output of fort14togdf()
            var: str. Default 'zmean'
                column name of the variable to plot
            bbox: GeoDataFrame, GeoSeries, (Multi)Polygon, list-like. Default None
                Polygon vector layer used to clip
            ax: Matplotlib axis. Default None
            fig: Matplotlib figure. Default None
            figsize: tuple. Default (8, 6)
                Size of the figure
            cbar: boolean. Default True
                True for plotting colorbar
            cbar_label: str. Default 'Elevation [mMSL]'
                Label of the colorbar
            background: boolean. Default True
                True for plotting background map with Cartopy, only works in lat/lon (epsg 4326)
            cmap: colormap. Default cmocean.cm.speed
            lw: float or int. Default 0.01
                with of the elements edge
            elem_color: str. Default 'k'
                color of the elements edge
            vmax: int or float. Default None
                maximum value to plot
            vmin: int or float. Default None
                minimum value to plot
            vcenter: int or float. Default None
                Center value of the colorbar, helpful when plotting topo and bathy, calls TwoSlopeNorm of matplotlib
            xlabel, ylabel: str. Default None
                x and y-axis labels
            alpha: float or int. Default 1
                Transparency
        Returns
            ax: Matplotlib ax

    '''


    if bbox is None:
        gdf2 = gdf.copy()
    else:
        gdf2 = gpd.clip(gdf, bbox)

    if ax == None and background == False:
        fig, ax = plt.subplots(figsize = fig_size)
        ax.grid(alpha = 0.5)
        ax.set_aspect('equal')
    
    elif ax == None and background == True:
        fig, ax = plt.subplots(figsize = fig_size, subplot_kw={'projection': ccrs.PlateCarree()}, 
                            constrained_layout=True)
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE,lw=0.25)
        ax.add_feature(cfeature.LAKES)
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    else:
        pass
    
    if vcenter is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = None

    gdf2.plot(column = var, legend = cbar, ax = ax, aspect = 'equal', cmap = cmap, edgecolor=elem_color, linewidth = lw, vmax = vmax, vmin = vmin, 
              norm = norm, alpha = alpha,
              legend_kwds={'label': cbar_label if cbar_label != None else None, 'orientation': 'vertical', 'fraction': 0.046, 'pad': 0.04})
    
    if xlabel is not None and ylabel is not None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    return ax