## Functions utilized in the 'Examples' branch of the Kalpana repository
## Created by Brandon Tucker in 2023

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
import cmocean.cm as cmo
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def vis_netcdf(nc, var, levels, xlims = None, ylims = None, 
           ax = None, fig = None, fsize = (8, 6), 
           cbar = False, cmap = 'viridis', cbar_label = None,
           ticks = None, background_map = True, point_circle = None):
    ''' Funtion to create 2D plots from netcdf files.
            ## modification of plot2d from concorde
        Parameters
            nc: netcdf object
                adcirc file already loaded to memory to plot as contours
            var: string
                name of the variable to plot. E.g. 'zeta', 'zeta_max'
            levels: list to define contours
                [min, max, step]
                must be length 3
            xlims, ylims: list
                limits of the plot
            ax: matplotlib axis
            fig: matplotlib figure
            fsize: tuple
                figure of the output size if fig and ax are not specified
            cbar: boolean. Default False
                True for adding colorbar
            cmap: string
                name of the cmap. For maxele viridis is recommended, but for fort.63 seismic works well
            cb_shrink: float
                useful to define size of colorbar
            cbar_label: string
                colorbar label
            ticks: list
                colorbar ticks
            background_map: boolean
                True for using cartopy to plot a background map, doesn't work on the HPC
            point_circle: shapely Point
                draws a circle around the point
        Returns
            ax: matplotlib axes subplots
    '''
    
    nc = netcdf.Dataset(nc, 'r')
    if ax == None and background_map == False:
        fig, ax = plt.subplots(figsize = fsize)
    elif ax == None and background_map == True:
        fig, ax = plt.subplots(figsize = fsize, subplot_kw={'projection': ccrs.PlateCarree()}, 
                            constrained_layout=True)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--') 
    gl.top_labels = False
    gl.right_labels = False
    
    mycmap=plt.cm.get_cmap(cmap, int(levels[1]/levels[2])) 
    if ticks is None:
        ticks = np.arange(levels[0], levels[1], levels[2])

    #create flood contours
    tri = mpl.tri.Triangulation(nc['x'][:].data, nc['y'][:].data, nc['element'][:,:] - 1)
    aux = nc[var][:].data
    aux = np.nan_to_num(aux, nan = -99999.0).reshape(-1)
    contours = ax.tricontourf(tri, aux, levels = np.arange(levels[0]-0.25, levels[1]+0.25, levels[2]), cmap = mycmap)
    
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    if cbar:
        cb = fig.colorbar(contours, extend = 'both', ax = ax, fraction=0.046, pad=0.04, ticks = ticks)
        cb.set_label(cbar_label)
    
    if background_map == True:        
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE,lw=0.25)
        ax.add_feature(cfeature.LAKES)

    if point_circle is not None: # add circle
        box = gpd.GeoDataFrame(geometry = [point_circle.buffer(6)])
        box.boundary.plot(ax = ax, color = 'k', ls = '--')

    return ax

def vis_pgons(gdf, levels, xlims = None, ylims = None,
              ax = None, fig = None, fsize = (8,6), 
              cbar = True, cmap = 'viridis', cbar_label = None, 
              ticks = None, background_map = True, point_circle = None):
    '''Plots polygon objects from a GeoDataframe file.
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

    mycmap=plt.cm.get_cmap(cmap, int(levels[1]/levels[2])) 
    
    if ticks == None:
        ticks = np.arange(levels[0], levels[1], levels[2])

    #plot polygons
    if cbar == False:
        gdf.plot(column = 'zMean', legend = False, ax = ax, aspect = 'equal')
    else:
        gdf.plot(column = 'zMean', legend = True, cmap = mycmap, vmin = levels[0]-0.25, vmax = levels[1]-0.25,
            legend_kwds={'label': cbar_label, 'orientation': 'vertical', 'fraction': 0.046, 'pad': 0.04, 'ticks': ticks}, 
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

def vis_mesh(gdf, var = 'zmean', xylims = None, 
              ax = None, fig = None, fsize = (8,6), 
              cbar = True, cmap = None, cbar_label = None, 
              vmin = None, vmax = None, ticks = None, 
              topo = False, background_map = True):
    ''' Funtion to create 2D plots from a GeoDataframe that represents a mesh.
    Parameters
        gdf: GeoDataframe object
            usually created from fort14togdf()
        var: string
            name of the variable to plot. E.g. 'zeta', 'zeta_max'
        xylims: list
            limits of the plot
            [minx, maxx, miny, maxy]
        ax: matplotlib axis
        fig: matplotlib figure
        fsize: tuple
            figure of the output size if fig and ax are not specified
        cbar: boolean. Default False
            True for adding colorbar
        cmap: string
            name of the cmap. recommended None to use custom colormap built into function
        cb_label: string
            colorbar label
        ticks: list
            colorbar ticks
        topo: boolean
            True to only plot elevations greater than 0.
        background_map: boolean
            True for using cartopy to plot a background map, doesn't work on the HPC
    Returns
        ax: matplotlib axes subplots
    '''

    if xylims is not None:
        mygdf = gpd.clip(gdf, [xylims[0]-0.5,xylims[2]-0.5,xylims[1]+0.5,xylims[3]+0.5])
    else:
        mygdf = gdf

    if topo:
        mygdf = mygdf[mygdf['zmean'] > 0]

    if ax == None and background_map == False:
        fig, ax = plt.subplots(figsize = fsize)
    elif ax == None and background_map == True:
        fig, ax = plt.subplots(figsize = fsize, subplot_kw={'projection': ccrs.PlateCarree()}, 
                            constrained_layout=True)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    if xylims is not None:
        ax.set_xlim(xylims[0], xylims[1])
        ax.set_ylim(xylims[2], xylims[3])

    if background_map == True:
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE,lw=0.25)
        ax.add_feature(cfeature.LAKES)

    vmin = vmin if vmin is not None else min(mygdf[var])
    vmax = vmax if vmax is not None else max(mygdf[var])
    ticks = ticks if ticks is not None else None

    if var == 'zmean' and cmap == None:
        if topo:
            cmap = cmo.tools.crop_by_percent(cmo.speed_r, 25, 'max')
            offset = None
        else:
            cmap = merge_cmap(cmo.tools.crop_by_percent(cmo.ice, 30, 'max'), cmo.tools.crop_by_percent(cmo.speed_r, 25, 'max'))            
            offset = mcolors.TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax)

        mygdf.plot(column = var, legend = cbar, ax = ax, aspect = 'equal', cmap = cmap, norm = offset if topo else None, vmin = vmin, vmax = vmax,
            legend_kwds={'label': cbar_label if cbar_label != None else None, 'orientation': 'vertical', 'fraction': 0.046, 'pad': 0.04, 'ticks': ticks},
            )
            
    else:
        mygdf.plot(column = var, legend = cbar, ax = ax, aspect = 'equal', cmap = cmap, edgecolor='black', linewidth = 0.2, vmin = vmin, vmax = vmax,
                legend_kwds={'label': cbar_label if cbar_label != None else None, 'orientation': 'vertical', 'fraction': 0.046, 'pad': 0.04, 'ticks': ticks},
                )

    return ax

def merge_cmap(cmap1, cmap2):
    ''' Combines two matplotlib colormaps into one continuous colormap.
    Parameters
        cmap1: matplotlib colormap
            bottom half of the new colormap
        cmap2: matplotlib colormap
            top half of the new colormap
    Returns
        cmap object
    '''
    colors1 = cmap1(np.linspace(0, 1, 128))
    colors2 = cmap2(np.linspace(0, 1, 128))
    colors = np.vstack((colors1, colors2))
    return mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
