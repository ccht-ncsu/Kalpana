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

## general plotting functions

def plot_netcdf(nc, var, levels, xlims = None, ylims = None, 
           ax = None, fig = None, fsize = (8, 6), cbar = False, 
           cmap = 'viridis', cb_shrink = 1, cb_label = None,
           ticks = None, background_map = True, point_circle = None):
    ''' Funtion to create 2D plots from netcdf files. WIP
            ## modification of plot2d from concorde
        Parameters
            nc: netcdf object
                adcirc file already loaded to memory to plot as contours
            var: string
                name of the variable to plot. E.g. 'zeta', 'zeta_max'
            levels: numpy array
                contours to plot
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
    
    nc = netcdf.Dataset(nc, 'r')
    if ax == None and background_map == False:
        fig, ax = plt.subplots(figsize = fsize)
    elif ax == None and background_map == True:
        fig, ax = plt.subplots(figsize = fsize, subplot_kw={'projection': ccrs.PlateCarree()}, 
                            constrained_layout=True)
    
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
        cb = fig.colorbar(contours, shrink = cb_shrink, extend = 'both', ax = ax, fraction=0.046, pad=0.04, ticks = ticks)
        cb.set_label(cb_label)
    
    if background_map == True:
        # show coordinates and grid
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--') 
        gl.top_labels = False
        gl.right_labels = False
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE,lw=0.25)
        ax.add_feature(cfeature.LAKES)

    if point_circle is not None: # add circle
        box = gpd.GeoDataFrame(geometry = [point_circle.buffer(6)])
        box.boundary.plot(ax = ax, color = 'k', ls = '--')

    return ax

def plot_gdfmesh(gdf, var = 'zmean', xylims = None, 
              ax = None, fig = None, fsize = (8,6), 
              cbar = True, cmap = None, cbar_label = None, 
              vmin = None, vmax = None, ticks = None, background_map = True):
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
        background_map: boolean
            True for using cartopy to plot a background map, doesn't work on the HPC
    Returns
        ax: matplotlib axes subplots
    '''

    if xylims is not None:
        mygdf = gpd.clip(gdf, [xylims[0]-0.5,xylims[2]-0.5,xylims[1]+0.5,xylims[3]+0.5])
    else:
        mygdf = gdf

    if ax == None:
        fig, ax = plt.subplots(figsize = fsize, subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
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

    if var == 'zmean':
        if cmap == None:

            cmap = merge_cmap(cmo.tools.crop_by_percent(cmo.ice, 30, 'max'), cmo.tools.crop_by_percent(cmo.speed_r, 25, 'max'))            
            offset = mcolors.TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax)

            mygdf.plot(column = var, legend = cbar, ax = ax, aspect = 'equal', cmap = cmap, norm = offset, vmin = vmin, vmax = vmax,
                legend_kwds={'label': cbar_label if cbar_label != None else None, 'orientation': 'vertical', 'fraction': 0.046, 'pad': 0.04, 'ticks': ticks if ticks is not None else None},
                )

    else:
        mygdf.plot(column = var, legend = cbar, ax = ax, aspect = 'equal', cmap = 'viridis' if cmap == None else cmap, edgecolor='black', linewidth = 0.2, vmin = vmin, vmax = vmax,
                legend_kwds={'label': cbar_label if cbar_label != None else None, 'orientation': 'vertical', 'fraction': 0.046, 'pad': 0.04, 'ticks': ticks if ticks is not None else None},
                )

    return ax

def merge_cmap(cmap1, cmap2):
    '''
    Combines two matplotlib colormaps into one continuous colormap.
    '''
    colors1 = cmap1(np.linspace(0, 1, 128))
    colors2 = cmap2(np.linspace(0, 1, 128))
    colors = np.vstack((colors1, colors2))
    return mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

## specific functions for example visualizations

def plot_mesh(gdf, var, xlims, ylims, title, cbar_label, vmin = None, vmax = None, ticks = None, background_map = True):
    fig, ax = plt.subplots(figsize = (8,6), nrows = 1, ncols = 1, subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
    plot_gdfmesh(gdf, var, xylims = xlims + ylims, ax = ax, fig = fig, fsize = (8,6), cbar_label = cbar_label, vmin = vmin, vmax = vmax, ticks = ticks, background_map = background_map)
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    fig.suptitle(title, fontsize = 16)

def plot_maxele(nc, levels):
    
    fig, ax = plt.subplots(figsize = (8,4), nrows = 1, ncols = 2,  subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
    plot_netcdf(nc, 'zeta_max', levels, ax = ax[0], cbar = False, point_circle = Point(-76.8, 35.2))
    ax[0].set_xlabel('Longitude [deg]')
    ax[0].set_ylabel('Latitude [deg]')
    ax[0].set_title('Full Domain')
    plot_netcdf(nc, 'zeta_max', levels, xlims = [-78.5, -75], ylims = [33.5, 37], ax = ax[1], fig = fig, cbar = True, cb_label = 'Max water level [m MSL]', ticks = np.arange(levels[0]-0.25, levels[1]+0.25, levels[2]))
    ax[1].set_xlabel('Longitude [deg]')
    ax[1].set_ylabel('Latitude [deg]')
    ax[1].set_title('North Carolina')
    fig.suptitle('Maximum Water Levels (Florence)', fontsize = 16)

## need to fix still

def plot_polylines(gdf, levels):
    fig, ax = plt.subplots(figsize = (9, 4.5), nrows = 1, ncols = 2, subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

    ##### subplot 0, 0

    # extrapolated functions from plot2D_v2 in concorde

    ax[0].set_xlabel('Longitude [deg]')
    ax[0].set_ylabel('Latitude [deg]')
    ax[0].set_xlim([-98, -60])
    ax[0].set_ylim([8, 46])

    # additional features of the plot
    nc = Point((-76.8, 35.2))
    box = gpd.GeoDataFrame(geometry = [nc.buffer(6)])
    box.boundary.plot(ax = ax[0], color = 'k', ls = '--') #draw circle
    gl = ax[0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--') # show coordinates and grid
    gl.top_labels = False
    gl.right_labels = False
    ax[0].add_feature(cfeature.LAND)
    ax[0].add_feature(cfeature.COASTLINE,lw=0.25)
    ax[0].add_feature(cfeature.LAKES)
    ax[0].set_title('Full Domain')


    ##### subplot 0, 1

    # plot2D_v2
    ax[1].set_xlabel('Longitude [deg]')
    ax[1].set_ylabel('Latitude [deg]')
    ax[1].set_xlim([-78.5, -75])
    ax[1].set_ylim([33.5, 37])

    # additional
    ax[1].add_feature(cfeature.LAND)
    ax[1].add_feature(cfeature.COASTLINE,lw=0.25)
    ax[1].add_feature(cfeature.LAKES)
    gl = ax[1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--') # show coordinates and grig
    gl.top_labels = False
    gl.right_labels = False
    ax[1].set_title('North Carolina')


    # polylines
    colors = [cm.viridis(x) for x in linspace(0, 1, int(levels[1]/levels[2])+1)]

    for index, row in gdf.iterrows():
        polyline = row['geometry']
        x, y = polyline.coords.xy
        x = x.tolist()
        y = y.tolist()
        ax[0].plot(x, y, row['z'], color = colors[int(row['z']/levels[2])])
        ax[1].plot(x, y, row['z'], color = colors[int(row['z']/levels[2])])

    clist = []
    for i, c in enumerate(colors):
        clist.append(mpatches.Patch(color = c, label = str(levels[0] + i*0.5)))
    ax[1].legend(handles = clist, loc='center left', bbox_to_anchor=(1, 0.5), title='Max water level \n      [m MSL]', alignment='center')

    fig.suptitle(f'Polyline Contours created from nc2shp()', fontsize = 16)

def plot_polygons(gdf, levels):
    fig, ax = plt.subplots(figsize = (8, 4), nrows = 1, ncols = 2, subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
    cmap=plt.cm.get_cmap('viridis', int(levels[1]/levels[2])) 
    ticks = np.arange(levels[0], levels[1], levels[2])

    ##### subplot 0, 0

    # extrapolated functions from plot2D_v2 in concorde

    ax[0].set_xlabel('Longitude [deg]')
    ax[0].set_ylabel('Latitude [deg]')
    ax[0].set_xlim([-98, -60])
    ax[0].set_ylim([8, 46])

    # additional features of the plot
    nc = Point((-76.8, 35.2))
    box = gpd.GeoDataFrame(geometry = [nc.buffer(6)])
    box.boundary.plot(ax = ax[0], color = 'k', ls = '--') #draw circle
    gl = ax[0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--') # show coordinates and grid
    gl.top_labels = False
    gl.right_labels = False
    ax[0].add_feature(cfeature.LAND)
    ax[0].add_feature(cfeature.COASTLINE,lw=0.25)
    ax[0].add_feature(cfeature.LAKES)
    ax[0].set_title('Full Domain')

    gdf.plot(column = 'zMean', legend = False, ax = ax[0], aspect = 'equal')


    ##### subplot 0, 1

    # plot2D_v2
    ax[1].set_xlabel('Longitude [deg]')
    ax[1].set_ylabel('Latitude [deg]')
    ax[1].set_xlim([-78.5, -75])
    ax[1].set_ylim([33.5, 37])

    # additional
    ax[1].add_feature(cfeature.LAND)
    ax[1].add_feature(cfeature.COASTLINE,lw=0.25)
    ax[1].add_feature(cfeature.LAKES)
    gl = ax[1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--') # show coordinates and grig
    gl.top_labels = False
    gl.right_labels = False
    ax[1].set_title('North Carolina')

    gdf.plot(column = 'zMean', legend = True, cmap = cmap, vmin = levels[0]-0.25, vmax = levels[1]-0.25,
            legend_kwds={'label': 'Max water level [m MSL]', 'orientation': 'vertical', 'fraction': 0.046, 'pad': 0.04, 'ticks': ticks}, 
            ax = ax[1], aspect = 'equal')

    fig.suptitle(f'Polygon Contours created from nc2shp()', fontsize = 16)

def polygon_compare(ncfile, levels, gdf):

    ## plot max flooding and gdf side by side

    f1 = ncfile
    nc1 = netcdf.Dataset(f1, 'r')
    fig, ax = plt.subplots(figsize = (9, 4.5), nrows = 1, ncols = 2, subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)
    cmap=plt.cm.get_cmap('viridis', int(levels[1]/levels[2])) 
    ticks = np.arange(levels[0], levels[1], levels[2])

    ##### subplot 0, 0

    # plot2D_v2
    tri = mpl.tri.Triangulation(nc1['x'][:].data, nc1['y'][:].data, nc1['element'][:,:] - 1)
    aux = nc1['zeta_max'][:].data
    aux = np.nan_to_num(aux, nan = -99999.0).reshape(-1)
    contours = ax[0].tricontourf(tri, aux, levels = np.arange(levels[0]-0.25, levels[1]+0.25, levels[2]), cmap = cmap)
    ax[0].set_xlabel('Longitude [deg]')
    ax[0].set_ylabel('Latitude [deg]')
    ax[0].set_xlim([-78.5, -75])
    ax[0].set_ylim([33.5, 37])
    cb = fig.colorbar(contours, extend = 'both', ax = ax[0], fraction=0.046, pad=0.04, ticks = ticks)
    cb.set_label('Max water level [m MSL]')

    # additional
    ax[0].add_feature(cfeature.LAND)
    ax[0].add_feature(cfeature.COASTLINE,lw=0.25)
    ax[0].add_feature(cfeature.LAKES)
    gl = ax[0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--') # show coordinates and grig
    gl.top_labels = False
    gl.right_labels = False
    ax[0].set_title('Florence Maximum Water Levels')


    ##### subplot 0, 1

    # plot2D_v2
    ax[1].set_xlabel('Longitude [deg]')
    ax[1].set_ylabel('Latitude [deg]')
    ax[1].set_xlim([-78.5, -75])
    ax[1].set_ylim([33.5, 37])

    # additional
    ax[1].add_feature(cfeature.LAND)
    ax[1].add_feature(cfeature.COASTLINE,lw=0.25)
    ax[1].add_feature(cfeature.LAKES)
    gl = ax[1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--') # show coordinates and grig
    gl.top_labels = False
    gl.right_labels = False
    ax[1].set_title('Polygon Contours from nc2shp()')

    gdf.plot(column = 'zMean', legend = True, cmap = cmap, vmin = levels[0]-0.25, vmax = levels[1]-0.25,
            legend_kwds={'label': 'Max water level [m MSL]', 'orientation': 'vertical', 'fraction': 0.046, 'pad': 0.04, 'ticks': ticks}, 
            ax = ax[1], aspect = 'equal')

def polyline_compare(ncfile, levels, gdf, lev):
    
    ## plot max flooding and gdf side by side

    f1 = ncfile
    nc1 = netcdf.Dataset(f1, 'r')
    fig, ax = plt.subplots(figsize = (10, 5), nrows = 1, ncols = 2, subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

    ##### subplot 0, 0

    # plot2D_v2
    tri = mpl.tri.Triangulation(nc1['x'][:].data, nc1['y'][:].data, nc1['element'][:,:] - 1)
    aux = nc1['zeta_max'][:].data
    aux = np.nan_to_num(aux, nan = -99999.0).reshape(-1)
    contours = ax[0].tricontourf(tri, aux, levels = levels, cmap = 'viridis')
    ax[0].set_xlabel('Longitude [deg]')
    ax[0].set_ylabel('Latitude [deg]')
    ax[0].set_xlim([-78.5, -75])
    ax[0].set_ylim([33.5, 37])
    cb = fig.colorbar(contours, extend = 'both', ax = ax[0], fraction=0.046, pad=0.04)
    cb.set_label('Max water level [m MSL]')

    # additional
    ax[0].add_feature(cfeature.LAND)
    ax[0].add_feature(cfeature.COASTLINE,lw=0.25)
    ax[0].add_feature(cfeature.LAKES)
    gl = ax[0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--') # show coordinates and grig
    gl.top_labels = False
    gl.right_labels = False
    ax[0].set_title('Florence Maximum Water Levels')

    ##### subplot 0, 1

    # plot2D_v2
    ax[1].set_xlabel('Longitude [deg]')
    ax[1].set_ylabel('Latitude [deg]')
    ax[1].set_xlim([-78.5, -75])
    ax[1].set_ylim([33.5, 37])

    # additional
    ax[1].add_feature(cfeature.LAND)
    ax[1].add_feature(cfeature.COASTLINE,lw=0.25)
    ax[1].add_feature(cfeature.LAKES)
    gl = ax[1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--') # show coordinates and grig
    gl.top_labels = False
    gl.right_labels = False
    ax[1].set_title('Polyline Contours from nc2shp()')


    # polylines
    colors = [cm.viridis(x) for x in linspace(0, 1, int(lev[1]/lev[2])+1)]

    for index, row in gdf.iterrows():
        polyline = row['geometry']
        x, y = polyline.coords.xy
        x = x.tolist()
        y = y.tolist()
        ax[1].plot(x, y, row['z'], color = colors[int(row['z']/lev[2])])

    clist = []
    for i, c in enumerate(colors):
        clist.append(mpatches.Patch(color = c, label = str(levels[0] + i*0.5)))
    ax[1].legend(handles = clist, loc='center left', bbox_to_anchor=(1, 0.5), title='Max water level \n      [m MSL]', alignment='center')

def plot_overlay(ncfile, levels, gdf):
    ## plot max flooding and gdf side by side

    f1 = ncfile
    nc1 = netcdf.Dataset(f1, 'r')
    fig, ax = plt.subplots(figsize = (5, 5), nrows = 1, ncols = 1, subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

    ##### subplot 0, 0 flooding map

    # plot2D_v2
    tri = mpl.tri.Triangulation(nc1['x'][:].data, nc1['y'][:].data, nc1['element'][:,:] - 1)
    aux = nc1['zeta_max'][:].data
    aux = np.nan_to_num(aux, nan = -99999.0).reshape(-1)
    contours = ax.tricontourf(tri, aux, levels = levels, cmap = 'viridis')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_xlim([-78.5, -75])
    ax.set_ylim([33.5, 37])
    cb = fig.colorbar(contours, extend = 'both', ax = ax, fraction=0.046, pad=0.04)
    cb.set_label('Max water level [m MSL]')

    # additional
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE,lw=0.25)
    ax.add_feature(cfeature.LAKES)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--') # show coordinates and grig
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title('Polylines over Florence Max Flooding Levels')

    ##### subplot 0, 0 polylines

    # polylines

    for index, row in gdf.iterrows():
        polyline = row['geometry']
        x, y = polyline.coords.xy
        x = x.tolist()
        y = y.tolist()
        ax.plot(x, y, row['z'], color = 'k', linewidth = 0.8)

    ax.legend([Line2D([0], [0], color='k', lw=1.1)], ['Polyline Contour'], loc = 'lower right')