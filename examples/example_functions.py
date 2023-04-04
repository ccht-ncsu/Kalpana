import geopandas as gpd
import numpy as np
from numpy import linspace
import netCDF4 as netcdf
from shapely.geometry import Point, LineString
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import cmocean.cm as cmo
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_maxele(ncfile, levels):
    
    # plot the maximum elevations of a given maxele.63.nc file
    # functions extrapolated from plot2D_v2 from concorde

    f1 = ncfile
    nc1 = netcdf.Dataset(f1, 'r')
    fig, ax = plt.subplots(figsize = (8, 4), nrows = 1, ncols = 2, subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)


    ##### subplot 0, 0

    # extrapolated functions from plot2D_v2 in concorde
    tri = mpl.tri.Triangulation(nc1['x'][:].data, nc1['y'][:].data, nc1['element'][:,:] - 1)
    aux = nc1['zeta_max'][:].data
    aux = np.nan_to_num(aux, nan = -99999.0).reshape(-1)
    contours = ax[0].tricontourf(tri, aux, levels = levels, cmap = 'viridis')
    ax[0].set_xlabel('Longitude [deg]')
    ax[0].set_ylabel('Latitude [deg]')

    # additional features of the plot
    nc = Point((-76.8, 35.2))
    box = gpd.GeoDataFrame(geometry = [nc.buffer(6)])
    box.boundary.plot(ax = ax[0], color = 'k', ls = '--')
    gl = ax[0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--') # show coordinates and grig
    gl.top_labels = False
    gl.right_labels = False
    ax[0].add_feature(cfeature.LAND)
    ax[0].add_feature(cfeature.COASTLINE,lw=0.25)
    ax[0].add_feature(cfeature.LAKES)
    ax[0].set_title('Full Domain')


    ##### subplot 0, 1

    # plot2D_v2
    tri = mpl.tri.Triangulation(nc1['x'][:].data, nc1['y'][:].data, nc1['element'][:,:] - 1)
    aux = nc1['zeta_max'][:].data
    aux = np.nan_to_num(aux, nan = -99999.0).reshape(-1)
    #print(aux)
    contours = ax[1].tricontourf(tri, aux, levels = levels, cmap = 'viridis')
    ax[1].set_xlabel('Longitude [deg]')
    ax[1].set_ylabel('Latitude [deg]')
    ax[1].set_xlim([-78.5, -75])
    ax[1].set_ylim([33.5, 37])
    cb = fig.colorbar(contours, extend = 'both', ax = ax[1], fraction=0.046, pad=0.04)
    cb.set_label('Max water level [m MSL]')

    # additional
    ax[1].add_feature(cfeature.LAND)
    ax[1].add_feature(cfeature.COASTLINE,lw=0.25)
    ax[1].add_feature(cfeature.LAKES)
    gl = ax[1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--') # show coordinates and grig
    gl.top_labels = False
    gl.right_labels = False
    ax[1].set_title('North Carolina')

    fig.suptitle(f'Maximum Water Levels (Florence)', fontsize = 16)

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

def mesh_plot(gdf, column, bounds, title, cmap = None):
    '''    
    x and y inputs must be in same coordinate system as gdf
    bounds = [minx, maxx, miny, maxy]
    automatically chooses topo plot features if column is zmean, viridis otherwise
    can override cmap with cmap
    '''

    trim = gdf[(gdf['centX'] <= bounds[0]-0.5) | (gdf['centX'] >= bounds[1]+0.5) | (gdf['centY'] <= bounds[2]-0.5) | (gdf['centY'] >= bounds[3]+0.5)].index
    mygdf = gdf.drop(trim, inplace=False)

    fig, ax = plt.subplots(figsize = (8, 6), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

    plt.xlabel('Longitude [deg]')
    plt.ylabel('Latitude [deg]')
    ax.set_xlim([bounds[0], bounds[1]])
    ax.set_ylim([bounds[2], bounds[3]])
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE,lw=0.25)
    ax.add_feature(cfeature.LAKES)
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    fig.suptitle(title, fontsize = 16)

    if column == 'zmean':
        if cmap == None:
            cmap = cmo.topo
        # plot the mesh with elevations correlation with topo colormap
        mygdf.plot(column = column, legend = True, ax = ax, aspect = 'equal', cmap = cmap, vmin = min(gdf['zmean']), vmax = -min(gdf['zmean']),
                legend_kwds={'label': 'Element Area [square km]', 'orientation': 'vertical', 'fraction': 0.046, 'pad': 0.04},
                )
    else:
        mygdf.plot(column = column, legend = True, ax = ax, aspect = 'equal', cmap = 'viridis', edgecolor='black', linewidth = 0.2,
                legend_kwds={'label': 'Element Area [square km]', 'orientation': 'vertical', 'fraction': 0.046, 'pad': 0.04},
                )

