import sys
import time
import warnings

warnings.filterwarnings("ignore")

#################################### INPUTS #####################################

# Path of the github repository. If the path has any whitespace please use quation marks (""). If the "kalpanaExport.py" script is in the folder where the code 
# is being executed, the path must be specified as "." instead (without the quation marks).
scriptPath = r'C:\Users\tacuevas\Documents\GitHub\kalpana_v2\code'

# Complete path of the adcirc output, must be a netcdf file (*.nc). If the path has any whitespace please use quation marks ("").
ncFile = r'C:\Users\tacuevas\NCSU\Research\kalpana\Florence\fort.63.nc'

# Name of the variable to export, e.g: 'zeta_max'.
var = 'zeta'

# Contour levels. Start, stop and step (stop not included). Values must be comma-delimited, do not use whitespaces in between.
levels = [0, 5, 0.5]

# Contours type. Only 'polyline' and 'polygon' are suported.
conType = 'Polygon'

# epsg code, e.g: 4326 for lat/lon.
epsg = 4326

# Complete path of the output file. Only shapefile (*.shp), geopackage (*.gpkg) and kmz files are suported.
pathOut = r'C:\Users\tacuevas\NCSU\Research\kalpana\py3\multiprocessing\outputs\A.shp'

# Number of process, if more than 1 and a the input is time-varying filepython multiprocessing Pool class is used.
# If you want to use all available processors, input npro equals to 999. 
npro = 1

# Complete path of the subdomain polygon kml or shapelfile. Input can also be a list with the uper-left x, upper-left y, lower-right x and lower-right y
# coordinates. Values must be comma-delimited, do not use whitespaces in between. THIS INPUT IS OPTIONAL, if not specified the complete domain will be exported.
subDom = None

####################################  END INPUTS #####################################

if __name__ == '__main__':

    now = time.time()

    sys.path.append(scriptPath)
    
    try:
        from kalpanaExport import nc2shp, nc2kmz
    except:
        print('kalpana module not found!')
        sys.exit(-1)

    if pathOut.endswith('.kmz'):
        gdf = nc2kmz(ncFile, var, levels, conType, epsg, pathOut, subDom)

    elif pathOut.endswith('.shp') or pathOut.endswith('.gpkp'):
        gdf = nc2shp(ncFile, var, levels, conType, epsg, pathOut, npro, subDom)

    else:
        print('Only ".kmz", ".shp" or ".gpkg" formats are suported!')
        sys.exit(-1)

    print(f'Script finished succsesfully after: {(time.time() - now)/60:0.3f} min')
