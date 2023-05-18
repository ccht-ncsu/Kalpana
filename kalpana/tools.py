import os
import numpy as np
import pandas as pd
from PIL import Image
import rioxarray as rxr
from pathlib import Path
from itertools import islice

def demToPNG(fileIn, pathOut, noData = 255, tileSize = 5_000):
    ''' Transform DEM to png
        Parameters
            fileIn: str
                full path of the input DEM
            pathOut: str
                path of the folder to save the png files
            noData: int. Default 255
                pixel value for the non data DEM cells
            tileSize.int Default 5_000
                target size of the output pngs. If equals to -1,
                full DEM is exported as one png.
        Output
            if tileSize !== -1:
                out: list
                    tiled pngs
                demAux: numpy array
                    full DEM as numpy array
    '''
    pathOut = Path(pathOut)
    dem = rxr.open_rasterio(fileIn)
    ## get nodata value of dem
    noDataDem = dem.rio.nodata
    ## get only first channel, outputs of kalpana are
    ## greyscale images or tifs with only one channel
    demAux = dem[0,:,:].data
    ## replace that value with "noData" input
    demAux[demAux == noDataDem] = noData
    ## change data type from float64 to int8
    demAux = demAux.astype('uint8')
    if tileSize != -1:
        ## indices of the tiles
        xs = np.arange(0, dem['x'].size, tileSize)
        ys = np.arange(0, dem['y'].size, tileSize)
        ## get number of tiles to  generate
        n = len(xs) * len(ys)
        n = len(str(n))
        counter = 0
        out = []
        dummy = []
        for x in xs:
            for y in ys:
                dummy.append(counter)
                ## get tiles of the dem
                aux = demAux[y:y + tileSize, x:x + tileSize]
                ## reshape to drop 3rd dimension
                tileNp = aux.reshape((aux.shape[0], aux.shape[1]))
                ## transform it to image
                tileImg = Image.fromarray(tileNp, mode = 'L')
                out.append(tileNp)
                ## save tile png
                tileImg.save(pathOut/f'{os.path.splitext(os.path.basename(fileIn))[0]}_tile{counter:0{n}d}.png')
                counter += 1
        ## save how the dem was tiled
        dummy = np.reshape(dummy, (len(xs), len(ys)))
        np.savetxt(pathOut/f'{os.path.splitext(os.path.basename(fileIn))[0]}_tilesOrder.txt',
                  dummy.astype(int), fmt = '%i', delimiter = ",")
        return out
    else:
        ## transform complete DEM to png
        img = Image.fromarray(demAux, mode = 'L')
        img.save(pathOut/f'{os.path.splitext(os.path.basename(fileIn))[0]}.png')
        return demAux
        
def mergeTiles(pathIn, txtFile, fileOut):
    ''' Fx to merge tiles generated with demToPNG
        Parameters
            pathin: str
                path to tiled PNG dems, not including file name.
            txtFile: str
                full path of txt with the ordering of the tiles.
            fileOut: str
                full output path of the merged PNG (including file name).
    '''
    pathIn = Path(pathIn)
    aux = pd.read_csv(txtFile, header = None)
    aux2 = aux.values
    list_h = []
    counter = 0
    listFiles = sorted([x for x in os.listdir(pathIn) if x.endswith('.png') and 'tile' in x])
    
    for i in range(aux.shape[0]):
        list_v = []
        for j in range(aux.shape[1]):
            im = Image.open(pathIn/listFiles[aux2[i,j]])
            im_arr = np.array(im)
            list_v.append(im_arr)
            counter += 1
        aux_v = np.concatenate(list_v, axis = 0)
        list_h.append(aux_v)
    img_arr_all = np.concatenate(list_h, axis = 1)
    im_all = Image.fromarray(img_arr_all)
    im_all.save(fileOut)
    
    return img_arr_all

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
