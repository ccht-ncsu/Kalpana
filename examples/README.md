# Examples

*Kalpana* has two main capabilities:

1. Visualization: export *ADCIRC* output as geospatial vector data for visualization using QGIS or any similar GIS software.
2. Downscaling: transform the *ADICRC* *maxele.63.nc* file to a constant and higher resolution DEM considering small scale topographic/bathymetric features (Rucker et. al 2021).

## Visualization *If I was a new user, I would get stuck here - either unfamiliar with conda or git cloning. Here are some edits:*
Setup for running the visualization examples (export_exampleXX.ipynb):
1. Create a conda environment using the yml file provided in the *install* folder. See **this website** for instructions.
2. Clone the Kalpana repository to your local device. See **this webiste** for instructions.

**Example 01** (export_example01.ipynb)<br>
In this example, we will create contours as polygons based on the maximum flooding outputs from ADCIRC,
then export those polygons as a .shp file.

**More examples are comming soon**

## Downscaling

To use the downscaling method, it is required to have *GRASS GIS* installed. *Kalpana* uses *GRASS GIS* in the backend calling it from a python script. On a Linux OS, you just need to install/compile *GRASS GIS* (https://grass.osgeo.org/) and create a conda environment using the yml file provided in the *install* folder. On Windows it is more complicated, and the details on how to do it will come soon! To avoid this issue we have Docker images to run the downscaling tool of *Kalpana*.
The instructions for using them are listed below:


**Non interactive**<br>
This image has all the necessary files and has been set up to downscale *ADCIRC* simulations with *NC9* mesh on a DEM of North Carolina. It is configured to run automatically.
1. Pull the Docker image from Docker hub
'docker pull tacuevas/kalpana_nc:latest'
2. Create a folder, place the maxele.63.nc and runKalpanaStatic.inp files inside, and 'cd' to it. Examples of these two files can be found in [this google drive](https://drive.google.com/drive/folders/1cbQzN4SrLs_rVlz9q8zHCKbFtQpLO5CG?usp=sharing).
3. Modify runKalpanaStatic.inp if neccesary.
4. Run the container declaring a volume so kalpana can access the folder created in *step 2*.
'docker run -it -v "$(pwd)":/home/kalpana/inputs tacuevas/kalpana_nc:latest'

**Interactive**<br>
This image is configured to run kalpana interactively, all the python packages and *GRASS GIS* are installed. You need to copy the examples *downscaling_exampleXX.py*, the necessary inputs (availables [here](https://drive.google.com/drive/folders/1cbQzN4SrLs_rVlz9q8zHCKbFtQpLO5CG?usp=sharing)), and the *Kalpana* *downscaling.py* and *export.py* python modules from this repo to the container.


The steps for running are:

1) 'docker pull tacuevas/kalpana_m:latest'
2) 'docker run -it tacuevas/kalpana_m:latest'
3) *cp* all the files
4) *python3 downscaling_exampleXX.py* 

Each example is explained below:

**Example 01** (downscaling_example01.py)<br>
This script creates a grass location importing the DEM for downscaling and also creates a new DEM with same resolution and extend with the size of the mesh triangles. This step is key for the downscaling and can be run in advance, since only depends on the mesh.

**Example 02** (downscaling_example02.py)<br>
Example for doing the static downscaling using an existing grass location, and importing the DEM with the mesh elements size. Both inputs were created in the example_01. There is a short description of all inputs below, more detail can be found in the docstring of the function in the github repository.

**Example 03** (downscaling_example03.py)<br>
This example combine downscaling examples 1 and 2. DEM with mesh elements size and the grass location are created. This should be considerable more slow than running example 2, since creating the inputs for the downscaling is the slower part.
