# Examples

*Kalpana* has two main capabilities:

1. Visualization: export *ADCIRC* output as geospatial vector data for visualization using QGIS or any similar GIS software.
2. Downscaling: transform the *ADICRC* *maxele.63.nc* file to a constant and higher resolution DEM considering small scale topographic/bathymetric features (Rucker et. al 2021).

## Visualization 

Setup for running the visualization examples (export_exampleXX.ipynb):
1. Create a conda environment using the yml file provided in the *install* folder. See [the miniconda website](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
2. Clone the Kalpana repository to your local device. [Here](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) you can find the instructions.

**Export example 01** (export_example01.ipynb)<br>
Create contours as polygons based on the maximum flooding outputs from ADCIRC, then export the polygons as a .shp file.

**Export example 02** (export_example02.ipynb)<br>
Create contours as polylines based on the maximum flooding outputs from ADCIRC, then export the polylines as a .shp file.

**Export example 03** (export_example03.ipynb)<br>
Read fort.14 file as a GeoPandas GeoDataFrame, then export it as .shp file and some visualizations.

**More examples are comming soon**

## Downscaling

To use the downscaling method, it is required to have *GRASS GIS* installed. *Kalpana* uses *GRASS GIS* in the backend calling it from a python script. On a Linux OS, you just need to install/compile *GRASS GIS* (https://grass.osgeo.org/) and create a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) using the yml file provided in the *install* folder. On Windows it is more complicated. You need to install *GRASS GIS* and use the *Python* installation that comes with *GRASS GIS*. We will provide a short tutorial soon. To make it more user-friendly we made two *Docker* images to run the *Kalpana* downscaling tools. One of the images if for running the container interactively, and the other is non interactive. The instructions for using them are listed below:

**Non interactive**<br>
This image has all the necessary files and has been set up to downscale *ADCIRC* simulations using *NC9* mesh on a DEM of North Carolina. It is configured to run automatically, i.e. when running the container, the downscaling scripts are executed.
1. Install Docker, follow instructions [here](https://docs.docker.com/engine/install/).
2. Using the terminal, pull the Docker image from Docker hub with the command below. This image can be used only for running the downscaling for a simulation donw with NC9 in North Carolina. <br>
    'docker pull tacuevas/kalpana_nc:latest'
3. Create a folder, place the maxele.63.nc and runKalpanaStatic.inp files inside, and 'cd' to it. The *inp* file is provided in this folder, and the *ADCIRC* *maxele.63.nc* file can be found [here](https://go.ncsu.edu/kalpana-example-inputs).
4. Modify the file *runKalpanaStatic.inp* if you want to change the downscaling inputs (e.g. levels, crs, vertical unit, etc).
5. Run the container declaring a volume so kalpana can access the folder created in *step 3*. Before running the container, check you are located in the same folder where you placed the input files. We also provide a copy of the *Python* script executed with the container is ran (*runKalpanaStatic.py*)<br>
    'docker run -it -v "$(pwd)":/home/kalpana/inputs tacuevas/kalpana_nc:latest'
    

We are working to implement this same image for other areas of the US.

**Interactive**<br>
This image is configured to run kalpana interactively, all the python packages and *GRASS GIS* are installed. You need to copy the examples *downscaling_exampleXX.py* , the necessary inputs (availables [here](https://drive.google.com/drive/folders/1cbQzN4SrLs_rVlz9q8zHCKbFtQpLO5CG?usp=sharing)), and the *Kalpana* *downscaling.py* and *export.py* python modules from this repo to the container.


The steps for running the container:

1) Install Docker, follow instructions [here](https://docs.docker.com/engine/install/).
2) To pull the image from Docker hub, use the following command on the terminal: <br>
    'docker pull tacuevas/kalpana_m:latest'
3) Launch the container, use the following command on the terminal: <br>
    'docker run -it tacuevas/kalpana_m:latest'
4) *cp* all the files from your local device to the container. Follow instructions [here](https://docs.docker.com/engine/reference/commandline/cp/).
5) Run the python scripts from the Docker container with: <br>
    *python3 downscaling_exampleXX.py* 

Each example is explained below, remember to modify the paths!

**Example 01** (downscaling_example01.py)<br>
This script creates a grass location importing the DEM for downscaling and also creates a new DEM with same resolution and extend with the size of the mesh triangles. This step is key for the downscaling and can be run in advance, since it only depends on the mesh (*fort.14*).

**Example 02** (downscaling_example02.py)<br>
Example for doing the static downscaling using an existing grass location, and importing the DEM with the mesh elements size. Both inputs were created in the *example 01*. There is a short description of all inputs in the script, more detail can be found in the docstring of the function in the github repository.

**Example 03** (downscaling_example03.py)<br>
This example combines downscaling examples *1* and *2*. The DEM with mesh elements size and the grass location are created. This should be considerable more slow than running *example 2*, since creating the inputs for the downscaling is the slower part.

For questions regarding the visualization or downscaling examples please open an *Issue* or email tomascuevas@gmail.com
