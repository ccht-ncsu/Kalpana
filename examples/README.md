# Examples

*Kalpana* has two main capabilities:

1. Visualization: convert *ADCIRC* output as geospatial vector data for visualization using QGIS or any similar GIS software.
2. Downscaling: transform the *ADCIRC* *maxele.63.nc* file to a higher-resolution raster (Rucker et. al 2021).<br>
2.1 Static: by considering small-scale topographic features.<br>
2.2 Head Loss: considering small-scale topographic features and energy loss due to bottom friction using a land cover dataset.<br>

## Visualization 

**Export example 01** (export_example01.ipynb)<br>
Create contours as polygons based on the maximum flooding outputs from *ADCIRC*, then export the polygons as a .shp file.

**Export example 02** (export_example02.ipynb)<br>
Create contours as polylines based on the maximum flooding outputs from *ADCIRC*, then export the polylines as a .shp file.

**Export example 03** (export_example03.ipynb)<br>
Read *fort.14* file as a GeoPandas GeoDataFrame, then export it as .shp file and some visualizations.

**Export example 04** (export_example03.ipynb)<br>
In this example, we will visualize the maximum flooding levels from a netcdf file using a mesh from a *fort.14* file as the background map.

**Export example 05** (export_example03.ipynb)<br>
visualize the water level from a time-varying netcdf (*fort.63.nc*) file.


## Downscaling

To make it more user-friendly, we made two ***Docker*** images to run the *Kalpana* downscaling tools. Users can skip the software installation with these images. One image is for running the container interactively, and the other image is non-interactive. The instructions for using them are listed below:

**Non interactive**<br>
This image has all the necessary files and has been set up to downscale *ADCIRC* simulations using the *NC9* mesh on a DEM of North Carolina. It is configured to run automatically, i.e. when running the container, the downscaling scripts are executed.
1) Install Docker, follow instructions [here](https://docs.docker.com/engine/install/).
2) Using the terminal, pull the Docker image from Docker hub with the command below. This image can be used only for running the downscaling for a simulation done with the NC9 mesh in North Carolina. <br>
    ```docker pull tacuevas/kalpana_nc:latest```
3) Create a folder, place the maxele.63.nc and runKalpanaStatic.inp files inside, and 'cd' to it. The *inp* file is provided in this folder, and the *ADCIRC* *maxele.63.nc* file can be found [here](https://go.ncsu.edu/kalpana-example-inputs).
4) Modify the file *runKalpanaStatic.inp* if you want to change the downscaling inputs (e.g. levels, crs, vertical unit, etc).
5) Run the container declaring a volume so kalpana can access the folder created in *step 3*. Before running the container, check you are located in the same folder where you placed the input files. We also provide a copy of the Python script executed when the container is ran (*runKalpanaStatic.py*)<br>
    ```docker run -it -v "$(pwd)":/home/kalpana/inputs tacuevas/kalpana_nc:latest```
6) This image only supports the *Static* downscaling method.


**Interactive**<br>
This image is configured to run kalpana interactively, all the Python packages and *GRASS GIS* are installed. You need to copy the examples *downscaling_exampleXX.py* , the necessary inputs (available [here](https://drive.google.com/drive/u/2/folders/14gOAzbfuMUk3asRFsMCtOup3NL3V6EgF)), and the *Kalpana* *downscaling.py* and *export.py* Python modules from this repo to the container.

The steps for running the container:

1) Install Docker, follow instructions [here](https://docs.docker.com/engine/install/).
2) To pull the image from Docker hub, use the following command on the terminal: <br>
    ```docker pull tacuevas/kalpana_m:latest```
3) Launch the container, use the following command on the terminal: <br>
    ```docker run -it tacuevas/kalpana_m:latest```
4) *cp* all the files from your local device to the container. Follow instructions [here](https://docs.docker.com/engine/reference/commandline/cp/).
5) Run the python scripts from the Docker container with: <br>
    ```python3 downscaling_exampleXX.py```
6) This image support both downscaling methods.

Each example is explained below, remember to modify the paths!

**Example 01** (downscaling_example01.py)<br>
This script creates a grass location by importing the DEM for downscaling. It also creates a new DEM with the same resolution and extent, but instead of the ground surface, the DEM represents the sizes of the mesh elements. This step is key for the downscaling and can be run in advance, because it only depends on the mesh (*fort.14*).

**Example 02** (downscaling_example02.py)<br>
Example for doing the static downscaling using an existing grass location, and importing the DEM with the mesh element sizes. Both inputs were created in the *example 01*. There is a short description of all inputs in the script, more detail can be found in the docstring of the function in the github repository.

**Example 02b** (downscaling_example02.py)<br>
Idem to previous example but in this case a levees Shapefile is provided. Kalpana uses this file to correct the downscaling by removing the flooded cells that overlap with the levees.

**Example 03** (downscaling_example03.py)<br>
This example combines downscaling examples *1* and *2*. The DEM with mesh elements size and the grass location are created. This should be considerably more slow than running *example 2*, because creating the inputs for the downscaling is the slower part.

**Example 04** (downscaling_example04.py)<br>
This example shows how to precompute the cost surface to run the *Head Loss* downscaling method. This is a separete process that needs to be run only once per DEM and before running the *Head Loss* method. Depending of the DEM resolution and extent, this process can take a very long time.

**Example 05** (downscaling_example05.ipynb)<br>
This example shows how to run the *Head Loss* downscaling method using data generated with the previous example.

For questions regarding the visualization or downscaling examples please open an *Issue* or email tomascuevas@gmail.com.
