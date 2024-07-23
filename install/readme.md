## Kalpana installation

The first step is to clone the Kalpana repository to your local machine. You can find the instructions [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository). <br>

The following steps depend on your OS and what features of Kalpana you want to use.<br>

On ***Linux***:<br><br>

1. Create a conda environment. See [the miniconda website](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Kalpana has been tested with Python 3.11. You can use mamba instead if you prefer.<br>
   ```conda create --name kalpana python=3.10```<br>
2. Activate the conda environment.<br>
   ```conda activate kalpana```<br>
3. Within the conda environment, navigate to the Kalpana GitHub repository.<br>
   ```cd Kalpana```<br>
4. Install ***Kalpana*** dependencies using pip.<br>
   ```pip install -e .```<br>
5. Then you can use ***Kalpana*** within that conda environment to visualize ADCIRC results (e.g. examples in '2_adcirc_to_vector'). However, to use the downscaling tools, you also need to install GRASS GIS (https://grass.osgeo.org/). GRASS versions >= 8.2 are supported.<br><br>

On ***Windows***:<br><br>
If you just want to use the visualization functions or export the NetCDF as GIS files, use steps 1 and 2 of the Linux installation.<br>

If you want to use the downscaling tools, You need to use the Python that comes with GRASS GIS, and you can not have more Python versions on your system. Follow the steps below to install Kalpana:<br>
1. Install GRASS GIS (ttps://grass.osgeo.org/). Versions >= 8.2 are supported.<br>
2. Launch GRASS GIS, close the GUI and continue using the GRASS GIS cmd.
3. Navigate to the GitHub repo using the GRASS GIS cmd.<br>
```python -m pip install -e .```
4. To use Jupyter Notebooks, you have to install it with pip <br>
   ```python -m pip install notebook```
5. For using Jupyter Notebooks with the GRASS GIS Python installation paste the lines below on the GRASS GIS cmd:<br>
```set PATH=%PATH%;C:\Program Files\GRASS GIS X.X\```<br>
```set PATH=%PATH%;C:\Program Files\GRASS GIS X.X\Python39\Scripts\```<br>
```jupyter notebook```<br>
(Replace X.X by the version of GRASS you have installed)


THIS IS NOT UP-TO-DATE<be><br>
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
