## Kalpana installation

The first step is to clone the *Kalpana* repository to your local machine. You can find the instructions [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository). <br>

Installation depends on how you want to use *Kalpana*. For visualization of ADCIRC outputs as vector products (e.g. GIS shapefiles), then you need to create a Conda environment and install *Kalpana* dependencies. For downscaling of ADCIRC outputs to higher resolutions, then you also need to install GRASS GIS (https://grass.osgeo.org/) and make it available to *Kalpana*.<br>

The following steps depend on your OS and what features of *Kalpana* you want to use.<br>

On ***Linux***:<br>
1. Create a conda environment:<br>
a. See [the miniconda website](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). *Kalpana* has been tested with Python 3.11. You can use mamba instead if you prefer.<br>
      ```conda create --name kalpana python=3.11```<br>
b. Activate the conda environment.<br>
      ```conda activate kalpana```<br>
c. Within the conda environment, navigate to the *Kalpana* GitHub repository.<br>
      ```cd Kalpana```<br>
d. Install *Kalpana* dependencies using pip.<br>
      ```pip install -e .```<br>
f. Then you can use *Kalpana* within that conda environment to visualize ADCIRC results (e.g. examples in '2_adcirc_to_vector').
2. To use the downscaling tools, you also need to install GRASS GIS (https://grass.osgeo.org/). GRASS versions >= 8.2 are supported. Nothing special is needed to link GRASS to *Kalpana*, but the GRASS executable needs to be available/findable in your path.<br>

On ***Windows***:<br>
1. If you just want to use the visualization functions or export the NetCDF as GIS files, then use step 1 and 2 of the Linux installation.<br>
2. If you want to use the downscaling tools, then you need to use the Python that comes with GRASS GIS, and you cannot have more Python versions on your system. Follow the steps below to install *Kalpana*:<br>
a. Install GRASS GIS (https://grass.osgeo.org/). Versions >= 8.2 are supported.<br>
b. Launch GRASS GIS, close the GUI and continue using the GRASS GIS cmd.<br>
c. Navigate to the GitHub repo using the GRASS GIS cmd.<br>
      ```python -m pip install -e .```<br>
d. To use Jupyter Notebooks, you have to install it with pip <br>
      ```python -m pip install notebook```<br>
e. For using Jupyter Notebooks with the GRASS GIS Python installation paste the lines below (but replace X.X by the version of GRASS you have installed) on the GRASS GIS cmd:<br>
      ```set PATH=%PATH%;C:\Program Files\GRASS GIS X.X\```<br>
      ```set PATH=%PATH%;C:\Program Files\GRASS GIS X.X\Python39\Scripts\```<br>
      ```jupyter notebook``` or ```python -m notebook```<br>

On ***Mac***:<br>
1. Follow the same Step 1 as the Linux installation, but with an additional sub-step:<br>
g. To make the conda environment to be available in Jupyter notebooks (e.g. from Anaconda), you need add the ipykernel to your conda environment:<br>
      ```conda install ipykernel```<br>
      ```ipython kernel install --user --name=kalpana```<br>
2. Follow the same Step 2 as the Linux installation.

<br><br><hr><br><br>

**The remainder of these installation instructions are out-of-date.** We created a Docker image for an earlier version of *Kalpana*, but we have not updated the image for the latest version of *Kalpana*.<br><br>

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
