## Kalpana installation

The first step is to clone the Kalpana repository to your local machine. You can find the instructions [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository). <br>

The following steps depend on your OS and what features of Kalpana you want to use.<br>

On ***Linux***:<br><br>

1. Create a conda environment. See [the miniconda website](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Kalpana has been tested with Python 3.11. You can use mamba instead if you prefer.<br>
   ```conda create --name kalpana python=3.10```<br>
2. Install ***Kalpana*** dependencies using pip. Navigate to the Kalpana GitHub repository.<br>
  ```pip install -e .```<br>
3. (Not necessary) To run the downscaling tools, you must install GRASS GIS (ttps://grass.osgeo.org/). Versions >= 8.2 are supported.<br><br>

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
```set PATH=%PATH%;C:\Users\tacuevas\AppData\Roaming\Python\Python39\Scripts\```<br>
```jupyter-notebook```
