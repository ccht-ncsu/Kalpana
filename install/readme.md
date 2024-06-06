To install ***Kalpana*** on Linux follow the steps below:

1. Clone the Kalpana repository to your local device. You can find the instructions [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).
2. Create a conda environment. See [the miniconda website](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Kalpana has been tested with Python 3.11. You can use mamba instead if you like.<br>
   ```conda create --name kalpana python=3.11```
3. Install ***Kalpana*** dependencies using pip. Navigate to the Kalpana GitHub repository.<br>
  ```pip install -e .```

For using the visualization functions or to export the NetCDF as GIS files, you are done! To use the downscaling feature (f, you need to have GRASS GIS installed. Versions >= 8.2 are supported ((ttps://grass.osgeo.org/). On Linux, you need to install ***Kalpana*** and GRASS GIS, but for Windows, it is more complicated.

For Windows, miniconda is ignored, and you need to use the Python installation that comes with GRASS GIS. It's worth noting that you can not have more Python installations on your system. To use the grass Python.exe, you need to launch GRASS GIS and then use the GRASS cmd. In this case, you need to install all the necessary dependencies using pip. With the GRASS cmd navigate to Kalpana GitHub repository and install Kalpana. <br>
  ```python -m pip install -e .```

If you want to use Jupyter notebooks, you need to launch GRASS GIS and close the GUI without closing the terminal. Then you need to do the following:

set PATH=%PATH%;C:\Program Files\GRASS GIS X.X\
set PATH=%PATH%;C:\Users\tacuevas\AppData\Roaming\Python\Python39\Scripts\
