Instructions for set-up the python environment to work on Windows
1) Install miniconda.
2) Open a cmd prompt and navigate to the install folder on the github repo.
3) Open the yml file  with a text editor and modify last line (prefix) with the path of your miniconda folder.
4) Execute the command: conda env create -f kalpana.yml
5) Install the 4 wheel files available on the install folder. Follow the next order: gdal, fiona, shapely and rtree. For installing, execute: pip install name_of_the_wheel_file (use tab for auto-complete).
6) Install geopandas using pip: pip install geopandas

Instructions for set-up the python environment to work on Ubuntu 20.04
1) Install virtual env: apt-get install python3-venv
2) Create a python virtual environment: python3 -m venv my_env_project
3) Activate the environment: source/my_env_project/bin/activate
4) pip install -r ubuntuReqs.txt

Instructions for set-up the environment in Henry2
1) Add python 3 to the PATH: module load conda
2) Create an environment using kalpana_env_H2.yml: conda env create -f kalpana_env_H2.yml -p /path/to/the/environment
* The environment can be to big to be instaled in the HOME directory, so specifying the path all packages will be instaled in a diferent place.
