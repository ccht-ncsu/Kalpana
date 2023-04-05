## Instractions on how to build Kalpana docker image and container

### Build Docker Image

To build the Kalpana docker image run the following command from the build directory:

./build.sh version

where version is a version number, such as v0.0.1 or a names such as latest.

### Create Docker Container

To create the Kalpana docker container, first edit the createcontainer.sh file changing the line:

--volume /xxxx/xxxxx/xxxx:/data

by replacing the xxx's to a directory path on your computer where you plan to write data too. Then
run the follwing command:

./createcontainer.sh version

where version is a version number, such as v0.0.1 or a names such as latest.

### Access Docker Container

To access the docker container's shell run the following command:

./shell.sh version

where version is a version number, such as v0.0.1 or a names such as latest.
