# openfisca_married_couples
Repo to store codes using the microsimulation tool OpenFisca in order to simulate a separate tax for married couples

# First Installation

First, install Docker Desktop on your computer (https://www.docker.com/products/docker-desktop/)

Launch Docker Desktop. Docker enables you to get a good reproductibility of codes, a bit like a virtual machine.

Go to a folder in your computer, open a terminal, clone this repository and go into it :
```sh
git clone https://github.com/pvanborre/openfisca_married_couples.git
cd openfisca_married_couples
```


Then construct the Docker Image (it indicates all versions that are needed to run codes thanks to the Dockerfile)
```sh
docker build --tag public-openfisca-image .
# the . at the end is important... (it indicates that the Dockerfile is at this location)
```



# Usage 

Launch Docker Desktop. Docker Desktop needs to be launched every time you want to run a code, otherwise you will get the following error : 'error during connect: this error may indicate that the docker daemon is not running'

Then launch your container from the Docker image:

```sh
# replace my 3 paths by where_you_cloned/openfisca_married_couples/codes and where_you_cloned/openfisca_married_couples/data and where_you_cloned/openfisca_married_couples/outputs
docker run -it --name openfisca-container -v C:/Users/pvanb/Projects/openfisca_married_couples/codes:/app/codes -v C:/Users/pvanb/Projects/openfisca_married_couples/data:/app/data -v C:/Users/pvanb/Projects/openfisca_married_couples/outputs:/app/outputs public-openfisca-image /bin/bash
# The -v flags indicates that your codes and data on your disk will be accessible inside the container, in the folders app/codes and app/data
```

Now you should be inside your container in your terminal, in the app/codes folder.
To check that your installation worked, please run : 
```sh
python test_without_data.py
# you should get the following results : irpp [-756.]
```


You can now launch python codes running on data provided you have the .h5 data (see below)
```sh
python without_reform.py
python reform_towards_individualization.py
```


If you want to edit the codes, you can just open any code editor you like on your disk and inside the container it will be able to recognize modifications.

When you are finished running codes, write 
```sh
exit
# to exit the container
``` 
and then 
```sh
docker rm openfisca-container
``` 

# The tough point : get your .h5 data 

In the folder data/year you need to place the flat_year.h5 dataset. This dataset comes from the ERFS-FPR and were transformed to be consumable by OpenFisca.

How to get this .h5 file ?
TODO 

