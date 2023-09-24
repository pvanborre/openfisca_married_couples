# openfisca_married_couples
Repo to store codes using the microsimulation tool OpenFisca in order to simulate a separate tax for married couples

# First Installation 

## On Windows

First, install a WSL : open a terminal in adminsitrator mode and run.
```sh
wsl --install
```
Then restart your computer


Second, install Docker Desktop on your computer (follow the instructions on this link : https://docs.docker.com/desktop/install/windows-install/)

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

## On Linux (not tested)

Install Docker Desktop on your computer (follow the instructions on this link : https://docs.docker.com/desktop/install/linux-install/. If you are on Ubuntu for instance follow the instructions here https://docs.docker.com/desktop/install/ubuntu/)

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

# On MacOSX 

This is not recommended since Docker works poorly on Mac...

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

Check you have git on your computer by opening a terminal and running
```sh
git --version
``` 

Then go to a folder of your choice and clone the openfisca-france-data repository

```sh
git clone https://github.com/openfisca/openfisca-france-data.git
cd openfisca-france-data
``` 

Then build your docker image 
```sh
docker build -t public-openfisca-france-data . -f ./docker/Dockerfile
``` 

From now on you have 2 docker images (that you can see in Docker Desktop) :

+ one is called public-openfisca-image and containers from this image are designed to run python codes to do simulations, reforms etc.

+ this other is called public-openfisca-france-data and is useful to transform your ERFS-FPR files in SAS to data consumable by OpenFisca (.h5 files)

Now the steps to follow are : 

+ Download 4 files per year on the 'Reseau Quetelet' (https://commande.progedo.fr/fr/utilisateur/connexion) : you should now have files named fpr_menage_YEAR.sas7bdat, fpr_indiv_YEAR.sas7bdat, fpr_mrfARe02.sas7bdat, fpr_mrfARe02.sas7bdat (where AR are the 2 last digits of the year). You may also prefer to get your files on CASD.


+ Place these 4 files in _openfisca-france-data/docker/data/data-in_ folder.

+ Place these 4 files also in _openfisca-france-data/docker/data/data-in/erfs-fpr/donnees\_sas\_year_ folder. (you should create the folder _donnees\_sas\_year_ for that)

+ Then edit _openfisca-france-data/docker/data/raw_data.ini_ to specify the year.
It should look for instance like this 
```ini
[erfs_fpr]

2018 = ./data-in/erfs-fpr/donnees_sas_2018
```

Now you are able to run your container from your image
```sh
# replace the first link by where you cloned openfisca-france-data 
# C:/Users/where_you_cloned_openfisca-france-data/openfisca-france-data/docker/data

# replace the second link by where you cloned openfisca_married_couples 
# C:/Users/where_you_cloned_openfisca_married_couples/openfisca_married_couples/mon_input_data_builder

docker run -it --name openfisca-container2 -v C:/Users/pvanb/Projects/my_openfisca/gestion_donnees_erfs/openfisca-france-data/docker/data:/mnt -v C:/Users/pvanb/Projects/openfisca_married_couples/mon_input_data_builder:/mnt/mon_input_data_builder public-openfisca-france-data /bin/bash
``` 

You are now in your container and you should be in the mnt/ folder. Run the following script (copy and paste it in your terminal/container): this takes between 5 and 10 minutes
```sh
if [[ -z "${DATA_FOLDER}" ]]; then
  export DATA_FOLDER=/mnt
fi
cd $DATA_FOLDER
# Cleaning
echo "Cleaning directories..."
rm $DATA_FOLDER/data-out/tmp/*
rm $DATA_FOLDER/data-out/*.h5
rm $DATA_FOLDER/data-out/data_collections/*.json
# Clean config file
sed -i '/erfs_fpr = /d' $DATA_FOLDER/config.ini
sed -i '/openfisca_erfs_fpr = /d' $DATA_FOLDER/config.ini
echo "Building collection in `pwd`..."
build-collection -c erfs_fpr -d -m -v  -p $DATA_FOLDER 2>&1
if [ $? -eq 0 ]; then
    echo "Building collection finished."
else
    echo "ERROR in build-collection"
    echo "Content of $DATA_FOLDER : "
    ls $DATA_FOLDER
    echo "Content of $DATA_FOLDER/data-in/: "
    ls $DATA_FOLDER/data-in/
    echo "Content of $DATA_FOLDER/data-out/ : "
    ls $DATA_FOLDER/data-out/
    echo "Content of $DATA_FOLDER/data-out/tmp/ : "
    ls $DATA_FOLDER/data-out/tmp/
    echo "---------------- DONE WITH ERROR -----------------------------"
    exit 1
fi
``` 

Now run this command for years 2019 and 2018
```sh
# replace flat_YEAR.h5 by the YEAR you are interested in (2018 or 2019)
python /opt/openfisca-france-data/openfisca_france_data/erfs_fpr/input_data_builder/__init__.py  --configfile ~/.config/openfisca-survey-manager/raw_data.ini --file ./data-out/flat_YEAR.h5 2>&1
``` 
 
For years 201X - 2017 run
```sh
python /mnt/mon_input_data_builder/new_erfs/__init__.py  --configfile ~/.config/openfisca-survey-manager/raw_data.ini --file ./data-out/flat_YEAR.h5 2>&1
``` 

For even older years run
```sh
python /mnt/mon_input_data_builder/old_erfs/__init__.py  --configfile ~/.config/openfisca-survey-manager/raw_data.ini --file ./data-out/flat_YEAR.h5 2>&1
``` 