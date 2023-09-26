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