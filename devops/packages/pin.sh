#!/bin/bash

SOURCE=${BASH_SOURCE[0]}
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

echo "$DIR"


#CACHE_DIR=~/conda_cache/tmp
unamestr=$(uname)
echo Operating system is  "$unamestr"
if [[ "$unamestr" == "Linux" ]]; then
   PIN_FILE=$DIR/../../etc/requirements/conda_pinned_requirements_linux
elif [[ "$unamestr" == "Darwin" ]]; then
   PIN_FILE=$DIR/../../etc/requirements/conda_pinned_requirements_osx
else
  raise error "Operating system $unamestr is not supported."
fi

#PIN_FILE=$DIR/../../etc/requirements/conda_pinned_requirements
CONDA_ENV=pin_$(git rev-parse HEAD)

#CURRENT_ENV=$CONDA_DEFAULT_ENV
# ensure conda and mamba are installed
command -v mamba
command -v conda

# create an environment to solve in
conda create -y -n "$CONDA_ENV"
mamba install -y -n "$CONDA_ENV" --file $DIR/../../etc/requirements/conda_requirements.in -c conda-forge -c bioconda -c r -c pytorch --override-channels

# run the pin command
conda list -n "$CONDA_ENV" --no-pip --explicit > "$PIN_FILE"

# finally, remove the environment
conda remove --all -y -n "$CONDA_ENV"
