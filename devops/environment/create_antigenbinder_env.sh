set -x -e

SOURCE=${BASH_SOURCE[0]}
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

unamestr=$(uname)
echo Creating environment for "$unamestr" operating system
if [[ "$unamestr" == "Linux" ]]; then
   PIN_FILE=$DIR/../../etc/requirements/conda_pinned_requirements_linux
elif [[ "$unamestr" == "Darwin" ]]; then
   PIN_FILE=$DIR/../../etc/requirements/conda_pinned_requirements_osx
else
  raise error "Operating system $unamestr is not supported."
fi

conda install --freeze-installed --file "$PIN_FILE" -c r -c anaconda -c conda-forge -c bioconda -c defaults -c pytorch --override-channels
python -m pip install -r $DIR/../../etc/requirements/pip_requirements.in --upgrade --force-reinstall --no-dependencies
python -m pip install --editable .