#!/bin/bash
set -e

# find the directory of this script
SOURCE=${BASH_SOURCE[0]}
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

echo $DIR
PYTEST_FLAGS="--run-extra-slow"

echo "pytest_flags=" $PYTEST_FLAGS
echo "ignore nbs=" $IGNORE_NBS

cd $DIR
cd ../..

black . --check --line-length 120

nice -n 50 pytest --tb=long \
     --junitxml=test-reports/results.xml \
     $1