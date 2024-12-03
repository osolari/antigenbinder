set -x -e

mkdir ~/.pip
{
  echo "[global]"
  echo "index-url = $PIP_INDEX_URL"
  echo "timeout = 120"
} >> ~/.pip/pip.conf