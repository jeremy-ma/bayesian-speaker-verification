
conda update conda
conda create -n runenv python=2.7 anaconda
source activate runenv
python setup.py install
conda install -n runenv -c conda-forge bob.bio.gmm=2.0.7
conda install -n runenv -c conda-forge bob.bio.spear=2.0.5
