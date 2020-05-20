#!/usr/bin/env bash

set -e

if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    pushd .
    cd
    mkdir -p download
    cd download
    echo "Cached in $HOME/download :"
    ls -l
    echo
    if [[ ! -f miniconda.sh ]]; then
      wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
    fi
    chmod +x miniconda.sh && bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    cd ..

    conda update --yes conda
    popd

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION numpy scipy coverage

    source activate testenv

    pip install scikit-learn
    pip install pandas
    pip install coveralls
    pip install pytest

    python setup.py develop;
fi

