#!/bin/bash

conda env create -f environment.yml
conda activate FireHydrant

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
cat <<EOF > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
#!/bin/sh
export FH_BASE=$PWD
export PYTHONPATH_FHOLD=\$PYTHONPATH
export PYTHONPATH=\$FH_BASE:\$PYTHONPATH
EOF

cat <<EOF > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
#!/bin/sh
export PYTHONPATH=\$PYTHONPATH_FHOLD
unset PYTHONPATH_FHOLD
unset FH_BASE
EOF
