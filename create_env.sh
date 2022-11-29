conda env create -f environment.yml && conda activate stega-nerf
# install customized cuda kernels
pip install . --upgrade --use-feature=in-tree-build
