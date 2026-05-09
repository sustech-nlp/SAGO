# Environment setup
conda create -n sago python=3.11 -y
conda activate sago
# Install specific torch version first to avoid conflicts
pip install torch==2.4.1
pip install .[lm_eval]
pip install --no-build-isolation flash-attn==2.6.3

# Data setup
python setup_data.py --wmdp
python setup_data.py --rwku_batch

# Additional datasets are supported — run below for options:
# python setup_data.py --help
