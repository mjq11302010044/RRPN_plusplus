## Installation

## Environments
- Ubuntu 16.04
- Cuda 10.0 or 9.0
- python >=3.5
- **pytorch 1.7 or higher**
- Other packages like numpy, cv2.




```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name rrpn_pytorch
source activate rrpn_pytorch

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
conda install pytorch

# install torchvision
cd ~/github
git clone https://github.com/pytorch/vision.git
cd vision
python setup.py install

# install pycocotools
cd ~/github
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd ~/github
git clone https://github.com/mjq11302010044/RRPN_plusplus.git
cd RRPN_plusplus
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

#-------
python rotation_setup.py install

# If you use Python 3.x to compile, the lib folder will be like 'lib.xxxxx'
mv build/lib/rotation/*.so ./rotation
#-------
