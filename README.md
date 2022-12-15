# Project_DS_IceCube_RNN
Methods for RNN-based detection of neutrino events and rejection of noise background signals for the IceCube collaboration

Instructions
==========

Setting up the solution
==================
The solution consists of one runnable notebook (.ipynb) and seven class definition files (.py). In order to execure the notebook, the following preparations must be made:

1. Install the required Python libraries (using requirements.txt)
2. Make sure all the class definition files are in the same directory as the notebook file
3. Download the required datasets (as explained below), and update the corresponding path definitions in the notebook so they point to the directory with the datasets
4. Run the notebook from top and down
** NOTE: Different parts of the initial section of the notebook should be run depending on whether the environment is Google Colab or a local environment **

Datasets
=======
** NOTE: Due to size limitations, only the first dataset is available in the Github repo. Please contact us if you need access to the other two sets. **

For training and testing univariate processing (both batch and continuous), the first dataset suffices. The second dataset is only required if you want to do additional tests on univariate models. The third set is required for multivariate processing.

1: datasets/ARIANNA_100_1CH: 7 npy files with simulated data from one channel (600 000 noise events and 100 000 signal events, pruned to size 100)
2: datasets/ARIANNA_256_1CH: 11 npy files with simulated data from one channel
3: datasets/ARIANNA_256_5CH: 13 npy files with simulated data from one channel (900 000 noise events and 120 000 signal events)
