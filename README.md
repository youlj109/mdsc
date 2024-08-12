# Online Unsupervised Feature Selection on Data Streams via Manifold and Discriminative Structure Consistency
__This repo is officical PyTorch implement of 'Online Unsupervised Feature Selection on Data Streams via Manifold and Discriminative Structure Consistency'  by Linjing You,Yaming Huang,Jiabao Lu,Yaozu Liu,Zhiyi Yang,Xiayuan Huang.__  
## Dependence
We use `python==3.8.13`, other packages including:
```
torch==1.12.0+cu113
numpy==1.24.4
pandas==2.0.3
tqdm==4.66.2
timm==0.9.16
pillow==10.3.0
```
We also share our python environment that contains all required python packages. Please refer to the `./MDSC.yml` file.  
You can import our environment using conda:
```
conda env create -f MDSC.yml -n MDSC
```
## Dataset
Download datasets used in our paper from:  
[COIL-20](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)  
[USPS](https://paperswithcode.com/dataset/usps)  
[EMNIST](https://yann.lecun.com/exdb/mnist/)  
[ISOLET](https://archive.ics.uci.edu/dataset/54/isolet)  
[DrivFace](https://archive.ics.uci.edu/dataset/378/drivface)  
[Acoustic](https://www.archive.ics.uci.edu/dataset/406/anuran+calls+mfccs)  
## Feature Selection
Please use `MDSC.py` to choose a subset of features. For example:
```
data = scipy.io.loadmat('example.mat')
```
Replace `example.mat` with your data.
```
cd code/
python train.py --m 10 \
                --nanpta1 1 \
                --nanpta2 1 \
                --nanpta3 1 \
                --nanpta4 1 \
                --lr 0.2 \
                --num_epochs 100  \
                --M 10  \
                --num_batches 10  \
```
Set the hyperparameters here.
## Tested Environment
We tested our code in the environment described below.
```
OS: Ubuntu 18.04.6 LTS
GPU: NVIDIA GeForce RTX 4090
GPU Driver Version: 535.129.03
CUDA Version: 12.2
```
