# Background Subtraction with Two-Stage Network of Density Estimation and Difference Approximation

This repository stores the implementation of Background subtraction using Convolutional Density Network.

Requirements:
- For Python:
	- Python 3.x (well-tested with Python 3.6)
	- Tensorflow >= 2.1.0
	- PyTorch >= 1.7.1
	- NumPy >= 1.18.1
	- CuPy >= 7.4.0
	- Chainer >= 7.4.0
	- OpenCV >= 4.2.0
	**Notice**: This project uses the minimum required version of the libraries specified above. If you want to use newer Chainer version, please note that different Chainer versions may required different CuPy versions. Also, CUDA 10.1 with cuDNN 7.6.5 is used here.
- For C++:
	- g++ >= 7.x (well-tested with g++ 7.5.0)
	- LibTorch >= 1.7.1
	- UPC++ >= 2021.3.0
	**Notice**: This project uses the minimum required version of the libraries specified above. Also, CUDA 10.1 with cuDNN 7.6.5 is used here. Because LibTorch (PyTorch C++ API) and UPC++ are unpopularly-used packages. The warm-up guides for installation and compilation can be reached at here:
		- OpenCV on C++: [compilation guide](https://github.com/phithangcung/Installation-Notes/blob/main/Install_OpenCV4_with_CUDA.md)
		- LibTorch (PyTorch C++ API): [compilation guide](https://github.com/phithangcung/LibTorch-UPCXX-OpenCV), [compilation with `xeus-cling`](https://github.com/phithangcung/Installation-Notes/blob/main/Install-Jupyter-LibTorch.md).
		- UPC++: [installation](https://bitbucket.org/berkeleylab/upcxx/wiki/INSTALL), [Programmer's Guide](https://bitbucket.org/berkeleylab/upcxx/downloads/upcxx-guide-2020.10.0.pdf)



# Executing Instruction
## For PyTorch branch:

We need to link the data from TensorFlow directory to PyTorch directory by executing the file `download_data.sh` in the directory `torch-models/data`

### A. Setting hyperparameters

In the `config` directory and in the main directory contain a file called `config.json` which is used to set the hyperparameters used by the program. The meaning of those parameters are as follow:

- `FPS`: the number of frames in temporal dimension to consider when generating the background.
- `KMIXTURES`: the number of Gaussian components used to represent the temporal data of a pixel.
- `PIXEL_BATCH`: this number is similar to **batch_size** parameter when training or evaluating the model, i.e. during inference, the model will take a tensor of `PIXEL_BATCH` pixels to give the corresponding output for those pixels.

### B. Training
#### 1. Training Convolution Density Network for Background Modelling
Because our framework contains a duo of networks, we need to perform training on the model of background subtraction before training the model of foreground detection. To train the network on a video, first set the desire parameters in the file `config.json` then run the file `train_cdn.py` in the `torch-models` folder.

The `train_cdn.py` file contains some notations:
1. We aim to train the model with scene-specific pipelines
2. We configure to train all video sequences, except the scenario of PTZ

**NOTE**: During training process, the pretrained weights will be saved in the directory `training_ckpt/<scenario-name>/<sequence-name>` with the name in the form `cdn.pth`. The training is quick to approach to the convergence of the statistic learning. We experimented with:
```
FPS = 240,
KMIXTURE = 4,
PIXEL_BATCH = 128,
LEARNING_RATE = 1e-4,
EPOCHS = 2,
SLIDING_STEP = 80,
WINDOW_STRIDE = 2
```

#### 2. Evaluating

To run the algorithm on videos, Run `eval_cdn.py`. 

**Notice:** We need to pretrain video sequences before evaluating models.


### Important notes

- The video frame should be normalized into [0, 1] before training and evaluating (by dividing with 255).
- All computations of this method is done on normalized tensors.
- The output of the network is the mixture weights, the normalized means, and the normalized variances. All of these parameters' values are in the range [0, 1].
- The variances is constraint to take only values between [16, 32] (`variance = (16 + 16*normalized_variance) / 255` - this formula will give the normalized variance in range [0, 255], the `normalized_variance` variable is the output of the network which is a tensor of the normalized values in range [16, 32]).


### C. Generating the data for foreground training

Before training the model of foreground detection, we need to generate batches of pairs `{[input_image + modelled_background], labelled_foreground}`. The idea of this step is to generate randomly 200 sampling pairs and.  To generate the data, we just run `generate_fg_data.py`. This pipeline aims to process in two kinds of data:
- Compressed data in `h5py` format to train model on PyTorch,  stored this in `torch-models/data/fg_train_data`
- Raw data of image frames to train model on PyTorch C++ API,  stored this in `torch-models/data/fg_train_frame`

### D. Train the model of foreground detection
We need to initialize the training configuration in `config.json` file. Then we run the file `fg_trainer.py` in the directory `torch-models/train`. Because we focus on training the model on PyTorch with distributed paradigm, we created a distributed pipeline on C++ with LibTorch and UPC++.

## For LibTorch branch:
### A. Initializing the training data
The implementation of LibTorch branch is provided in directory `torch-models/libtorch`. We aim to implement distributed worker that perform asynchronuous comminication following the pipeline of EventGraD. In the case that you do not want to pretrain the model of background modelling, you can download the pre-generated data via download our data with:
```
$ apt install curl

$ fileid="1ocUVL57Fm51IW-HyLfUzzQdNGGfzFS2A"
$ filename="fg_train_frame.zip"
$ curl -c ./cookie -s -L \
	"https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
$ curl -Lb ./cookie \
	"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" \
	-o ${filename}

$ fileid="1uOfvbr4WIjiedzdIriKK5-SEda2OagLj"
$ filename="fg200.zip"
$ curl -c ./cookie -s -L \
	"https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
$ curl -Lb ./cookie \
	"https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" \
	-o ${filename}
```
or you can run the pre-configure setting by running `./download_data.sh` in the directory `data`

After retrieve the data to train the model of foreground detection, you need to link with the local branch of LibTorch via running `./link_data.sh` in the directory `torch-models/libtorch/data` to form a soft link of data from PyTorch to LibTorch implementation. 

### B. Training model
In this section, we suppose that the current directory is `torch-models/libtorch`. Because our implementation exploits UPC++, OpenCV and LibTorch, we need to compile from source with `cmake`.  To compile the model, first, we need to create a folder for `cmake build`.
```
$ mkdir build
$ cd build
```
 Then, we execute as follow:
```
$ cmake -DCMAKE_PREFIX_PATH="<path-to-libtorch>" \
    -DCMAKE_CXX_COMPILER=upcxx \
    -DCMAKE_CXX_FLAGS="-O " .. 
$ cmake --build . --config Debug
$ cp example-app ../
``` 
where `<path-to-libtorch>` is the absolute path to your libtorch as configuring in the setting part. If you miss them, you can review our guide at [compilation guide](https://github.com/phithangcung/LibTorch-UPCXX-OpenCV). The compilation will create an executable file, named `example-app` in the directory `torch-models/libtorch`.

In order to perform the model training, we execute:

```
$ upcxx-run -n <num_process> example-app \
    <threshold_type> \
    <threshold_value> \
    <batch_size> \
    <num_epochs> \
    <logging_mode> \
    <logging_frequency>
```
where :
- `threshold_type` is either 0 or 1, where 0 is the mode of using constant threshold, 1 is for adaptive threshold.
- `threshold_value` is the value of the threshold, which was configure with respect to the norm values of model's parameters
- `batch_size` is the batch size of training procedure
- `num_epochs` is the number of epoch that we use to train the model
- `logging_mode` is either 0 or 1, where 0 is to disable showing log history of training procedure, and 0 is to enable showing log history of training procedure.
- `logging_frequency` is to describe how frequently the logging is displayed. This value is counted on the training steps.

*Note*: Some pre-configured implementation to bench mark the method is presented in the directory of run. We just need to override the default run file and recompile from source to execute the model properly.














