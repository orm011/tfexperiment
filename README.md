## CLONE. took the miniplaces folder code from

https://github.com/hangzhaomit/tensorflow-tutorial
ran the 2to3 python tool to make it mostly work with python3

then fixed a couple of divisions that meant integer division in python2
but mean floating point in python3.

After doing that, I tested it by running alexnet_train on a small dataset.

## useful link on getting GPU tensforflow working with anaconda:
When using conda install tensforflow, one gets a version that does not support gpus even
if one has them.

but one can keep conda following the instructions here:
https://devtalk.nvidia.com/default/topic/936429/-solved-tensorflow-with-gpu-in-anaconda-env-ubuntu-16-04-cuda-7-5-cudnn-/
with the link for our version here:
https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#using-pip

One also needs to get CuDNN (from nvidia, need to register for that, takes a couple of days)
and Cuda 8 installed (this one is easy to download)

## Sample code for miniplaces challenge

To get started:

- Modify data paths accordingly in prepro_data.py
- Preprocess data into .h5

        python prepro_data.py
      
- Run AlexNet training script

        python alexnet_train.py
