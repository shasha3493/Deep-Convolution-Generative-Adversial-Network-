# Deep Convolutional Genrative Adversial Network

An implemetation of the paper [Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)

# DataSet

In this tutorial we will use the [Celeb-A Faces dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) which can be downloaded at the linked site, or in [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg). The dataset will download as a file named img_align_celeba.zip. Once downloaded, create a directory named celeba and extract the zip file into that directory. Then, set the dataroot input for this notebook to the celeba directory you just created. The resulting directory structure should be:

    -> celeba
      -> img_align_celeba
          -> 188242.jpg
          -> 173822.jpg
          -> 284702.jpg
          -> 537394.jpg
           ...

# Getting Started

- Execute the following commands in this folder to set up the require virtual environment for running these experiments.

    python3 -m venv proj_env
  
    source proj_env/bin/activate
  
    pip install -r requirements.txt



- To train the network execute the following command
  
  python3 train_dcgan.py

# Results with Visualization Notebook

https://github.com/shasha3493/Deep-Convolution-Generative-Adversial-Network-DGGAN-/blob/master/test_dcgan.ipynb

# Built With
- [PyTorch](https://pytorch.org/) 
