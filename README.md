# Scaling Neural Face Synthesis to High FPS and Low Latency by Neural Caching (WIP)
### [Paper](https://arxiv.org/abs/2211.05773) | [Website](https://yu-frank.github.io/lowlatency/)
>**Scaling Neural Face Synthesis to High FPS and Low Latency by Neural Caching**\
>[Frank Yu](https://yu-frank.github.io/), [Sidney Fels](https://ece.ubc.ca/sid-fels/), and [Helge Rhodin](https://www.cs.ubc.ca/~rhodin/web/)\
>WACV 2023

## Abstract
Recent neural rendering approaches greatly improve image quality, reaching near photorealism. However, the underlying neural networks have high runtime, precluding telepresence and virtual reality applications that require high resolution at low latency. The sequential dependency of layers in deep networks makes their optimization difficult. We break this dependency by caching information from the previous frame to speed up the processing of the current one with an implicit warp. The warping with a shallow network reduces latency and the caching operations can further be parallelized to improve the frame rate. In contrast to existing temporal neural networks, ours is tailored for the task of rendering novel views of faces by conditioning on the change of the underlying surface mesh. We test the approach on view-dependent rendering of 3D portrait avatars, as needed for telepresence, on established benchmark sequences. Warping reduces latency by 70% (from 49.4ms to 14.9ms on commodity GPUs) and scales frame rates accordingly over multiple GPUs while reducing image quality by only 1%, making it suitable as part of end-to-end view-dependent 3D teleconferencing applications.

## Installation Instructions
```
conda create -n snfs python=3.8
conda activate danbo

# install pytorch for your corresponding CUDA environments
conda install torch

# install other dependencies
pip install -r requirements.txt
```

## Training Modles
Using default config.py file will train the full caching and warping models jointly or the baseline model
### Training Caching/Warping Models
```
python train.py --data path_to_dataset
```
### Training Baseline Model
```
python train_baseline.py --data path_to_dataset
```

Weights for the VGG16 network (perceptual loss) can be found here: 'https://download.pytorch.org/models/vgg16-397923af.pth'. Place the downloaded model in 
```
model/vgg16-397923af.pth
```

## Multithreading Script
An example of our multithread/multi-GPU code can be run:
```
python multithread_inference.py 
```
Please contact frankyu@cs.ubc.ca for a pretrained model

## Obtaining Data
Please contact frankyu@cs.ubc.ca for data

## WIP
* Adding more documentation to functions/models and config.py file
* Adding sample data for training/benchmarking 

## Citation
```
@InProceedings{yu2023scaling,
    author    = {Yu, Frank and Fels, Sid and Rhodin, Helge},
    title     = {Scaling Neural Face Synthesis to High FPS and Low Latency by Neural Caching},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year      = {2023},
}
```

## Acknowledgements
This code is built upon:
* https://github.com/SSRSGJYD/NeuralTexture (Deferred Neural Rendering)
* https://github.com/yfeng95/DECA 
