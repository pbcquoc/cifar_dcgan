# DCGAN for CIFAR10

![](./out/DCGAN.png)

A clean implementation of DCGAN for CIFAR 10 from [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) in tensorflow 1.1

## PREREQUISITES
- Python 2.7
- Tensorflow 1.1
- numpy

## TRAINING
You have to download cifar10 into data folder 
```python
./download.sh
```
To train model 
```python
python cifar_gan.py
```

You can find generated images at out folder

## RESULTS
at 5 iters

![](./out/0005.png)

at 85 iters

![](./out/0085.png)

at 185 iters

![](./out/0185.png)

at 285 iters

![](./out/0285.png)

at 385 iters

![](./out/0385.png)

You can see it begin with noise image at first iteration and will generate image more clearly but in the end, the model collapse happen, images are destroyed
 
![](./out/cifar_gan.gif)
