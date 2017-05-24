# DCGAN for CIFAR10
A clean implementation of DCGAN for CIFAR 10 from [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) in tensorflow 1.1

# TRAINING
You have to download cifar10 into data folder 
```python
./download.sh
```
To train model 
```python
python cifar_gan.py
```
# EXAMPLE 
at 10000 iter

You can see it begin with noise image at first iteration and will generate image more clearly but in the end, the model collapse happen, images are destroyed
 
![](./out/cifar_gan.gif)
