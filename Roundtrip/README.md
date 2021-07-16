# Roundtrip

Roundtrip is a deep generative neural density estimator which exploits the advantage of GANs for generating samples and estimates density by importance sampling. This repository provides a non official implementation of the model in PyTorch.

## Table of Contents

- [Install](#install)
- [Reproduction](#reproduction)
- [Citation](#citation)

## Installation

Clone this repo.
```bash
git git@github.com:FlorentinCDX/Roundtrip.git
cd Roundtrip/
```

This code requires PyTorch 1.0 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```

## Reproduction

This section provides instructions on how to reproduce results in the original paper.

### Simulation data

This repository proposes to test Roundtrip model on three types of simulation datasets. (1) Indepedent Gaussian mixture. (2) 8-octagon Gaussian mixture. (3) Involute.

The main python script `train_roundtrip.py` is used for implementing Roundtrip. Model architecture for Roundtrip can be find in `model.py`. Data  sampler can be find in `sampler.py`.

Taking the (1) for an example, one can run the following commond to train a Roundtrip model with indepedent Gaussian mixture data.

```shell
python main_density_est.py  --dx 2 --dy 2 --data indep_gmm --epochs 100 --cv_epoch 30 --cuda 1
[dx]  --  dimension of latent space
[dy]  --  dimension of observation space
[data]  --  dataset name
[epochs] -- maximum training epoches
[cv_epoch] -- epoch where checkpoint save begins
[cuda]  --  train model on GPU
```
After training the model, you will get the four sub-models (two generators and two discriminators) saved in the `model_saved/` folder.

Next, we want to visulize ability of the model to transform data from the latent space to the data space (and vice versa). One can then run the following script. 

 ```shell
python vizualise.py --data indep_gmm --dx 2 --dy 2
 [dx]  --  dimension of latent space
 [dy]  --  dimension of observation space
 [data]  --  dataset name
 ```
You will optain three plots representing sampling from latent space to data space, sampling from data space to latent space and a roundtrip (successive double pass through the generators). These figures will be placed in the `images/` folder.

Finally, in order to visulize the estimated density on a 2D region. One can then run the following script.

```shell
python evaluate.py --data indep_gmm --N 40000
 [data]  --  dataset name
 [N]  --  Number of samples used for Importance Sampling
```
 It also easy to implement Roundtrip with other two simulation datasets by changing the `data`.

- 8-octagon Gaussian mixture
    Model training:
    ```shell
    python main_density_est.py  --dx 2 --dy 2 --data eight_octagon_gmm --epochs 300 --cv_epoch 200
    ```
    Density esitmation on a 2D grid region:
    ```shell
    python evaluate.py --data eight_octagon_gmm --N 40000
    ```
- involute
    Model training:
    ```shell
    python main_density_est.py  --dx 2 --dy 2 --data involute --epochs 300 --cv_epoch 200 
    ```
    Density esitmation on a 2D grid region:
    ```shell
    python evaluate.py  --data involute --N 40000
    ```
## Citation

Once again, this is an unofficial implementation, you can find the official implementation (in python 2 and TensorFlow 1) and the link to the paper below:

![Officiel implementation](https://github.com/kimmo1019/Roundtrip)

![Paper](https://www.pnas.org/content/118/15/e2101344118#sec-11)

