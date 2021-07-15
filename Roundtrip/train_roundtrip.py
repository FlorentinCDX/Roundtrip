import argparse
import model
import torch
import torch.optim as optim
import numpy as np
import math
import sys, os 
import argparse
import time
import sampler

class RoundtripModel(object):
    """
    Implementation of a Roundtrip model from the Paper : 'Density estimation using deep generative neural networks' by
    Qiao Liu, Jiaze Xu, Rui Jiang, Wing Hung Wong

    Initialisation inputs:
        - g_net (nn.Module): The first Generator model G
        - h_net (nn.Module): The second Generator model H
        - dx_net (nn.Module): The first Discriminator 
        - dy_net (nn.Module): The second Discriminator
        - data (string): Name of the data set used (for name of the saved model)
        - x_sampler : A python method implementing a sampler from the complex distribution  
        - y_sampler : A python method implementing a sampler from lattent distribution 
        - batch_size (int) : Siez of the batches for training 
    """
    def __init__(self, g_net, h_net, dx_net, dy_net, x_sampler, data, y_sampler, alpha, beta, device='cpu'):
        self.g_net = g_net.to(device)
        self.h_net = h_net.to(device)
        self.dx_net = dx_net.to(device)
        self.dy_net = dy_net.to(device)
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.device = device

        self.g_h_optim = torch.optim.Adam(list(self.g_net.parameters()) + list(self.h_net.parameters()), \
                                lr = 2e-4, betas = (0.5, 0.9))
        self.d_optim = torch.optim.Adam(list(self.dx_net.parameters()) + list(self.dy_net.parameters()), \
                                lr = 2e-4, betas = (0.5, 0.9))

    def discriminators_loss(self, x, y):
        fake_y = self.g_net(x)
        fake_x = self.h_net(y)

        dx = self.dx_net(x)
        dy = self.dy_net(y)

        d_fake_x = self.dx_net(fake_x)
        d_fake_y = self.dy_net(fake_y)
        #(1-d(x))^2
        dx_loss = (torch.mean((0.9*torch.ones_like(dx) - dx)**2) \
                        +torch.mean((0.1*torch.ones_like(d_fake_x) - d_fake_x)**2))/2.0
        dy_loss = (torch.mean((0.9*torch.ones_like(dy) - dy)**2) \
                        +torch.mean((0.1*torch.ones_like(d_fake_y) - d_fake_y)**2))/2.0
        d_loss = dx_loss + dy_loss
        return d_loss
    
    def generators_loss(self, x, y):
        y_ = self.g_net(x)
        x_ = self.h_net(y)

        x__ = self.h_net(y_)
        y__ = self.g_net(x_)

        dy_ = self.dy_net(y_)
        dx_ = self.dx_net(x_)

        l2_loss_x = torch.mean((x - x__)**2)
        l2_loss_y = torch.mean((y - y__)**2)

        #(1-d(x))^2
        g_loss_adv = torch.mean((0.9*torch.ones_like(dy_)  - dy_)**2)
        h_loss_adv = torch.mean((0.9*torch.ones_like(dx_) - dx_)**2)

        g_loss = g_loss_adv + self.alpha*l2_loss_x + self.beta*l2_loss_y
        h_loss = h_loss_adv + self.alpha*l2_loss_x + self.beta*l2_loss_y
        g_h_loss = g_loss_adv + h_loss_adv + self.alpha*l2_loss_x + self.beta*l2_loss_y
        return g_h_loss
        
    def train(self, epochs, cv_epoch):
        for epoch in range(epochs):
            start_time = time.time()

            self.g_h_optim.zero_grad()
            self.d_optim.zero_grad()

            x = torch.Tensor(self.x_sampler.sample()).to(self.device)
            y = torch.Tensor(self.y_sampler.sample()).to(self.device)

            d_loss = self.discriminators_loss(x, y)
            g_h_loss = self.generators_loss(x, y)

            d_loss.backward()
            g_h_loss.backward()
            self.g_h_optim.step()
            self.d_optim.step()

            if epoch % 100 == 0 :
                print('Epoch [%d] Time [%5.4f] g_h_loss [%.4f] d_loss [%.4f]' %
                        (epoch, time.time() - start_time, g_h_loss, d_loss))
            
            if epoch >= cv_epoch:
                self.save()
            
    def save(self):
        torch.save(self.g_net.state_dict(), 'model_saved/g_net_{}'.format(self.data))
        torch.save(self.h_net.state_dict(), 'model_saved/h_net_{}'.format(self.data))
        torch.save(self.dx_net.state_dict(), 'model_saved/dx_net_{}'.format(self.data))
        torch.save(self.dy_net.state_dict(), 'model_saved/dy_net_{}'.format(self.data))

    def load(self):
        self.g_net.load_state_dict(torch.load('model_saved/g_net_{}'.format(self.data)))
        self.h_net.load_state_dict(torch.load('model_saved/h_net_{}'.format(self.data)))
        self.dx_net.load_state_dict(torch.load('model_saved/dx_net_{}'.format(self.data)))
        self.dy_net.load_state_dict(torch.load('model_saved/dy_net_{}'.format(self.data)))
        print('Restored model weights.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='indep_gmm',help='name of data type')
    parser.add_argument('--model', type=str, default='model',help='model path')
    parser.add_argument('--dx', type=int, default=10,help='dimension of latent space')
    parser.add_argument('--dy', type=int, default=10,help='dimension of data space')
    parser.add_argument('--bs', type=int, default=1000,help='batch size for training')
    parser.add_argument('--epochs', type=int, default=2000,help='maximum training epoches')
    parser.add_argument('--cv_epoch', type=int, default=1800,help='epoch starting for evaluating')
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--cuda', type=int, default=1, help='usage of cuda GPU')
    args = parser.parse_args()
    data = args.data
    x_dim = args.dx
    y_dim = args.dy
    batch_size = args.bs
    epochs = args.epochs
    cv_epoch = args.cv_epoch
    alpha = args.alpha
    beta = args.beta
    is_train = args.train
    cuda = args.cuda
    device = torch.device("cuda:0" if (cuda and torch.cuda.is_available()) else "cpu")

    g_net = model.Generator_(latent_dim = x_dim, out_shape= y_dim, n_layers=10, n_units=512)   
    h_net = model.Generator_(latent_dim= y_dim, out_shape = x_dim, n_layers=10, n_units=256)
    dx_net = model.Discriminator_(inp_shape = x_dim, n_layers=2, n_units=128)
    dy_net = model.Discriminator_(inp_shape = y_dim, n_layers=4, n_units=256)

    xs = sampler.Gaussian_sampler(mean=np.zeros(x_dim),sd=1.0)

    if data == 'indep_gmm':
        ys = sampler.GMM_indep_sampler(sd=0.1, dim=y_dim, n_components=3, bound=1, batch_size=batch_size)
    
    elif data == "eight_octagon_gmm":
        n_components = 8
        def cal_cov(theta,sx=1,sy=0.4**2):
            Scale = np.array([[sx, 0], [0, sy]])
            c, s = np.cos(theta), np.sin(theta)
            Rot = np.array([[c, -s], [s, c]])
            T = Rot.dot(Scale)
            Cov = T.dot(T.T)
            return Cov
        radius = 3
        mean = np.array([[radius*math.cos(2*np.pi*idx/float(n_components)),radius*math.sin(2*np.pi*idx/float(n_components))] for idx in range(n_components)])
        cov = np.array([cal_cov(2*np.pi*idx/float(n_components)) for idx in range(n_components)])
        ys = sampler.GMM_sampler(batch_size=batch_size,mean=mean,cov=cov)

    RTM = RoundtripModel(g_net, h_net, dx_net, dy_net, xs, data, ys, alpha, beta, device)
    RTM.train(epochs=epochs,cv_epoch=cv_epoch)