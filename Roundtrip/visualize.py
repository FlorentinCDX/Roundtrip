import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from train_roundtrip import RoundtripModel
import sampler
import argparse
import model

class RTMload:
    def __init__(self, x_dim, y_dim, n_particules=1000) -> None:
        self.g_net = model.Generator_(latent_dim = x_dim, out_shape= y_dim, n_layers=10, n_units=512)   
        self.h_net = model.Generator_(latent_dim= y_dim, out_shape = x_dim, n_layers=10, n_units=256)
        self.dx_net = model.Discriminator_(inp_shape = x_dim, n_layers=2, n_units=128)
        self.dy_net = model.Discriminator_(inp_shape = y_dim, n_layers=4, n_units=256)
        self.xs = sampler.Gaussian_sampler_(mean=np.zeros(x_dim),sd=1.0, batch_size = n_particules)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_particules = n_particules
    def visualize(self):
        pass

class RTModel(RTMload):
    def __init__(self, data = 'indep_gmm', x_dim = 2, y_dim = 2):
        super(RTModel, self).__init__(x_dim, y_dim)
        self.data = data
        if data == 'indep_gmm':
            self.ys = sampler.GMM_indep_sampler_(sd=0.1, dim=y_dim, n_components=3, bound=1, batch_size=self.n_particules)
            
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
            self.ys = sampler.GMM_sampler_(batch_size=self.n_particules,mean=mean,cov=cov)
        
        model = RoundtripModel(self.g_net, self.h_net, self.dx_net, self.dy_net, self.xs, data, self.ys, 10, 10)
        self.model = model.load()

    def visualize_true_vs_emulate_data(self):
        assert self.y_dim == 2
        true_sample = self.ys.sample()
        latent_sample = torch.Tensor(self.xs.sample())
        fake_sample = self.g_net(latent_sample).detach().numpy()
        fig, ax = plt.subplots( nrows=1, ncols=2 )
        ax[0].scatter(true_sample[:, 0], true_sample[:, 1], color='green')
        ax[0].set_title('Sample from the true data space')
        ax[1].scatter(fake_sample[:, 0], fake_sample[:, 1], color='blue')
        ax[1].set_title('Data sample recovers from the generator NN')

        fig.savefig('images/data_{}.png'.format(self.data))
        plt.close(fig)
        print("A figure has been saved in the images folder")
    
    def visualize_true_vs_emulate_latent(self):
        assert self.y_dim == 2
        data_sample = torch.Tensor(self.ys.sample())
        true_sample = self.xs.sample()
        fake_sample = self.h_net(data_sample).detach().numpy()
        fig, ax = plt.subplots( nrows=1, ncols=2 )
        ax[0].scatter(true_sample[:, 0], true_sample[:, 1], color='green')
        ax[0].set_title('Sample from the true data')
        ax[1].scatter(fake_sample[:, 0], fake_sample[:, 1], color='blue')
        ax[1].set_title('Sample from the generator NN')

        fig.savefig('images/latent_{}.png'.format(self.data))
        plt.close(fig)
        print("A figure has been saved in the images folder")

    def visualize_evolution(self):
        assert self.y_dim == 2
        data_sample = torch.Tensor(self.ys.sample())
        latent_sample = self.h_net(data_sample)
        recon_sample = self.g_net(latent_sample).detach().numpy()
        fig, ax = plt.subplots( nrows=1, ncols=2 )
        ax[0].scatter(data_sample[:, 0], data_sample[:, 1], color='green')
        ax[0].set_title('Sample from the true data')
        ax[1].scatter(recon_sample[:, 0], recon_sample[:, 1], color='blue')
        ax[1].set_title('Sample from the generator NN')

        fig.savefig('images/evolution_{}.png'.format(self.data))
        plt.close(fig)
        print("A figure has been saved in the images folder")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='indep_gmm', help='Name od the data type')
    parser.add_argument('--dx', type=int, default=10, help='Dimention of the latent space')
    parser.add_argument('--dy', type=int, default=10, help='Dimention of the data space') 
    args = parser.parse_args()

    model = RTModel(data=args.data, x_dim=args.dx, y_dim=args.dy)

    model.visualize_true_vs_emulate_data()
    model.visualize_true_vs_emulate_latent()
    model.visualize_evolution()