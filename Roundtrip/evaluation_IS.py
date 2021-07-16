import argparse
import torch 
from train_roundtrip import RoundtripModel
import model 
import sampler
import numpy as np
import math 
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

def load(data, g_net, h_net):
        g_net.load_state_dict(torch.load('model_saved/g_net_{}'.format(data)))
        h_net.load_state_dict(torch.load('model_saved/h_net_{}'.format(data)))
        print('Restored model weights.')

class ImportanceSampling(object):
    def __init__(self, g_net, h_net, y_point, dx, dy, sd_y=0.5, scale=0.5, N=40000, df =1):
        super(ImportanceSampling, self).__init__()
        self.g_net = g_net
        self.y = y_point
        self.sd = sd_y
        self.N = N
        self.dx = dx
        self.dy = dy
        self.mean = h_net(y_point)
        self.df = df
        self.scale = scale
        self.q = torch.distributions.studentT.StudentT(df, loc=self.mean, scale=scale)
        testq = self.q.sample()

    def sample_q(self):
        return self.q.rsample(torch.Size([self.N]))
    
    def prob_pz(self, z, log=True):
        if log:
            return torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(self.dy), torch.eye(self.dy)).log_prob(z)
        else:
            return torch.pow(torch.sqrt(2*torch.Tensor([np.pi])),-self.dy) * torch.exp((-1/2)*torch.norm(z, dim=1))
        
    def prob_qz(self, z, log = True):
        if log :
            return torch.sum(self.q.log_prob(z), axis=1)
        else:
            from scipy.stats import t
            z = z.detach().numpy()
            prob = np.prod(t.pdf(z,self.df,loc=self.mean.detach().numpy(),scale=self.scale), axis=1)
            return torch.Tensor(prob)
    
    def weights(self, z, log = True):
        if log:
            return self.prob_pz(z) - self.prob_qz(z)
        else:
            return self.prob_pz(z) / self.prob_qz(z)
    
    def prob_y_given_x(self, z, log = True):
        g_z = self.g_net(z)
        if log:
            return -self.dy*torch.sqrt(2*torch.Tensor([np.pi]))-(torch.norm(self.y-g_z,dim=1))/(2.*self.sd**2)
        else:
            return 1/torch.pow(torch.sqrt(2*torch.Tensor([np.pi])),self.dy) * torch.exp(-torch.norm(self.y-g_z, dim=1)/(2.*self.sd**2))

    def evaluate(self, log = False):
        z = self.sample_q()
        w = self.weights(z, log)
        prob_y_given_x = self.prob_y_given_x(z, log)
        if log:
            return torch.Tensor([1/self.N]) + torch.sum(prob_y_given_x + w) 
        else:
            return torch.Tensor([1/self.N]) * torch.sum(prob_y_given_x * w) 


def create_2d_grid_data(x1_min, x1_max, x2_min, x2_max,n=100):
    grid_x1 = np.linspace(x1_min, x1_max, n)
    grid_x2 = np.linspace(x2_min, x2_max, n)
    v1,v2 = np.meshgrid(grid_x1,grid_x2)
    data_grid = np.vstack((v1.ravel(),v2.ravel())).T
    return v1, v2, data_grid


def visualization_2d(x1_min, x1_max, x2_min, x2_max, sd_y, scale, n=100):
    v1, v2, data_grid = create_2d_grid_data(x1_min, x1_max, x2_min, x2_max,n)
    py = []
    for i in tqdm(data_grid):
        IS = ImportanceSampling(g_net, h_net, torch.Tensor(i), dx = args.dx, dy = args.dy, N=args.N)
        py.append(IS.evaluate().detach().numpy())
    py = np.array(py)
    py = py.reshape((n,n))
    plt.figure()
    plt.rcParams.update({'font.size': 22})
    plt.imshow(py, extent=[v1.min(), v1.max(), v2.min(), v2.max()], cmap='Blues', alpha=0.9)
    plt.colorbar()
    plt.savefig('images/2d_grid_density_pre_{}.png'.format(args.data))
    plt.close()    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='indep_gmm', help='Name od the data type')
    parser.add_argument('--dx', type=int, default=2, help='Dimention of the latent space')
    parser.add_argument('--dy', type=int, default=2, help='Dimention of the data space') 
    parser.add_argument('--N', type=int, default=1000, help='Sample size') 

    args = parser.parse_args()

    g_net = model.Generator_(latent_dim = args.dx, out_shape= args.dy, n_layers=10, n_units=512)   
    h_net = model.Generator_(latent_dim= args.dy, out_shape = args.dx, n_layers=10, n_units=256)

    xs = sampler.Gaussian_sampler_(mean=np.zeros(args.dx),sd=1.0, batch_size = args.N)

    if args.data == 'indep_gmm':
        best_sd, best_scale = 0.05, 0.5
        ys = sampler.GMM_indep_sampler_(sd=0.1, dim=args.dy, n_components=3, bound=1, batch_size=args.N)
            
    elif args.data == "eight_octagon_gmm":
        best_sd, best_scale = 0.1, 0.5
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
        ys = sampler.GMM_sampler_(batch_size=args.N,mean=mean,cov=cov)
    
    load(args.data, g_net, h_net)
    
    y_point = torch.Tensor(ys.sample()[0])
    
    IS = ImportanceSampling(g_net, h_net, y_point, dx = args.dx, dy = args.dy, sd_y=best_sd, scale=best_scale, N=args.N)

    evaluate = IS.evaluate()
    print(evaluate)

    if args.data == "indep_gmm":
        visualization_2d(-1.5, 1.5, -1.5, 1.5, 0.05, 0.5)
    elif args.data == "eight_octagon_gmm":
        visualization_2d(-5, 5, -5, 5, 0.1, 0.5)
    