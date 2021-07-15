import torch 
import pyro
import numpy as np
import math

pyro.set_rng_seed(101)

class GMM_indep_sampler(object):
    def __init__(self, sd, dim, n_components, batch_size = 64, weights=None, bound=1):
        #np.random.seed(1024)
        self.dim = dim
        self.sd = sd
        self.n_components = n_components
        self.batch_size = batch_size
        self.centers = np.linspace(-bound, bound, n_components)
        self.X = np.vstack([self.generate_gmm() for _ in range(dim)]).T
        
    def generate_gmm(self, weights = None):
        if weights is None:
            weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        Y = np.random.choice(self.n_components, size=self.batch_size, replace=True, p=weights)
        return np.array([np.random.normal(self.centers[i],self.sd) for i in Y],dtype='float64')
    
    def sample(self):
        return np.vstack([self.generate_gmm() for _ in range(self.dim)]).T
    
class Gaussian_sampler(object):
    def __init__(self, mean, sd=1, batch_size = 64):
        self.mean = mean
        self.sd = sd
        self.batch_size= batch_size
        #np.random.seed(1024)

    def sample(self):
        return np.random.normal(self.mean, self.sd, (self.batch_size,len(self.mean)))

class GMM_sampler(object):
    def __init__(self, batch_size=64, mean=None, n_components=None, cov=None, sd=None, dim=None, weights=None):
        #np.random.seed(1024)
        self.batch_size = batch_size
        self.n_components = n_components
        self.dim = dim
        self.sd = sd
        self.weights = weights
        if mean is None:
            assert n_components is not None and dim is not None and sd is not None
            self.mean = np.random.uniform(-5,5,(self.n_components,self.dim))
        else:
            assert cov is not None    
            self.mean = mean
            self.n_components = self.mean.shape[0]
            self.dim = self.mean.shape[1]
            self.cov = cov
       
    def sample(self):
        if self.weights is None:
            self.weights = np.ones(self.n_components, dtype=np.float64) / float(self.n_components)
        self.Y = np.random.choice(self.n_components, size=self.batch_size, replace=True, p=self.weights)
        if self.mean is None:
            self.X = np.array([np.random.normal(self.mean[i],scale=self.sd) for i in self.Y],dtype='float64')
        else:
            self.X = np.array([np.random.multivariate_normal(mean=self.mean[i],cov=self.cov[i]) for i in self.Y],dtype='float64')
        return self.X

        
if __name__ == "__main__":
    sampler = GMM_indep_sampler(sd=0.1, dim=2, n_components=3, batch_size = 5, bound=1)
    print(sampler.sample())
    gaus_sampler = Gaussian_sampler(mean=np.zeros(2),sd=1.0, batch_size=5)
    print(gaus_sampler.sample())
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
    octa_sampler = GMM_sampler(batch_size=5,mean=mean,cov=cov)
    print(octa_sampler.sample())