import torch 
import pyro
import numpy as np

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

        
if __name__ == "__main__":
    sampler = GMM_indep_sampler(sd=0.1, dim=2, n_components=3, batch_size = 5, bound=1)
    print(sampler.sample())
    gaus_sampler = Gaussian_sampler(mean=np.zeros(2),sd=1.0, batch_size=5)
    print(gaus_sampler.sample())