import sys
sys.path.insert(0,'.')
import model
import unittest
import torch

class Test(unittest.TestCase):
    def test_model_img(self):
        generator = model.Generator_img(img_size = 32, channels = 1, latent_dim = 100)
        discriminator = model.Discriminator_img(1, 32)
        generator.apply(model.weights_init_normal)
        discriminator.apply(model.weights_init_normal)        
        img_samp = generator(torch.rand(100).view((1, 100)))
        dis_val = discriminator(img_samp)
        self.assertEqual(img_samp.shape, (1, 1, 32, 32))
        self.assertGreaterEqual(dis_val, 0)
    
    def test_model_img(self):
        generator = model.Generator(latent_dim = 10, out_shape = (10, 10))
        discriminator = model.Discriminator(inp_shape = (10, 10))
        img_samp = generator(torch.rand(20).view((2, 10)))
        dis_val = discriminator(img_samp)
        self.assertEqual(img_samp.shape, (2, 10, 10))
        self.assertGreaterEqual(dis_val[0], 0)
        self.assertGreaterEqual(dis_val[1], 0)

if __name__ == "__main__":
    unittest.main()
