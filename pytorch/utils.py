import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from inception import InceptionV3
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from scipy import linalg

model = InceptionV3()
model.eval()

def add_channel(images, augmentation_bits, real):
    if augmentation_bits is None:
        labels = (
            torch.ones(size=(images.size(0),), device=images.device)
            if real
            else torch.zeros(size=(images.size(0),), device=images.device)
        )
        return images, labels
    else:
        #XOR
        if real:
            labels = augmentation_bits.clone()
            labels[augmentation_bits==0] = 1
            labels[augmentation_bits==1] = 0
        else:
            labels = augmentation_bits.clone()

        #Add channels
        print('augmentation bits size: ', augmentation_bits.size())
        #print('matrix size', torch.ones(1,28,28).size())
        matrix = torch.ones(1,28,28)
        batch_size = augmentation_bits.size()[1]
        batch_matrix = torch.ones(batch_size,28,28)
        print('batch_matrix size: ', batch_matrix.size())
        print('matrix size: ', matrix.size())
        augmented_matrix = augmentation_bits[:,:,28,28]
        print('augmented_matrix size: ', augmented_matrix)
        #new_channels = torch.mul(augmentation_bits, matrix)
        #new_channels = batch_matrix*augmentation_bits[0,:].expand_as(batch_matrix)
        #new_channels = matrix * augmentation_bits
        #new_channels = np.array(augmentation_bits[0,:]) * np.array(matrix[:,28,28])
        #print('from numpy:', torch.from_numpy(new_channels).size())
        #new_channels = torch.stack(torch.from_numpy(new_channels))

        print('new channels size: ', new_channels.size())


        return images, labels


def compute_kid(real, fake, batch_size=128):
    model.to(real.device)
    print("Computing Inception Activations")
    real_activations = get_activations(real, model, batch_size)
    fake_activations = get_activations(fake, model, batch_size)
    print("Computing KID")
    return _kid(real_activations, fake_activations)


def compute_fid(real, fake, batch_size=128):
    model.to(real.device)
    real_activations = get_activations(real, model, batch_size)
    real_mu = np.mean(real_activations, axis=0)
    real_sigma = np.cov(real_activations, rowvar=False)

    fake_activations = get_activations(fake, model, batch_size)
    fake_mu = np.mean(fake_activations, axis=0)
    fake_sigma = np.cov(fake_activations, rowvar=False)
    return _fid(real_mu, real_sigma, fake_mu, fake_sigma)


def get_activations(images, model, batch_size=128, dims=2048):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """

    loader = DataLoader(images, batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing inception activations"):
            pred = model(batch)[0]
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            
            preds.append(pred)
        preds = torch.cat(preds, dim=0).squeeze().cpu().detach().numpy()
    return preds

def _kid(X, Y):
    """
    Given X, Y (numpy) batches of inception outputs of generated and real images,
    return the Kernel Inception Distance. X and Y have to have the same dimensions.
    """
    assert np.all(X.shape == Y.shape)

    n = X.shape[0]

    def k(x, y):
        # Kernel
        return (1/x.shape[0]*np.dot(x, y)+1)**(1/3)

    def f(X, Y):
        # First 2 sums. We use the fact that k is symmetric
        res = 0
        for i in range(n):
            for j in range(i+1, n):
                res += k(X[i], Y[j])
        
        return 2*res
    
    def f2(X, Y):
        # Third sum.
        res = f(X, Y)

        for i in range(n):
            res += k(X[i], Y[i])
        
        return res

    return 1/(n*(n-1))*f(X, X) + 1/(n*(n-1))*f(Y, Y) - 2/(n**2)*f2(X, Y)


def _fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an 
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an 
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
