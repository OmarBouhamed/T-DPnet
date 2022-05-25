#!/usr/bin/python 3.6
#-*-coding:utf-8-*-

'''
Utility functions
'''
import torch 
import numpy as np
import os
import random

def get_data_path():
    folder = os.path.dirname(__file__)
    return os.path.join(folder, "data")

def RSE(ypred, ytrue):
    rse = np.sqrt(np.square(ypred - ytrue).sum()) / \
            np.sqrt(np.square(ytrue - ytrue.mean()).sum())
    return rse

def quantile_loss(ytrue, ypred, qs):
    '''
    Quantile loss version 2
    Args:
    ytrue (batch_size, output_horizon)
    ypred (batch_size, output_horizon, num_quantiles)
    '''
    L = np.zeros_like(ytrue)
    for i, q in enumerate(qs):
        yq = ypred[:, :, i]
        diff = yq - ytrue
        L += np.max(q * diff, (q - 1) * diff)
    return L.mean()

def SMAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel() + 1e-4
    mean_y = (ytrue + ypred) / 2.
    return np.mean(np.abs((ytrue - ypred) \
        / mean_y))

def MAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel() + 1e-4
    ypred = np.array(ypred).ravel()
    return np.mean(np.abs((ytrue - ypred) \
        / ytrue))

def train_test_split(X, y, train_ratio=0.7):
    num_ts, num_periods, num_features = X.shape
    train_periods = int(num_periods * train_ratio)
    random.seed(2)
    Xtr = X[:, :train_periods, :]
    ytr = y[:, :train_periods]
    Xte = X[:, train_periods:, :]
    yte = y[:, train_periods:]
    return Xtr, ytr, Xte, yte

class StandardScaler:
    
    def fit_transform(self, y):
        self.mean = np.mean(y)
        self.std = np.std(y) + 1e-4
        return (y - self.mean) / self.std
    
    def inverse_transform(self, y):
        return y * self.std + self.mean

    def transform(self, y):
        return (y - self.mean) / self.std

class MaxScaler:

    def fit_transform(self, y):
        self.max = np.max(y)
        return y / self.max
    
    def inverse_transform(self, y):
        return y * self.max

    def transform(self, y):
        return y / self.max


class MeanScaler:
    
    def fit_transform(self, y):
        self.mean = np.mean(y)
        return y / self.mean
    
    def inverse_transform(self, y):
        return y * self.mean

    def transform(self, y):
        return y / self.mean

class LogScaler:

    def fit_transform(self, y):
        return np.log1p(y)
    
    def inverse_transform(self, y):
        return np.expm1(y)

    def transform(self, y):
        return np.log1p(y)


def gaussian_likelihood_loss(z, mu, sigma):
    '''
    Gaussian Liklihood Loss
    Args:
    z (tensor): true observations, shape (num_ts, num_periods)
    mu (tensor): mean, shape (num_ts, num_periods)
    sigma (tensor): standard deviation, shape (num_ts, num_periods)

    likelihood: 
    (2 pi sigma^2)^(-1/2) exp(-(z - mu)^2 / (2 sigma^2))

    log likelihood:
    -1/2 * (log (2 pi) + 2 * log (sigma)) - (z - mu)^2 / (2 sigma^2)
    '''
    negative_likelihood = torch.log(sigma + 1) + (z - mu) ** 2 / (2 * sigma ** 2) + 6
    return negative_likelihood.mean()

def negative_binomial_loss(ytrue, mu, alpha):
    '''
    Negative Binomial Sample
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)

    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))

    minimize loss = - log l_{nb}

    Note: torch.lgamma: log Gamma function
    '''
    batch_size, seq_len = ytrue.size()
    likelihood = torch.lgamma(ytrue + 1. / alpha) - torch.lgamma(ytrue + 1) - torch.lgamma(1. / alpha) \
        - 1. / alpha * torch.log(1 + alpha * mu) \
        + ytrue * torch.log(alpha * mu / (1 + alpha * mu))
    return - likelihood.mean()

def gamma_likelihood_loss(z, alpha, beta):
    '''
    Gamma Liklihood Loss
    Args:
    z (tensor): true observations, shape (num_ts, num_periods)
    alpha (tensor): shape param, shape (num_ts, num_periods)
    beta (tensor): scale param, shape (num_ts, num_periods)

    likelihood: 
    z**(alpha-1) * exp(-beta*z) * beta**alpha / Gamma(alpha) 

    log likelihood:
    (alpha-1)*log(z) - beta*z + alpha*log(beta) - log(Gamma(alpha))
    '''
    log_unnormalized_prob = torch.xlogy(alpha - 1., z) - beta * z
    log_normalization = torch.lgamma(alpha) - alpha * torch.log(beta)
    likelihood = log_unnormalized_prob - log_normalization 
    return -likelihood.mean()

def Betaprm_likelihood_loss(z, alpha, beta):
    '''
    Beta prime Liklihood Loss
    Args:
    z (tensor): true observations, shape (num_ts, num_periods)
    alpha (tensor): shape param, shape (num_ts, num_periods)
    beta (tensor): scale param, shape (num_ts, num_periods)

    likelihood: 
    z**(alpha-1) * (1+z)**-(alpha+beta) / Beta(alpha,beta)  

    log likelihood:
    (alpha-1)*log(z) - (alpha+beta)*log(z+1) - log(Beta(alpha, beta))
    '''
    likelihood = torch.lgamma(alpha+beta)-torch.lgamma(alpha)-torch.lgamma(beta)+ torch.xlogy(alpha - 1., z)- torch.xlogy(alpha + beta, z+1)
    return -likelihood.mean()
def Igamma_likelihood_loss1(z, alpha, beta):
    '''
    Beta prime Liklihood Loss
    Args:
    z (tensor): true observations, shape (num_ts, num_periods)
    alpha (tensor): shape param, shape (num_ts, num_periods)
    beta (tensor): scale param, shape (num_ts, num_periods)

    likelihood: 
    z**(alpha-1) * (1+z)**-(alpha+beta) / Beta(alpha,beta)  

    log likelihood:
    (alpha-1)*log(z) - (alpha+beta)*log(z+1) - log(Beta(alpha, beta))
    '''
    likelihood = torch.xlogy(alpha , beta) - torch.lgamma(alpha) - torch.xlogy(1+alpha, z) - (beta/z)
    return -likelihood.mean()

def Igamma_likelihood_loss(z, alpha, beta):

    likelihood = -torch.xlogy(alpha+1 , z) - torch.lgamma(alpha) - (1/z)
    return -likelihood.mean()

def Igaussian_likelihood_loss(z, mu, sigma):

    negative_likelihood = torch.log(z + 1) + (z - mu) ** 2 / (z* 2 * (mu ** 2)) + 6 
   # negative_likelihood = 0.5*( 3* torch.log(z + 1) + math.log(2*math.pi) ) + (z - mu) ** 2 / (z* 2 * (mu ** 2)) 
    return negative_likelihood.mean()

def batch_generator(X, y, num_obs_to_train, seq_len, batch_size):
    '''
    Args:
    X (array like): shape (num_samples, num_features, num_periods)
    y (array like): shape (num_samples, num_periods)
    num_obs_to_train (int):
    seq_len (int): sequence/encoder/decoder length
    batch_size (int)
    '''
    num_ts, num_periods, _ = X.shape
    if num_ts < batch_size:
        batch_size = num_ts
    t = random.choice(range(num_obs_to_train, num_periods-seq_len))
    batch = random.sample(range(num_ts), batch_size)
    X_train_batch = X[batch, t-num_obs_to_train:t, :]
    y_train_batch = y[batch, t-num_obs_to_train:t]
    Xf = X[batch, t:t+seq_len]
    yf = y[batch, t:t+seq_len]
    return X_train_batch, y_train_batch, Xf, yf


def compute_quantile_loss(y_true, y_pred, quantile):
    """
    
    Parameters
    ----------
    y_true : 1d ndarray
        Target value.
        
    y_pred : 1d ndarray
        Predicted value.
        
    quantile : float, 0. ~ 1.
        Quantile to be evaluated, e.g., 0.5 for median.
    """
    residual = y_true - y_pred
    return np.maximum(quantile * residual, (quantile - 1) * residual)
