import numpy as np
import scipy
import pandas as pd
from scipy import stats
from math import comb
from abc import ABCMeta, abstractmethod

class ExponentialFamily(metaclass=ABCMeta):
    """
    # https://github.com/madrury/py-glm/blob/83081444e2cbba4d94f9e6b85b6be23e0ff600b8/glm/families.py
    
    A Exponential family must implement the following methods in order
    to work with the GLM class
    
    Methods
    -------
    _inverse_link:
        The inverse link function.
    _variance:
        The variance funtion linking the mean to the variance of the
        distribution.
    _log_likelihood
        The log likelihood function
    _deviance:
        The deviance of the family. Used as a measure of model fit.
    """
    @abstractmethod
    def _inverse_link(self, nu):
        pass

    @abstractmethod
    def _variance(self, nu, mu):
        pass

    @abstractmethod
    def _log_likelihood(self, mu):
        pass

    @abstractmethod
    def _deviance(self, y, mu):
        pass

class LogisticRegression(ExponentialFamily):
            
    @staticmethod
    def _inverse_link(eta):
        """
        Inverse of the link function for the
        bernoulli distribution
        
        Parameters
        ----------
        eta : 2D np.array
            X @ Weights 
        """
        #return scipy.special.expit(eta)
        return 1 / (1 + np.exp(-eta))
    
    @staticmethod
    def _variance(mu, ):
        """
        Variance of bernoulli predictions (mu * (1 - mu))
        
        Parameters
        ----------
        mu: 1D np.array 
            predicted class probabilities (0,1)
        """
        return mu * (1 - mu)
    
    
    @staticmethod
    def _log_likelihood(mu, y, n):
        """
        Find the log likelihood of the bernoulli predictions
        """
        # get the n choose pi term at the beginning of the log likelihood
        
        # calculate the combinations
        nchoosepi = np.zeros((len(n), 1))
        for i in np.arange(len(n)):
            nchoosepi[i, 0] = comb(int(n[i]), int(mu[i]))
            
        return np.sum(np.log(nchoosepi) + y / n * np.log(mu) + (1 - y / n) * np.log(1 - mu))
    
    def _deviance(self, mu, y, n):
        """
        Deviance of the model. Saturated model log likelihood is 1 here
        
        Parameters
        ----------
        x : 2D np.array
            independent variables
        y: 1D np.array
            target variable
        n: int
            vector of number of trials
        """
        
        return -2 * self._log_likelihood(mu, y, n) 
        
        
class PoissonRegression(ExponentialFamily):
    
    @staticmethod
    def _inverse_link(eta):
        """
        Inverse of the link function for the
        poisson distribution
        
        Parameters
        ----------
        eta : 2D np.array
            X @ Weights 
        """
        return np.exp(eta)
    
    @staticmethod
    def _variance(mu, ):
        """
        Variance of poisson predictions (mean)
        """
        return mu
    
    @staticmethod
    def _log_likelihood(mu, y, n):
        """
        Find the log likelihood of the poisson predictions
        """
        # ensure mu is not 0 for deviance calc in saturated model
        log_mu = np.zeros(mu.shape)
        log_mu[mu == 0] = 0
        log_mu[mu != 0] = np.log(mu[mu != 0])
        
        fact = scipy.vectorize(scipy.math.factorial, otypes='O')
        return np.sum(y / n * log_mu - mu - np.log(fact(y / n).astype(np.float64)))
        
    def _deviance(self, mu, y, n):
        """
        Deviance of the model.
        
        Parameters
        ----------
        x : 2D np.array
            independent variables
        y: 1D np.array
            target variable
        n: int
            vector of number of trials
        """
        return 2*np.sum(self._log_likelihood(y, y, n) - self._log_likelihood(mu, y, n))
        
class NegativeBinominalRegression(ExponentialFamily):
    """
    Negative binominal regression
    using the log link.
    
    Models poisson data with extra dispersion
    using the poisson-gamma parameterization
    
    Parameters
    ---------- 
    alpha: float
        alpha value for the negative binominal distribution
        
    """
    def __init__(self, alpha=1):
        self.alpha = alpha
    
    def _inverse_link(self, eta, ):
        """
        Inverse of the link function for the
        negative binominal distribution
        """
        return np.exp(eta)
    
    def _variance(self, mu, ):
        """
        Variance of negative binominal predictions
        
        Returns
        ---------- 
        w_diag, v_diag: tuple
            diagonal elements of W, V variance matricies
        """
        theta_exp =  mu / (self.alpha + mu) # e^theta
        w_diag = self.alpha / (self.alpha + mu) 
        v_diag =  (self.alpha * theta_exp)  / (( 1 - theta_exp )**2)
        return v_diag, w_diag
    
    def _log_likelihood(self, mu, y, n):
        """
        Find the log likelihood of the negative binominal predictions
        """
        # TODO fix 
        
        theta = np.log( mu / (self.alpha + mu) )
        gamma_constant = scipy.special.gamma(y + self.alpha) / (scipy.special.gamma(y + self.alpha) * scipy.special.gamma(y + 1))
        print(np.sum(gamma_constant))
        return np.sum(gamma_constant + y * theta + self.alpha * np.log( self.alpha / (self.alpha * mu) ))
        
    def _deviance(self, mu, y, n):
        """
        Deviance of the model. Saturated model log likelihood is 1 here
        
        Parameters
        ----------
        x : 2D np.array
            independent variables
        y: 1D np.array
            target variable
        n: int
            vector of number of trials
        """
        # TODO fix
        
        log_y = np.empty(y.shape)
        log_y[y == 0] = 0 
        log_y[y != 0] = np.log(y[y != 0])
        return 2 * np.sum(log_y - np.log(mu) - (y +  self.alpha) * np.log( (1 + self.alpha * y) / (1 + self.alpha * mu) ))
        
        
class Gamma:
    """
    Gamma regression
    using the log link
    Parameters
    ---------- 
    alpha: float
        shape parameter for gamma dist
    beta: float
        scale parameter for gamma dist
    """
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        
    def _inverse_link(self, eta, ):
        """
        Inverse of the link function for the
        gamma distribution
        """
        return np.exp(eta)
    
    def _variance(self, mu, eta):
        """
        Variance of negative gamma predictions
        """
        return mu * mu
    
    @staticmethod
    def _log_likelihood(X,):
        """
        Find the log likelihood
        """
        
        raise("Not implemented")
        
        
    def _deviance(self, mu, y, n):
        """
        Deviance of the model. Saturated model log likelihood is 1 here
        
        Parameters
        ----------
        x : 2D np.array
            independent variables
        y: 1D np.array
            target variable
        n: int
            vector of number of trials
        """
        
        return 1