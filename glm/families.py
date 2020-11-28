import numpy as np
import scipy
import pandas as pd
from scipy import stats
from math import comb

class LogisticRegression:
            
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
        
        
class PoissonRegression:
    
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
        Find the log likelihood of the bernoulli predictions
        """
        factorial = np.log( scipy.special.factorial(y.astype(np.int64)) )
        return None
        raise("Not implemented")
        
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
        return 2 * np.sum((y - mu) / mu - np.log(y / mu))
        raise("Not implemented")
        
class NegativeBinominalRegression:
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
        theta_exp = np.exp( np.log( mu / (self.alpha + mu) ) ) # e^theta
        w_diag = self.alpha / (self.alpha + mu) 
        v_diag =  (self.alpha * theta_exp)  / (( 1 - theta_exp )**2)
        return v_diag, w_diag
    
    @staticmethod
    def _log_likelihood(X,):
        """
        Find the log likelihood
        """
        
        raise("Not implemented")
        
        
class Gamma:
    """
    Gamma regression
    using the log link
    """ 
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