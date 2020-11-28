from collections import namedtuple
import pandas as pd
import numpy as np
from numpy.linalg import inv, norm
from scipy import stats

class GLM:
    f"""
    Generalized linear model
    
     Parameters
    ----------
    x : 2D np.array
        independent variables
    faimly: 
        class of model 
    max_iterations: int
        maximum number of iterations to reach convergence
    tol: float
        stopping criteria 
    alpha : float
        significance level - default=0.05
    """
    def __init__(self, family=None, max_iterations=5000, tol=1e-5, alpha=0.05):
        # instantiate model of family's class
        self.family = family
        self.max_iterations = max_iterations
        self.tol = tol
        self.alpha = alpha
        self.isfit = False # indicator to see if model has been fit
    
    def _find_weights(self, X, y, n, save_weights=True):
        """
        Estimate parameters using Iteratively Reweighted Least Squares (IRLS) algorithm with
        Newton-Raphson 
        
        Parameters
        ----------
        x : 2D np.array
            independent variables
        y: 1D np.array
            target variable
        n: 1D np.array int
            vector of number of trials
        save_weights: bool
            save model weights to the class object
        """
        
        # initialize weights with OLS estimate
        weights = inv(X.T @ X) @ X.T @ (y / self.n)
        old_weights = weights
        
        # intialize Variance matrices
        W = np.identity(len(y), np.float64) # used for non canonical links
        V = np.zeros((len(y), len(y)), np.float64)
            
        # iterate until convergence/max iter
        for i in range(self.max_iterations):
            
            # calculate new linear predictions
            eta = X @ weights
            
            # use the inverse link to transform the predictions into the response dist
            mu = self.family._inverse_link(eta)
            
            # find variance matricies
            variance_diag = self.family._variance(mu, ) # find diag elements of variance matrix/matrices
            if type(variance_diag) == tuple: # diag elements of W, V
                np.fill_diagonal(V, self.n * variance_diag[0])
                np.fill_diagonal(W, self.n * variance_diag[1])
                
            else: # just updating V matrix (W is identity)
                np.fill_diagonal(V, self.n * variance_diag)
            
            # find z, in logistic case, z is a bernoulli rv
            z = eta + inv(V) @ (y / self.n - mu)
            
            # update weights with new estimate based off of z
            weights = inv(X.T @ W @ V @ X) @ X.T @ W @ V @ z
            
            # early stopping if convergence is reached, otherwise update loss
            if abs(norm(old_weights) - norm(weights)) < self.tol:
                if save_weights:
                    self.weights = weights
                    self.isfit = True
                    self.X = X
                    self.y = y
                    self.n_iter = i
                    self.log_likeli = self.family._log_likelihood(mu, y, self.n)
                    return
                else:
                    return weights
            else:
                # store weights to check for convergence
                old_weights = weights
            
        print(f"Convergence was not reached in {self.max_iterations} iterations")
        if save_weights:
            self.weights = weights
            self.isfit = True
            self.X = X
            self.y = y
            self.n_iter = max_iterations
            self.log_likeli = self.family._log_likelihood(mu, y, self.n)
            return
        else:
            return weights
        
    def fit(self, X, y, n=None, save_weights=True, intercept=True):
        """
        Parameters
        ----------
        x : 2D np.array
            independent variables
        y: 1D np.array
            target variable
        n: 1D np.array int
            vector of number of trials
        save_weights: bool
            save model weights to the class object
        intercept: bool
            fit the model with an intercept term
        """
        # convert from pandas -> numpy if needed
        if type(X) == pd.DataFrame:
            X = X.values
        if type(X) == pd.DataFrame or type(y) == pd.Series:
            y = y.values
        y = y.reshape(-1, ) 
        
        # add an intercept term if needed
        if save_weights:
            self.intercept = intercept
        if intercept:
            self.intercept = intercept
            X =  np.column_stack( (np.ones((len(X), 1)), X) )
            
        # set dimension parameters for later use
        self.n = X.shape[0]
        self.p = X.shape[1]
        
        # Case for no repeated measures
        if not n:
            self.n = np.ones((len(y), 1))
        
        # run gradient descent
        if save_weights: 
            # save weights to class
            self._find_weights(X, y.reshape(-1, 1), n=n, save_weights=save_weights)
        else: 
            # return weights
            return self._find_weights(X, y.reshape(-1, 1), n=n, save_weights=save_weights)
        
        # return the full model (class)
        return self
        
    def likelihood_ratio_test(self, X, y, remaining_columns, n=None, intercept=True):
        """
        Perform a likelihood ratio test to see
        if the log likelihood is increased significantly
        by including the extra variable (columns)
        
        LRT = 2[logL(full) - logL(reduced)]
        
        Parameters
        ----------
        x : 2D np.array
            independent variables
        y: 1D np.array
            target variable
        remaining_columns: list of ints
            list of column indicies to keep in reduced model
        n: int
            vector of number of trials
        intercept: bool
            fit the model with an intercept term
        """
        #TODO figure out why log likelihood is negative
            
        # add an intercept term if needed
        if intercept:
            X =  np.column_stack( (np.ones((len(X), 1)), X) )
        
        # find degrees of freedom for the test
        df = X.shape[1] - len(remaining_columns)
        
        # verify the test can be performed
        if df < 1:
            print(f"Unable to perform LRT with {X.shape[1]} original columns"
                  f"and {len(remaining_columns)} columns in the subset."
                  f"The resulting degrees of freedom for the test is: {df}")
            raise("error")
        
        # find model weights
        full_weights = self.fit(X, y, n, save_weights=False, intercept=False)
        reduced_weights = self.fit(X[:, remaining_columns], y, n, save_weights=False, intercept=False)
        
        # make predictions with model weights
        mu_full = self.family._inverse_link(X @ full_weights)
        mu_reduced = self.family._inverse_link(X[:, remaining_columns] @ reduced_weights)
        
        # calculate log likelihoods
        full_log_likeli = self.family._log_likelihood(mu_full, y, self.n)
        reduced_log_likeli = self.family._log_likelihood(mu_reduced, y, self.n)
        
        # run the LRT
        lrt = 2 * (full_log_likeli - reduced_log_likeli)
        p_val = 1 - stats.chi2.cdf(lrt, df)
        
        # store the results in a dataframe
        results = pd.DataFrame()
        results['LRT statistic'] = [lrt]
        results['p-value'] = [p_val]
        
        return results
        
    
    def predict(self, X):
        """
        Calculates predicted values for observations 
        
        Parameters
        ----------
        x : -D np.array
            independent variables
        """
        return np.round(self.family._inverse_link(X @ self.weights))
    
    def predict_proba(self, X):
        """
        Calculates, return the predicted probability for
        an observation in X 
        
        Parameters
        ----------
        x : 2D np.array
            independent variables
        """
        return self.family._inverse_link(X @ self.weights)
    
    def _standard_error(self, X):
        """
        Calculates standard error for the weights
        
        Parameters
        ----------
        x : 2D np.array
            independent variables
        """
        
        # find the standard errors of the weights
        eta = X @ self.weights
        mu = self.family._inverse_link(eta) # predicted values
        #V = np.zeros((len(mu), len(mu)))
        #np.fill_diagonal(V, (self.n * self.family._variance(mu, )))
        V = np.diag( (self.n * self.family._variance(mu, )).reshape(-1) )
        
        # calculate stndard error
        se = np.sqrt( np.diag(inv(X.T @ V @ X)) )
        
        
        return se.reshape(-1, 1)
    
    def _wald_inference(self, ):
        """
        Test the null hypothesis that the coefficients (weights)
        are equal to 0 vs. the alternative hypothesis that they are
        not equal to 0. We are using the fact that the weights divided 
        by their standard error are N(0,1) under the null hypothesis
        
        Parameters
        ----------
        
        """
        
        # find the standard error of the coefficients
        se = self._standard_error(self.X)
        
        # calculate z statistics
        z_stats = self.weights / se
        
        # find the p-values
        p_values = stats.norm.sf(np.abs(z_stats)) * 2 # two sided
        
        # find confidence intervals
        lower_bounds = self.weights + stats.norm.ppf(self.alpha / 2) * se
        upper_bounds = self.weights + stats.norm.ppf(1 - self.alpha / 2) * se
        
        # get coefficient names
        if self.intercept:
            coef_names = ["int"] + [f"b{i}" for i in np.arange(len(upper_bounds) - 1)]
        else:
            coef_names =  [f"b{i}" for i in np.arange(len(upper_bounds))]
            
        # store results in namedtuple
        results = namedtuple('results', ['se', 'z_stats', 'p_values', 'lower_bounds',
                                        'upper_bounds', 'coef_names'])
        
        return results(se=se,
                          z_stats=z_stats,
                          p_values=p_values,
                          lower_bounds=lower_bounds,
                          upper_bounds=upper_bounds,
                          coef_names=coef_names)
        
    def _deviance(self, ):
        """
        Measure of deviance 
        
        Determine if the model is significantly 
        worse than the saturated model. Saturated 
        model is when number of parameters = 
        number of observations
        
        Deviance = -2ln(L(model) / L(saturated model)) ~ 
        asymtotically chi-sq with df=(n-p), where p = the number 
        of parameters in the model, n = number of observations.
        
        An insignificant value of deviance is an upper-tail
        test would imply that the fit of the model is not
        significantly worse than the saturated model.
        
        H0: 
        H1: 
        
        Parameters
        ----------
    
        """
        
        # TODO: Check deviance calculation
        
        
        # find class probabilities using estimated weights
        mu = self.family._inverse_link(self.X @ self.weights)
        
        # find the deviance for the family
        dev = self.family._deviance(mu, self.y, self.n)
        
        # find p-value (deviance is asymtotically chi-sq dist)
        df_model = self.weights.shape[0]
        if self.intercept:
            df_model = df_model - 1
        df = len(self.X) - df_model
        p_value = 1 - stats.chi2.cdf(dev, df)
        
        # store results in namedtuple
        results = namedtuple('results', ['df', 'deviance', 'p_value'])
        
        return results(df=df, deviance=dev, p_value=p_value)
        
    def summary(self, ):
        from tabulate import tabulate
        
        # check if model has been fit
        if not self.isfit:
            raise ValueError("Model must be fit before calling a summary")
        
        # get the deviance
        dev = self._deviance()
        
        # print basic information
        summary = [["No. Observations", f"{self.X.shape[0]}"],
                   ["No. Variables", f"{self.X.shape[1]}"],
                   ["No. Iterations", f"{self.n_iter}"],
                   ["Log-Likelihood", f"{self.log_likeli}"],
                   ["Deviance", f"{dev.deviance}",],
                   ["p_value", f"{dev.p_value}"],
                  ]
        print(tabulate(summary, showindex="never"))
        
        print("")
        
        # get confidence intervals for the coefficients
        ci = self._wald_inference()
        
        header = ["", "coef", "std err", "z", "P>|z|", f"[{self.alpha/2}", f"{1-self.alpha/2}]"]
        ci_data = [[ci.coef_names[i], self.weights[i], ci.se[i], ci.z_stats[i], ci.p_values[i], 
                   ci.lower_bounds[i], ci.upper_bounds[i]] for i in np.arange(len(self.weights))]
        summary = [header, ] + ci_data
        
        print(tabulate(summary, showindex="never", headers="firstrow"))
    
    def prediction_ci(self, X, y=None, alpha=0.05, n=None):
        """
        Confidence intervals around the predicted 
        y values for the given X values
        
        Parameters
        ----------
        x : 2D np.array
            independent variables
        y : 1D np.array
            target variable
        alpha: float
            significance level (default: alpha=0.05)
        n: 1D np.array int
            vector of number of trials
        """
        
        # TODO: check prediction intervals
        
        # check for intercept
        if self.intercept:
            X =  np.column_stack( (np.ones((len(X), 1)), X) )
        
        # get predicted eta value
        eta = X @ self.weights
        
        # get the predicted value
        mu = self.family._inverse_link(eta)
        
        # intialize Variance matrix
        V = np.zeros((len(self.weights), len(self.weights)), np.float64)
        
        # get variance
        np.fill_diagonal(V, ( self.family._variance(mu, )))
        
        # get standard error of eta
        se = V[0,0]
        for i in np.arange(X.shape[1]):
            se = se + X[:, i]**2 * V[i,i]

        # get the interval bounds
        lower_bounds = self.family._inverse_link(eta + stats.norm.ppf(alpha / 2) * se.reshape(-1,1))
        upper_bounds = self.family._inverse_link(eta + stats.norm.ppf(1 - alpha / 2) * se.reshape(-1,1))
        
        # store results in a dataframe
        results = pd.DataFrame()
        results['predicted value'] = mu.reshape(-1)
        results['eta standard error'] = se
        results[f'[{alpha / 2}'] = lower_bounds
        results[f'{1 - alpha / 2}]'] = upper_bounds
        
        if y is not None:
            results['True Value'] = y
        
        return results