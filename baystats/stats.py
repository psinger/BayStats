import pymc as pymc
from pymc import Normal, Uniform, MvNormal, Exponential
from numpy.linalg import inv, det
from numpy import log, pi, dot
import numpy as np
from scipy.special import gammaln
import pylab

def _correlation_model(data, robust=False):
    # priors might be adapted here to be less flat
    mu = Normal('mu', 0, 0.000001, size=2)
    sigma = Uniform('sigma', 0, 1000, size=2)
    rho = Uniform('r', -1, 1)

    # we have a further parameter (prior) for the robust case
    if robust == True:
        nu = Exponential('nu',1/29., 1)
        # we model nu as an Exponential plus one
        @pymc.deterministic
        def nuplus(nu=nu):
            return nu + 1

    @pymc.deterministic
    def precision(sigma=sigma,rho=rho):
        ss1 = float(sigma[0] * sigma[0])
        ss2 = float(sigma[1] * sigma[1])
        rss = float(rho * sigma[0] * sigma[1])
        return inv(np.mat([[ss1, rss], [rss, ss2]]))

    if robust == True:
        # log-likelihood of multivariate t-distribution
        @pymc.stochastic(observed=True)
        def mult_t(value=data.T, mu=mu, tau=precision, nu=nuplus):
            k = float(tau.shape[0])
            res = 0
            for r in value:
                delta = r - mu
                enum1 = gammaln((nu+k)/2.) + 0.5 * log(det(tau))
                denom = (k/2.)*log(nu*pi) + gammaln(nu/2.)
                enum2 = (-(nu+k)/2.) * log (1 + (1/nu)*delta.dot(tau).dot(delta.T))
                result = enum1 + enum2 - denom
                res += result[0]
            return res[0,0]

    else:
        mult_n = MvNormal('mult_n', mu=mu, tau=precision, value=data.T, observed=True)

    return locals()

def correlation(x, y, n_iter=5000, burn_in=2500, robust=False):
    """
    Correlation

    Parameters
    ----------
    x : (N,) array_like
    y : (N,) array_like
    n_iter : int, optional
        Number of MCMC iterations
    burn_in : int, optional
        Number of burn in iterations
    robust : bool, optional
        If True, the robust correlation is determined

    Returns
    -------
    PyMC MCMC

    References
    ----------
    http://www.philippsinger.info/?p=581
    http://www.sumsar.net/blog/2013/08/bayesian-estimation-of-correlation/
    http://www.sumsar.net/blog/2013/08/robust-bayesian-estimation-of-correlation/
    http://bayesmodels.com/
    """

    x = np.asarray(x)
    y = np.asarray(y)

    mcmc = pymc.MCMC(_correlation_model(np.array([x, y]),robust))
    mcmc.sample(5000,2500)

    return mcmc

if __name__ == "__main__":

    x = np.array([525., 300., 450., 300., 400., 500., 550., 125., 300., 400., 500., 550.])
    y = np.array([250., 225., 275., 350., 325., 375., 450., 400., 500., 550., 600., 525.])
    mcmc = correlation(x,y,robust=False)

    print mcmc.stats()['r']['mean']
    print mcmc.stats()['r']['95% HPD interval']


    #pymc.Matplot.plot(mcmc)
    #pylab.show()