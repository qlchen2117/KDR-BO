import numpy as np
from scipy.stats import norm


def log_expected_improvement(f_min: float, mu: np.ndarray, sigma: np.ndarray):
    """Expected improvement often yields zero. 
    Use more robust log expected improvement instead.
    """
    x = (f_min - mu) / sigma
    log_ei = np.zeros_like(mu)
    for i in range(len(mu)):
        if np.abs(f_min-mu[i]) == 0:
            # Degenerate case 1: first term vanishes.
            if sigma[i] > 0:
                log_ei[i] = np.log(sigma[i]) + norm.logpdf(x[i])
            else:
                log_ei[i] = -np.inf
        elif sigma[i] == 0:
            # Degenerate case 2: second term vanishes and first term has a special form.
            if mu[i] < f_min:
                log_ei[i] = np.log(f_min-mu[i])
            else:
                log_ei[i] = -np.inf
        else:
            # Normal case.
            b = np.log(sigma[i]) + norm.logpdf(x[i])
            # log(y+z) is tricky, we distinguish two cases:
            if f_min > mu[i]:
                # When y>0, z>0, we define a=ln(y), b=ln(z).
                # Then y+z = exp[ max(a,b) + ln(1 + exp(-|b-a|)) ],
                # and thus log(y+z) = max(a,b) + ln(1 + exp(-|b-a|))
                a = np.log(f_min-mu[i]) + norm.logcdf(x[i])
                log_ei[i] = np.maximum(a,b) + np.log(1 + np.exp(-np.abs(b-a)))
            else:
                # When y<0, z>0, we define a=ln(-y), b=ln(z), 
                # and it has to be true that b >= a in order to satisfy y+z>=0.
                # Then y+z = exp[ a + ln(exp(b-a) -1) ],
                # and thus log(y+z) = a + ln(exp(b-a) -1)
                a = np.log(mu[i]-f_min) + norm.logcdf(x[i])
                if a >= b:
                # a>b can only happen due to numerical inaccuracies or 
                # approximation errors
                    log_ei[i] = -np.inf
                else:
                    log_ei[i] = b + np.log(1-np.exp(a-b))
    return log_ei

if __name__ == '__main__':
    f_min = 1
    # Degenerate case 1: first term vanishes.
    mu = np.ones(2)
    sigma = np.array([0.1, 0.])
    print(log_expected_improvement(f_min, mu, sigma))
    # Degenerate case 2: second term vanishes and first term has a special form.
    mu = np.array([0., 2.])
    sigma = np.zeros(2)
    print(log_expected_improvement(f_min, mu, sigma))
    # Normal case.
    mu = np.array([0., 2.])
    sigma = np.array([0.1, 0.2])
    print(log_expected_improvement(f_min, mu, sigma))