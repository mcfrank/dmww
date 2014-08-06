import numpy as np
from random import *
from scipy.special import gammaln
from scipy import stats

########################################################################
## PROPOSAL DISTRIBUTIONS FOR HYPERPARAMETER INFERENCE

# proposal function for variables that can't go below zero
def alter_0inf_var(ob):
    nb = ob

    if random() > .5:
        nb = ob + np.exp(gauss(0, 2))
    else:
        nb = ob - np.exp(gauss(0, 2))

    # can't go below zero
    if nb < 0:
        nb = ob

    return nb

# proposal function for hyperparameter variables that are 0-1 bounded
def alter_01_var(ob):
    nb = ob

    nb = ob + inv_logit(gauss(-6, 2))

    # can't go below zero or above one
    if nb <= 0 or nb >= 1:
        nb = ob

    return nb

# inverse logistic function
def inv_logit(x):
    y = np.exp(x) / (1 + np.exp(x))

    return y

########################################################################
## FUNCTIONS FOR DIRICHLET-MULTINOMIAL UPDATES

# dirichlet multinomial conjugate posterior
def score_dm(c, alpha):
    k = len(c)
    n = sum(c)
    ls = sum(gammaln(np.add(c, alpha / k))) - \
             gammaln((alpha / k)) * k + \
             gammaln(alpha) - \
             gammaln(n + alpha)

    return ls


# DM conjugate prior update based on changing count by -1
def update_dm_minus(old_score, c, a, i):
    k = len(c)
    n = sum(c)
    new_score = old_score - np.log(c[i] + a / k) + np.log(n + a)
    return new_score


# DM conjugate prior update based on changing count by 1
def update_dm_plus(old_score, c, a, i):
    k = len(c)
    n = sum(c)
    new_score = old_score - np.log(n + a - 1) + np.log(c[i] + a / k - 1)
    return new_score

########################################################################
## MISC

def nans(desired_shape, dtype=float):
    a = np.empty(desired_shape, dtype)
    a.fill(np.nan)
    return a

def neg_infs(desired_shape, dtype=float):
    a = np.empty(desired_shape, dtype)
    a.fill(-np.inf)
    return a