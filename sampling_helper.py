from numpy import *
from random import *
from scipy.special import gammaln
from scipy import stats

#########
## for variables that can't go below zero

def alter_0inf_var(ob):
    nb = ob

    if random() > .5:
        nb = ob + exp(gauss(0, 2))
    else:
        nb = ob - exp(gauss(0, 2))

    # can't go below zero
    if nb < 0:
        nb = ob

    return nb


def alter_01_var(ob):
    nb = ob

    nb = ob + inv_logit(gauss(0, 10))

    # can't go below zero or above one
    if nb <= 0 or nb >= 1:
        nb = ob

    return nb


def inv_logit(x):
    y = exp(x) / (1 + exp(x))

    return y

########################################################################
## FUNCTIONS FOR DIRICHLET-MULTINOMIAL UPDATES
## matlab equivalents above

# % dirichlet multinomial conjugate posterior
# % note, is this right?
# function ls = scoreDM(c,alpha)

#   k = length(c);
#   n = sum(c);
#   ls = sum(gammaln(c + alpha/k)) - (gammaln((alpha/k))*k) ...
#     + gammaln(alpha) - gammaln(n + alpha);
# end

def score_dm(c, alpha):
    k = len(c)
    n = sum(c)
    ls = sum(gammaln(add(c, alpha / k))) - gammaln((alpha / k)) * k + gammaln(alpha) - gammaln(n + alpha)

    return ls


# % DM conjugate prior update based on changing count by -1
# function new_score = updateDMminus(old_score,c,a,i)
#   k = length(c);
#   n = sum(c);
#   new_score = old_score - log(c(i) + a/k) + log(n + a);
# end

def update_dm_minus(old_score, c, a, i):
    k = len(c)
    n = sum(c)
    new_score = old_score - log(c[i] + a / k) + log(n + a)
    return new_score


# % DM conjugate prior update based on changing count by 1
# function new_score = updateDMplus(old_score,c,a,i)
#   k = length(c);
#   n = sum(c);
#   new_score = old_score - log(n + a - 1) + log(c(i) + a/k - 1);
# end

def update_dm_plus(old_score, c, a, i):
    k = len(c)
    n = sum(c)
    new_score = old_score - log(n + a - 1) + log(c[i] + a / k - 1)
    return new_score

########################################################################
## MISC

def nans(desired_shape, dtype=float):
    a = empty(desired_shape, dtype)
    a.fill(nan)
    return a

def neg_infs(desired_shape, dtype=float):
    a = empty(desired_shape, dtype)
    a.fill(-Inf)
    return a