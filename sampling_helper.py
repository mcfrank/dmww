from numpy import *
from random import *
from scipy.special import gammaln

# function [i j s] = chooseClass(scores)

# new_scores = scores - max(max(scores)); % add constant to make calculation of ratios possible
# ps = exp(new_scores); % calculate relative probabilities
# ps = ps / sum(sum(ps)); % normalize to 1
# cumPs = reshape(cumsum(reshape(ps,1,numel(ps))),size(ps));
# [i j] = find(rand<cumPs,1,'first');
# s = scores(i,j);

def chooseClass(scores):
    new_scores = scores - scores.max()
    ps = exp(new_scores)
    ps = ps / sum(sum(ps))
    cum_ps = reshape(cumsum(ps),shape(ps))

    r = random()
    i = where(cum_ps > r)[0][0]
    j = where(cum_ps > r)[1][0]
    s = scores[i,j]
    
    # return tuple of indices for greater than 
    return (i, j, s)

# function [i j s] = chooseClassTemp(scores,temp)
# new_scores = scores - max(max(scores)); % add constant to make calculation of ratios possible
# ps = exp(new_scores); % calculate relative probabilities
# ps = ps .^ (1/temp);
# ps = ps / sum(sum(ps)); % normalize to 1
# cumPs = reshape(cumsum(reshape(ps,1,numel(ps))),size(ps));
# [i j] = find(rand<cumPs,1,'first');
# s = scores(i,j);


def chooseClassTemp(scores, temp): ## unfinished
    new_scores = scores - max(max(scores))
    ps = exp(new_scores)
    ps = ps ^ (1/temp)
    ps = ps / sum(sum(ps))
    None

#########
## proposeHyperparams
def proposeHyperparams(self,
                 params):

    params.alpha_nr = alterUnnoundedVar(params.alpha_nr)
    params.alpha_r = alterUnboundedVar(params.alpha_r)
    params.empty_intent = alterBoundedVar(params.empty_intent)
    params.no_ref_word = alterBoundedVar(params.no_ref_word)

#########
## for variables that can't go below zero
def alterUnboundedVar(ob):
    nb = ob
  
    if random() > .5:
        nb = ob + exp(gauss(0,2))
    else:
        nb = ob - exp(gauss(0,2))
  
    # can't go below zero
    if nb < 0:
        nb = ob

    return nb

def alterBoundedVar(ob):
    nb = ob
  
    if random() > .5:
        nb = ob + gauss(0,.1)
    else:
        nb = ob - gauss(0,.1)
  
    # can't go below zero or above one
    if nb <= 0 or nb >= 1:
         nb = ob


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

def scoreDM(c, alpha):
    k = len(c)
    n = sum(c)
    ls = sum(gammaln(add(c,alpha/k))) - gammaln((alpha/k))*k + gammaln(alpha) - gammaln(n+alpha)

    return ls

# % DM conjugate prior update based on changing count by -1
# function new_score = updateDMminus(old_score,c,a,i)
#   k = length(c);
#   n = sum(c);
#   new_score = old_score - log(c(i) + a/k) + log(n + a);
# end

def updateDMminus(old_score, c, a, i):
    k = len(c)
    n = sum(c)
    new_score = old_score - log(c[i] + a/k) + log(n + a)
    return new_score

    
# % DM conjugate prior update based on changing count by 1
# function new_score = updateDMplus(old_score,c,a,i)
#   k = length(c);
#   n = sum(c);
#   new_score = old_score - log(n + a - 1) + log(c(i) + a/k - 1);
# end

def updateDMplus(old_score, c, a, i):
    k = len(c)
    n = sum(c)
    new_score = old_score - log(n + a - 1) + log(c[i] + a/k - 1)
    return new_score


########################################################################
## MISC


def nans(shape, dtype=float):
    a = empty(shape, dtype)
    a.fill(nan)
    return a
