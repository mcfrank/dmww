from numpy import *
from random import *
from scipy.special import gammaln
from scipy import stats




# function [i j s] = choose_class(scores)
# new_scores = scores - max(max(scores)); % add constant to make calculation of ratios possible
# ps = exp(new_scores); % calculate relative probabilities
# ps = ps / sum(sum(ps)); % normalize to 1
# cumPs = reshape(cumsum(reshape(ps,1,numel(ps))),size(ps));
# [i j] = find(rand<cumPs,1,'first');
# s = scores(i,j);





# function [i j s] = chooseClassTemp(scores,temp)
# new_scores = scores - max(max(scores)); % add constant to make calculation of ratios possible
# ps = exp(new_scores); % calculate relative probabilities
# ps = ps .^ (1/temp);
# ps = ps / sum(sum(ps)); % normalize to 1
# cumPs = reshape(cumsum(reshape(ps,1,numel(ps))),size(ps));
# [i j] = find(rand<cumPs,1,'first');
# s = scores(i,j);
#
#
# def choose_class_temp(scores, temp):  # unfinished
#     new_scores = scores - max(max(scores))
#     ps = exp(new_scores)
#     ps ^= 1 / temp
#     ps /= sum(sum(ps))
#     None





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


def get_f(lex, gs_corpus):
    gs = squeeze(asarray(gs_corpus.sents))
    gs_num_mappings = shape(gs)[0]

    links = nonzero(lex)
    obj_is =  links[0]
    word_is = links[1]
    lex_num_mappings = size(obj_is)

    # compute precision, what portion of the target lex is composed of gold pairings
    p_count = 0
    for pair in range(lex_num_mappings):
        this_obj = obj_is[pair]
        this_word = word_is[pair]

        #loop over gold standard
        if size(where((gs[:,0] == this_obj) & (gs[:,1] == this_word))) > 0:
            p_count = p_count + 1

    if (lex_num_mappings == 0): #special case
      precision = 0
    else:
      precision = float(p_count) / float(lex_num_mappings)

    # compute recall, how many of the total gold pairings are in the target lex
    recall = float(p_count) / float(gs_num_mappings)

    # now F is just the harmonic mean
    f =  stats.hmean([recall, precision])
    return f


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