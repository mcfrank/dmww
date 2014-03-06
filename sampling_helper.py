from numpy import *
from scipy.special import gammaln

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


# function [i j s] = chooseClassTemp(scores,temp)
# new_scores = scores - max(max(scores)); % add constant to make calculation of ratios possible
# ps = exp(new_scores); % calculate relative probabilities
# ps = ps .^ (1/temp);
# ps = ps / sum(sum(ps)); % normalize to 1
# cumPs = reshape(cumsum(reshape(ps,1,numel(ps))),size(ps));
# [i j] = find(rand<cumPs,1,'first');
# s = scores(i,j);

# def chooseClassTemp(scores, temp):
#     new_scores = scores - max(max(scores))
#     ps = exp(new_scores)
#     ps = ps ^ (1/temp)
#     ps = ps / sum(sum(ps))
#     None

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
    


#     lex.ref_word_probs{i} = ... (i,1:length(corpus(i).words)+1) = ...
#       log([(1 - lex.no_ref_word)/length(corpus(i).words) * ones(size(corpus(i).words)) ...
#       lex.no_ref_word]);

#     % add count to the referential dirichlets
#     lex.ref(corpus(i).objects(lex.oi{i}==lex.intent_obj(i)),...
#             corpus(i).words(lex.wi{i}==lex.ref_word(i))) = ...
#       lex.ref(corpus(i).objects(lex.oi{i}==lex.intent_obj(i)),...
#               corpus(i).words(lex.wi{i}==lex.ref_word(i))) + 1; 

#     lex.non_ref(corpus(i).words(lex.wi{i}~=lex.ref_word(i))) = ...
#       lex.non_ref(corpus(i).words(lex.wi{i}~=lex.ref_word(i))) + 1;

#     lex.intent_obj_prob(i) = lex.intent_obj_probs{i}(lex.intent_obj(i));
#     lex.ref_word_prob(i) = lex.ref_word_probs{i}(lex.ref_word(i));
#   end

#   %% cache scores 

#   for o = 1:size(lex.ref,1)
#     lex.ref_score(o) = scoreDM(lex.ref(o,:),lex.alpha_r);
#   end

#   lex.nr_score = scoreDM(lex.non_ref,lex.alpha_nr);

# end

def nans(shape, dtype=float):
    a = numpy.empty(shape, dtype)
    a.fill(numpy.nan)
    return a

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
