from dmww_classes import *
# from plotting_helper import *

################################################################
# The model
################################################################

seed(1)

w = World(n_words=3,
          n_objs=3)
w.show()

# AB 1
# AC 2
# ABC 3

c = Corpus(w,
           n_per_sent=3,
           n_sents=3)
c.show()
print c.sents
c.sents = [[array([0,1]),array([0])],[array([1,2]),array([2])],[array([0,1,2]),array([1])]]
c.show()
print c.sents


p = Params(n_samps=1,
           n_particles=1,
           alpha_r=.1,
           alpha_nr=10,
           empty_intent=.0001,
           n_hypermoves=10)
p.show()

print "\n\n****************************************** GIBBS SAMPLER ******"
seed(1)
l = Lexicon(c, p,
            verbose=2,
            hyper_inf=True)

l.learn_lex_gibbs(c, p)
l.params.show()

print "\n\n****************************************** PARTICLE FILTER ******"
seed(1)
l = Lexicon(c, p,
            verbose=0,
            hyper_inf=False)

l.learn_lex_pf(c, p, resample=False)
l.output_lex_pf(c, p)


## PROFILE CODE
# import cProfile, pstats, StringIO
# pr = cProfile.Profile()
# pr.enable()
#
# l.learn_lex(c,p)
# # get_f(l.ref,c)
#
# pr.disable()
# s = StringIO.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print s.getvalue()
#

## CORPUS SIMS  ####
# seed(1)
# corpusfile = 'corpora/corpus.csv'
# w = World(corpus=corpusfile)
# w.show()
#
# c = Corpus(world=w, corpus=corpusfile)
#
# p = Params(n_samps=100,
#            n_particles=10,
#            alpha_r=.1,
#            alpha_nr=10,
#            empty_intent=.0001,
#            n_hypermoves=10)
#
# l = Lexicon(c, p,
#             verbose=0,
#             hyper_inf=True)
#
# l.learn_lex_pf(c, p)
#
# l.show()
# l.params.show()
# l.show_top_match(c,w)
#
#

# pr.disable()
# s = StringIO.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print s.getvalue()