from dmww_classes import *
from plotting_helper import *

################################################################
# The model
################################################################

seed(1)

w = World(n_words=2,
           n_objs=2)
w.show()

c = Corpus(w,
           n_per_sent=1,
           n_sents=4)
c.show()

p = Params(n_samps=10,
           n_particles=1,
           alpha_r=.1,
           alpha_nr=10,
           empty_intent=.0001,
           n_hypermoves=5)
p.show()


l = Lexicon(c,p,
            verbose=2,
            hyper_inf=False)

l.learn_lex_pf(c,p)
l.show()

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




### CORPUS SIMS  ####
# corpusfile = 'corpora/corpus.csv'
# w = World(corpus=corpusfile)
# w.show()
#
# c = Corpus(world=w, corpus=corpusfile)
#
# p = Params(n_samps=100,
#            alpha_r=.1,
#            alpha_nr=10,
#            empty_intent=.0001,
#            n_hypermoves=10)
#
# l = Lexicon(c, p,
#             verbose=0,
#             hyper_inf=True)
#
# l.learn_lex_gibbs(c,p)
#
# l.show()
# l.params.show()
# l.show_top_match(c,w)
#
# lexplot(l,w)
# pylab.show(block=True)


# pr.disable()
# s = StringIO.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print s.getvalue()
