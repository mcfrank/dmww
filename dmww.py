from dmww_classes import *
from plotting_helper import *

################################################################
# The model
################################################################

seed(1)

w = World(n_words=8,
           n_objs=8)
w.show()

c = Corpus(w,
           n_per_sent=2,
           n_sents=40)
c.show()

# print "*** coocurrence test ***"
# l = CoocLexicon(w)
# l.learn_lex(c)
# l.show()

# w 8, 8, c 1, 40, with a hundred samples and no hyperinf is around .18s / sample
# in the worst case
print "*** gibbs test ***"
p = Params(n_samps=100,
           alpha_r=.1,
           alpha_nr=10,
           empty_intent=.0001,
           n_hypermoves=5)
p.show()


l = GibbsLexicon(c,p,
                 verbose=0,
                 hyper_inf=True)

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
corpusfile = 'corpora/corpus.csv'
w = World(corpus=corpusfile)
w.show()

c = Corpus(world=w, corpus=corpusfile)

p = Params(n_samps=100,
           alpha_r=.1,
           alpha_nr=10,
           empty_intent=.0001,
           n_hypermoves=10)

l = GibbsLexicon(c, p,
                 verbose=0,
                 hyper_inf=True)

l.learn_lex(c,p)

l.show()
l.params.show()
l.show_top_match(c,w)

lexplot(l,w)
pylab.show(block=True)


# pr.disable()
# s = StringIO.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print s.getvalue()
