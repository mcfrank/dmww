from numpy import *
from random import *
from dmww_classes import *
from plotting_helper import *

################################################################
# The model
################################################################

seed(1)
#
# w = World(n_words=8,
#            n_objs=8)
# w.show()
#
# c = Corpus(w,
#            n_per_sent=3,
#            n_sents=100)
# c.sample_sents()
# c.show()
#
# print "*** coocurrence test ***"
# l = CoocLexicon(w)
# l.learn_lex(c)
# l.show()
#
# print "*** gibbs test ***"
# p = Params(n_samps=1,
#            alpha_r=.1,
#            alpha_nr=10,
#            empty_intent=.0001,
#            no_ref_word=.000001,
#            n_hypermoves=5)
# p.show()
#
#
# l = GibbsLexicon(c,p,
#                  verbose=0,
#                  hyper_inf=True)
# l.learn_lex(c,p)
# l.show()


###
# w = World(corpus = 'corpora/corpus_toy.csv')
# w.show()
#
# c = Corpus(world=w, corpus = 'corpora/corpus_toy.csv')
# c.show()

# l = CoocLexicon(w)
# l.learn_lex(c)
# l.show()


# print len(c.sents[3][1])
# p = Params(n_samps=1,
#            alpha_r=.1,
#            alpha_nr=10,
#            empty_intent=.0001,
#            no_ref_word=.000001,
#            n_hypermoves=5)
#
# l = GibbsLexicon(c, p,
#                  verbose=1,
#                  hyper_inf=True)
# print l.ref
# l.learn_lex(c,p)
# l.show()
# l.params.show()

corpusfile = 'corpora/corpus.csv'
w = World(corpus=corpusfile)
w.show()

c = Corpus(world=w, corpus=corpusfile)

p = Params(n_samps=10,
           alpha_r=.1,
           alpha_nr=10,
           empty_intent=.0001,
           no_ref_word=.000001,
           n_hypermoves=10)

l = GibbsLexicon(c, p,
                 verbose=0,
                 hyper_inf=True)

l.learn_lex(c,p)
lexplot(l,w)
# l.params.show()
l.show()
l.params.show()
l.show_top_match(c,w)