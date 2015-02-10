
from dmww_classes import *
# from plotting_helper import *

################################################################
# The model
################################################################

seed(1)

w = World(n_words=2,
          n_objs=2)
w.show()

#c = Corpus(w,
#           n_per_sent=1,
#           n_sents=10)
c = Corpus(w,
           n_per_sent=0,
           n_sents=0)
c.sents.append([array([0,1]), array([0])])
c.sents.append([array([1]), array([1])])
c.update()
c.show()

p = Params(n_samps=100,
           n_particles=100,
           alpha_r=.1,
           alpha_nr=10,
           empty_intent=.0001,
           n_hypermoves=10)
p.show()

print "\n\n****************************************** GIBBS SAMPLER ******"
seed(1)
l = Lexicon(c, p,
            verbose=0,
            hyper_inf=True)

l.learn_lex_gibbs(c, p)
l.params.show()
#for s in xrange(l.params.n_samps):
#    print "sample", s
#    print l.refs[s]
#    print l.nonrefs[s]

print "\naverage"
print np.around(np.mean(l.refs, axis=0), decimals=2)
print np.around(np.mean(l.nonrefs, axis=0), decimals=2)

print "\n\n****************************************** PARTICLE FILTER ******"
seed(1)
l = Lexicon(c, p,
            verbose=0,
            hyper_inf=False)

l.learn_lex_pf(c, p, resample=False)
l.output_lex_pf(c, p)
#for p in xrange(l.params.n_particles):
#    print "particle", p
#    print l.refs[p]
#    print l.nonrefs[p]

print "\naverage"
l.show()

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