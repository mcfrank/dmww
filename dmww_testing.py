import numpy as np
from random import *
from dmww_classes import *
from sampling_helper import *


def corpus_simulation(inference_algorithm, params):

    corpusfile = 'corpora/corpus.csv'
    w = World(corpus=corpusfile)
    #w.show()

    c = Corpus(world=w, corpus=corpusfile)

    l = Lexicon(c, params,
                verbose=0,
                hyper_inf=True)

    if inference_algorithm == 'gibbs':
        l.learn_lex_gibbs(c,params)
    elif inference_algorithm == 'pf':
        l.learn_lex_pf(c,params,resample=True)
    else:
        print "invalid inference algorithm"
        return

    gs_file = 'corpora/gold_standard.csv'
    c_gs = Corpus(world = w, corpus = gs_file)

    return get_f(l.ref, c_gs)

params = Params(n_samps=100,
                alpha_r=.1,
                alpha_nr=10,
                empty_intent=.0001,
                n_hypermoves=5)

print corpus_simulation('gibbs', params)
