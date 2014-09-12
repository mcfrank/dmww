import numpy as np
from random import *
from dmww_classes import *
from sampling_helper import *


def corpus_simulation(inference_algorithm, params, threshold):

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
        l.learn_lex_pf(c,params,resample=False)
        l.output_lex_pf(c, params)
    else:
        print "invalid inference algorithm"
        return

    gs_file = 'corpora/gold_standard.csv'
    c_gs = Corpus(world = w, corpus = gs_file)

    return l.get_f(c_gs, threshold)

params = Params(n_samps=100,
                alpha_r=.1,
                alpha_nr=10,
                empty_intent=.0001,
                n_hypermoves=5,
                n_particles=10)

threshold = 0.1

print corpus_simulation('gibbs', params, threshold)
