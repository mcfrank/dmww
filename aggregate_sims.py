import pickle
import os
import numpy as np
from dmww_classes import *

def find_best_sim(burn_samps):

    sims = {}
    for sim_id in os.listdir('simulations/'):

        f = open('simulations/'+sim_id+'/'+sim_id+'.data')
        lexicon = pickle.load(f)
        ref = np.mean(lexicon.refs[burn_samps:], axis=0)
        corpus_file = 'corpora/corpus.csv'
        world = World(corpus=corpus_file)
        corpus = Corpus(world=world, corpus=corpus_file)

        t, (p, r, f) = lexicon.get_max_f(ref, corpus)
        sims[sim_id] = (f, str(lexicon.params))
        print 'simulation', sim_id, 'score', f

    best_sim = max(sims, key = lambda t: sims[t][0])
    print 'best score:', sims[best_sim][0]
    print 'best params', sims[best_sim][1]

find_best_sim(0)