import csv
import pickle
import os
import numpy as np
from dmww_classes import *

def find_best_sim(dir, burn_samps):

    summary_file = open('gibbs200.summary.txt', 'w')
    summary_writer = csv.writer(summary_file)
    summary_writer.writerow(['alpha_r', 'alpha_nr', 'empty_intent', 'precision', 'recall', 'f_score'])
    sims = {}
    for sim_id in os.listdir(dir):

        f = open(dir+sim_id+'/'+sim_id+'.data')
        lexicon = pickle.load(f)
	if len(lexicon.refs) > burn_samps:
            ref = np.mean(lexicon.refs[burn_samps:], axis=0)
        else:
            ref = lexicon.ref
        corpus_file = 'corpora/corpus.csv'
        world = World(corpus=corpus_file)
        corpus = Corpus(world=world, corpus=corpus_file)

        t, (p, r, f) = lexicon.get_max_f(ref, corpus)
        sims[sim_id] = ((p,r,f), str(lexicon.params))
#        print 'simulation', sim_id, 'score', f

        ar = lexicon.params.alpha_r
        anr = lexicon.params.alpha_nr
        ei = lexicon.params.empty_intent
        summary_writer.writerow([str(ar), str(anr), str(ei), str(p), str(r), str(f)])

    best_sim = max(sims, key = lambda s: sims[s][0][2])
#    print 'best sim:', best_sim
#    print 'best score:', sims[best_sim][0]
#    print 'best params', sims[best_sim][1]

find_best_sim('simulations/gibbs200/results/', 100)
