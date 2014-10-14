import csv
import pickle
import os
import numpy as np
from random import sample
from dmww_classes import *


def find_best_sim(direc, burn_samps):
    # summary_file = open('gibbs1000.summary.txt', 'w')
    #    summary_writer = csv.writer(summary_file)
    #    summary_writer.writerow(['alpha_r', 'alpha_nr', 'empty_intent', 'precision', 'recall', 'f_score'])
    #    sims = {}
    for sim_id in os.listdir(direc):

        f = open(direc + sim_id + '/' + sim_id + '.data')
        lexicon = pickle.load(f)
        if len(lexicon.refs) > burn_samps:
            ref = np.mean(lexicon.refs[burn_samps:], axis=0)
        else:
            ref = lexicon.ref
        corpus_file = 'corpora/corpus.csv'
        world = World(corpus=corpus_file)
        corpus = Corpus(world=world, corpus=corpus_file)

        t, (p, r, f) = lexicon.get_max_f(ref, corpus)
        #        sims[sim_id] = ((p,r,f), str(lexicon.params))
        print 'simulation', sim_id, 'score', f


# ar = lexicon.params.alpha_r
#        anr = lexicon.params.alpha_nr
#        ei = lexicon.params.empty_intent
#        summary_writer.writerow([str(ar), str(anr), str(ei), str(p), str(r), str(f)])

#    best_sim = max(sims, key = lambda s: sims[s][0][2])
#    print 'best sim:', best_sim
#    print 'best score:', sims[best_sim][0]
#    print 'best params', sims[best_sim][1]


def make_fscore_plot(path_to_file, plot_filename):
    corpus_file = 'corpora/corpus.csv'
    world = World(corpus=corpus_file)
    corpus = Corpus(world=world, corpus=corpus_file)
    f = open(path_to_file)
    lex = pickle.load(f)
    lex.sample_fscores = [lex.get_max_f(lex.refs[s], corpus)[1][2] for s in xrange(len(lex.refs))]
    lex.plot_fscores()
    plt.savefig(plot_filename)
    plt.close()
    f.close()


def make_particle_plot(path_to_file, plot_filename, sample_sizes, num_samples):
    corpus_file = 'corpora/corpus.csv'
    world = World(corpus=corpus_file)
    corpus = Corpus(world=world, corpus=corpus_file)
    f = open(path_to_file)
    lex = pickle.load(f)
    if any([s > len(lex.particles) for s in sample_sizes]):
        raise ValueError, "sample sizes must be smaller than number of particles"
    averages = {}
    for s in sample_sizes:
        samples = []
        for n in xrange(num_samples):
            particles = sample(lex.particles, s)
            size_avg = mean([lex.get_max_f(p.ref, corpus)[1][2] for p in particles])
            samples.append(size_avg)
        averages[s] = mean(samples)
    fig, ax = plt.subplots()
    ax.plot(averages.keys(), averages.values(), '-')
    plt.xlabel('number of particles')
    plt.ylabel('average f-score')
    plt.title('Lexicon f-score for different numbers of particles')
    plt.savefig(plot_filename)
    plt.close()
    f.close()
    print averages

#find_best_sim('gibbs1000/', 300)

#sid ='cbe5a7df-c707-4da1-8887-0772af14bc34'
#make_fscore_plot(sid+'.data', sid+'.fscores.png')
sid = '20d2f363-1552-4b2c-b576-f9589ef892f7'
path = 'simulations/results/' + sid + '/' + sid + '.data'
make_particle_plot(path, sid+'.fscores.png', xrange(10), 10)