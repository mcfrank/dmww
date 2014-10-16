import csv
import pickle
import os
import numpy as np
from random import sample
from dmww_classes import *


def find_best_sim(direc, burn_samps):

    sim = {}

    for sim_id in os.listdir(direc):

        df = open(direc + sim_id + '/' + sim_id + '.data')
        lexicon = pickle.load(df)
        df.close()
        if lexicon.inference_method == 'gibbs' and len(lexicon.refs) > burn_samps:
            ref = np.mean(lexicon.refs[burn_samps:], axis=0)
        else:
            ref = lexicon.ref

        corpus_file = 'corpora/corpus.csv'
        world = World(corpus=corpus_file)
        corpus = Corpus(world=world, corpus=corpus_file)

        t, (p, r, f) = lexicon.get_max_f(ref, corpus)
        sims[sim_id] = ((p,r,f), str(lexicon.params))
        print 'simulation', sim_id, 'score', f

    best_sim = max(sims, key = lambda s: sims[s][0][2])
    print 'best sim:', best_sim
    print 'best score:', sims[best_sim][0]
    print 'best params', sims[best_sim][1]


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
    print 'loaded lex'
    if any([s > len(lex.particles) for s in sample_sizes]):
        raise ValueError, "sample sizes must be smaller than number of particles"
#    averages = {}
#    for s in sample_sizes:
#        samples = []
#        for n in xrange(num_samples):
#            particles = sample(lex.particles, s)
#            size_avg = mean([lex.get_max_f(p.ref, corpus)[1][2] for p in particles])
#            samples.append(size_avg)
#        samples = [mean([lex.get_max_f(p.ref, corpus)[1][2] for p in sample(lex.particles, s)])
#                   for n in xrange(num_samples)]
#        averages[s] = mean(samples)
    averages = {s: mean([mean([lex.get_max_f(p.ref, corpus)[0] for p in sample(lex.particles, s)])
                    for n in xrange(num_samples)])
                for s in sample_sizes}
    print 'computed averages'
    sizes = sort(averages.keys())
    scores = [averages[s] for s in sizes]
    fig, ax = plt.subplots()
    ax.plot(sizes, scores, '-')
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
#sid = 'db6a6173-c4e6-41c7-8bea-e53e53843061'
sid = '61cfa357-c9e8-4723-b213-028b1b606043'
path = 'simulations/results/' + sid + '/' + sid + '.data'
make_particle_plot(path, sid+'.fscores.png', [1,2,5,10,20,50,100], 10)