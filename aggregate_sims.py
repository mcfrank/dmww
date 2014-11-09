import csv
import pickle
import os
import numpy as np
from random import sample
from dmww_classes import *


def find_best_sim(direc, burn_samps):

    corpus_file = 'corpora/corpus.csv'
    world = World(corpus=corpus_file)
    corpus = Corpus(world=world, corpus=corpus_file)

    results_file = csv.writer(open('gibbs1000x.summary.csv', 'w'))
    results_file.writerow(['alpha_r', 'alpha_nr', 'empty_intent', 'p', 'r', 'f'])
    par_sims = {}

    for sim_id in filter(lambda d: d[0] != '.', os.listdir(direc)):

        print 'loading sim', sim_id
        df = open(direc + sim_id + '/' + sim_id + '.data')
        results = pickle.load(df)
        refs, non_refs, alg, params = results['refs'], results['non_refs'], results['alg'], results['params']
        df.close()

        if len(refs) != 1000:
            print 'non-1000 sim', sim_id
            continue

        if alg == 'gibbs' and len(refs) > burn_samps:
            ref = np.mean(refs[burn_samps:], axis=0)
        else:
            ref = np.mean(refs, axis=0)

        par = (params.alpha_r, params.alpha_nr, params.empty_intent)
        if par in par_sims:
            par_sims[par][sim_id] = ref
        else:
            par_sims[par] = {sim_id: ref}

    avgs = {}

    for par, sims in par_sims.iteritems():

        par_avg = np.mean(sims.values(), axis=0)
        t, (p, r, f) = Lexicon.get_max_f(par_avg, corpus)
        avgs[par] = (p, r, f)
        results_file.writerow(list(par) + [p, r, f])
        print par, p, r, f

    best_sim = max(avgs, key = lambda s: avgs[s][2])
    print 'best sim:', best_sim
    print 'best score:', avgs[best_sim]


def make_multi_fscore_plot(direc):

    corpus_file = 'corpora/corpus.csv'
    world = World(corpus=corpus_file)
    corpus = Corpus(world=world, corpus=corpus_file)

    par_sims = {}

    for sim_id in filter(lambda d: d[0] != '.', os.listdir(direc)):

        df = open(direc + sim_id + '/' + sim_id + '.data')
        results = pickle.load(df)
        refs, non_refs, alg, params = results['refs'], results['non_refs'], results['alg'], results['params']
        df.close()

        fscores = [Lexicon.get_max_f(refs[s], corpus)[1][2] for s in xrange(len(refs))]
        par = (params.alpha_r, params.alpha_nr, params.empty_intent)
        if par in par_sims:
            par_sims[par].append(fscores)
        else:
            par_sims[par] = [fscores]

    for par, sims in par_sims.iteritems():
#        fig, ax = plt.subplots()
        par_writer = csv.writer(open('fscores/' + ','.join([str(p) for p in par]) + '.csv', 'w'))
        par_writer.writerow(range(len(fscores)))
        for fscores in sims:
            par_writer.writerow(fscores)
#            ax.plot(np.arange(len(fscores)), fscores, '-')
#        plt.xlabel('Sample')
#        plt.ylabel('Sample f-score')
#        plt.title('Lexicon f-score over time with params' + str(par))
#        plt.savefig('simulations/' + ':'.join([str(p) for p in par]) + '.png')
#        plt.close()

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
    results = pickle.load(f)
    refs, non_refs, alg, params = results['refs'], results['non_refs'], results['alg'], results['params']
    f.close()
    print 'loaded results'
#    if any([s > len(lex.particles) for s in sample_sizes]):
#        raise ValueError, "sample sizes must be smaller than number of particles"
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
#    averages = {s: mean([mean([Lexicon.get_max_f(ref, corpus)[1][2] for ref in sample(refs, s)])
    averages = {s: mean([Lexicon.get_max_f(mean(sample(refs, s), axis=0), corpus)[1][2]
                         for n in xrange(num_samples)])
                for s in sample_sizes}
#    averages = {s: mean([mean([lex.get_max_f(p.ref, corpus)[0] for p in sample(lex.particles, s)])
#                    for n in xrange(num_samples)])
#                for s in sample_sizes}
    print 'computed averages'
    print averages

    sizes = sort(averages.keys())
    scores = [averages[s] for s in sizes]
    fig, ax = plt.subplots()
    ax.plot(sizes, scores, '-')
    plt.xlabel('number of particles')
    plt.ylabel('average f-score')
    plt.title('Lexicon f-score for different numbers of particles')
    plt.savefig(plot_filename+'.by_num_particles.png')
    plt.close()

#    fscores = sort([Lexicon.get_max_f(ref, corpus)[1][2] for ref in refs])
#    print fscores
#    fig, ax = plt.subplots()
#    ax.plot(range(len(refs)), fscores, '.')
#    plt.xlabel('particle')
#    plt.ylabel('f-score')
#    plt.title('Lexicon f-score for each particle')
#    plt.savefig(plot_filename+'.by_particle.png')


def find_right_sims(output_dir):

    ar, an, ei = '1.0', '1.0', '0.01'
    right_sims = []
    for out in os.listdir(output_dir):
        out_file = open(output_dir + out, 'r')
        out_lines = out_file.read().split('\n')
        out_lines.reverse()
        for line in out_lines:
            if ' ' in line:
                h, s = line.split(' ', 1)
                if h == 'sim':
                    sim_id = s.split(' ')[1]
                if h == 'params':
                    par = s[s.find('alpha-r')+11: s.find('alpha-nr')-8]
                    pan = s[s.find('alpha-nr')+12: s.find('empty-intent')-8]
                    pei = s[s.find('empty-intent')+16: s.find(']')-2]
                    if par==ar and pan==an and pei==ei:
                        right_sims.append(sim_id)
    return right_sims


def find_best_lexicon(output_dir, results_dir, burn_samps):

    right_sims = find_right_sims(output_dir)
    sim_refs = []
    for sim_id in right_sims:

        df = open(results_dir + sim_id + '/' + sim_id + '.data')
        results = pickle.load(df)
        refs, non_refs, alg, params = results['refs'], results['non_refs'], results['alg'], results['params']
        df.close()
        ref = np.mean(refs[burn_samps:], axis=0)        
        sim_refs.append(ref)

    corpus_file = 'corpora/corpus.csv'
    world = World(corpus=corpus_file)
    corpus = Corpus(world=world, corpus=corpus_file)

    avg_ref = np.mean(sim_refs, axis=0)
    t, (p, r, f) = Lexicon.get_max_f(avg_ref, corpus)
    print p, r, f
    for obj in range(world.n_objs):
        row_sums = avg_ref.sum(axis=1)
        links = where(avg_ref[obj, :] / row_sums[obj] > t)
        for word in links[0]:
            print 'o: %s, w: %s' % (world.objs_dict[obj][0], world.words_dict[word][0])


def compute_sample_fscores(output_dir, results_dir, burn_samps):

    right_sims = find_right_sims(output_dir)

    corpus_file = 'corpora/corpus.csv'
    world = World(corpus=corpus_file)
    corpus = Corpus(world=world, corpus=corpus_file)

    sim_scores = {}

    for sim_id in right_sims:

        df = open(results_dir + sim_id + '/' + sim_id + '.data')
        results = pickle.load(df)
        refs, non_refs, alg, params = results['refs'], results['non_refs'], results['alg'], results['params']
        fscores = [Lexicon.get_max_f(ref, corpus)[1][2] for ref in refs[burn_samps:]]
        sim_scores[sim_id] = np.mean(fscores)
        df.close()

    print sim_scores

#find_best_sim(os.path.expanduser('~/Documents/sims/gibbs.1000/'), 600)
#find_best_sim('/farmshare/user_data/mikabr/dmww_sims/gibbs1000x/results/', 300)
#find_best_sim('/farmshare/user_data/mikabr/dmww_sims/testgibbs/', 300)
#make_multi_fscore_plot('/farmshare/user_data/mikabr/dmww_sims/gibbs1000/results/')
#make_multi_fscore_plot('/farmshare/user_data/mikabr/dmww_sims/gibbs1000x/results/')
#make_multi_fscore_plot('/farmshare/user_data/mikabr/dmww_sims/testgibbs/')

#find_best_lexicon('/farmshare/user_data/mikabr/dmww_sims/gibbs1000/output/',
#                  '/farmshare/user_data/mikabr/dmww_sims/gibbs1000/results/', 300)

#compute_sample_fscores('/farmshare/user_data/mikabr/dmww_sims/gibbs1000/output/',
#                       '/farmshare/user_data/mikabr/dmww_sims/gibbs1000/results/', 300)

#sid ='cbe5a7df-c707-4da1-8887-0772af14bc34'
#make_fscore_plot(sid+'.data', sid+'.fscores.png')
sid = '03c2b37c-aa87-4d8e-905a-b94eec83d505'
path = '/farmshare/user_data/mikabr/dmww_sims/pf1000/results/' + sid + '/' + sid + '.data'
#make_particle_plot(path, '/farmshare/user_data/mikabr/dmww_sims/pf1000/plots/'+sid+'.fscores.png', [1,2,5,10], 5)
#make_particle_plot(path, '/farmshare/user_data/mikabr/dmww_sims/pf1000/plots/'+sid+'.fscores', [1,2,5,10,20,50,100], 10)
make_particle_plot(path, '/farmshare/user_data/mikabr/dmww_sims/pf1000/plots/'+sid+'.fscores', [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000], 100)
