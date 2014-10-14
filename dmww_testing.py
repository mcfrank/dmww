import getopt
import uuid
import pickle
import os
from dmww_classes import *


class Simulation:
    def __init__(self, corpus_file, inference_algorithm, lexicon_params):

        self.id = uuid.uuid4()
        self.alg = inference_algorithm
        self.world = World(corpus=corpus_file)
        self.corpus = Corpus(world=self.world, corpus=corpus_file)
        self.params = lexicon_params
        self.lexicon = Lexicon(self.corpus, self.params, verbose=0, hyper_inf=False)

        self.dir = 'simulations/results/' + str(self.id) + '/'
        os.mkdir(self.dir)
        self.filename = self.dir + str(self.id) + '.summary'
        self.data_file = open(self.dir + str(self.id) + '.data', 'a')

    def learn_lexicon(self):
        if self.alg == 'gibbs':
            self.lexicon.learn_lex_gibbs(self.corpus, self.params)
            pickle.dump(self.lexicon, self.data_file)
            ref = self.lexicon.ref
            nonref = self.lexicon.non_ref
            return ref, nonref
        elif self.alg == 'pf':
            self.lexicon.learn_lex_pf(self.corpus, self.params, resample=False)
            self.lexicon.output_lex_pf(self.corpus, self.params)
            pickle.dump(self.lexicon, self.data_file)
            return self.lexicon.ref, self.lexicon.non_ref

    def run(self):

        ref, nonref = self.learn_lexicon()
        threshold, (p, r, f) = self.lexicon.get_max_f(ref, self.corpus)

        sim_file = open(self.filename + '.txt', 'a')
        sim_file.write('Algorithm:' + self.alg)
        sim_file.write('\n\n')

        sim_file.write('Parameters:\n')
        sim_file.write(str(self.lexicon.params))
        sim_file.write('\n\n')

        sim_file.write('Lexicon:\n')
        for obj in range(self.world.n_objs):
            row_sums = ref.sum(axis=1)
            links = where(ref[obj, :] / row_sums[obj] > threshold)
            for word in links[0]:
                sim_file.write('o: %s, w: %s' % (self.world.objs_dict[obj][0], self.world.words_dict[word][0]))
                sim_file.write('\n')
        sim_file.write('\n\n')

        sim_file.write('Precision: %s\nRecall: %s\nF-score: %s\nThreshold: %s' % (p, r, f, threshold))
        sim_file.write('\n\n')

        sim_file.write(str([str(self.id), self.alg, self.params.n_samps, self.params.n_particles,
                            self.params.alpha_r, self.params.alpha_nr, self.params.empty_intent,
                            p, r, f, threshold]))

        sim_file.close()

        if self.alg == 'gibbs':
            self.lexicon.plot_scores()
            plt.savefig(self.filename + '_scores.png')
            self.lexicon.plot_fscores()
            plt.savefig(self.filename + '_fscores.png')
            plt.close()


def main(argv):
    inference_algorithm = None
    n = None
    alpha_r = None
    alpha_nr = None
    empty_intent = None

    usage = "usage: dmww_testing.py " \
            "-a <inference algorithm: gibbs or pf> " \
            "-n <number of samples/particles> " \
            "--alpha-r <referential alpha> " \
            "--alpha-nr <non-referential alpha> " \
            "--empty-intent <empty intent probability>"

    try:
        opts, args = getopt.getopt(argv, "ha:n:", ["alpha-r=", "alpha-nr=", "empty-intent="])
    except getopt.GetoptError:
        print usage
        sys.exit(2)
    print opts
    for opt, arg in opts:
        if opt == '-h':
            print usage
            sys.exit()
        elif opt == "-a":
            if arg != 'pf' and arg != 'gibbs':
                print "Invalid inference algorithm, must be gibbs or pf."
                sys.exit(2)
            inference_algorithm = arg
        elif opt == "-n":
            n = int(arg)
        elif opt == "-b":
            burn_samps = int(arg)
        elif opt == "--alpha-r":
            alpha_r = float(arg)
        elif opt == "--alpha-nr":
            alpha_nr = float(arg)
        elif opt == "--empty-intent":
            empty_intent = float(arg)

    params = Params(n_samps=n,
                    alpha_r=alpha_r,
                    alpha_nr=alpha_nr,
                    empty_intent=empty_intent,
                    n_particles=n)

    seed()
    sim = Simulation('corpora/corpus.csv', inference_algorithm, params)
    sim.run()


if __name__ == "__main__":
    main(sys.argv[1:])
