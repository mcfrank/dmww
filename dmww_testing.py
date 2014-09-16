import sys, getopt
from dmww_classes import *


def learn_lexicon(world, corpus_file, inference_algorithm, params):

    corpus = Corpus(world=world, corpus=corpus_file)

    lexicon = Lexicon(corpus, params,
                      verbose=0,
                      hyper_inf=False)
                      #hyper_inf=True)

    if inference_algorithm == 'gibbs':
        lexicon.learn_lex_gibbs(corpus, params)
    elif inference_algorithm == 'pf':
        lexicon.learn_lex_pf(corpus, params, resample=False)
        lexicon.output_lex_pf(corpus, params)
    else:
        print "invalid inference algorithm"
        return

    return lexicon


def evaluate(world, lexicon, threshold):

    gs_file = 'corpora/gold_standard.csv'
    c_gs = Corpus(world=world, corpus=gs_file)

    return lexicon.get_f(c_gs, threshold)


def maximize_score(world, lexicon):

    scores = {}
    threshold_opts = [float(t)/100 for t in xrange(101)]
    for t in threshold_opts:
        score = evaluate(world, lexicon, t)
        if score:
            scores[t] = score
    best_threshold = max(scores, key=lambda t: scores[t][2])
    return best_threshold, scores[best_threshold]


def corpus_simulation(corpus_file, inference_algorithm, params):

    world = World(corpus=corpus_file)
    lexicon = learn_lexicon(world, corpus_file, inference_algorithm, params)

#    for obj in range(world.n_objs):
#        wd = where(lexicon.ref[obj,:] == max(lexicon.ref[obj,:]))
#        print "o: %s, w: %s" % (world.objs_dict[obj][0], world.words_dict[wd[0][0]][0])

    return maximize_score(world, lexicon)


def main(argv):
    inference_algorithm = 'pf'
    n_samps = 10
    n_particles = 10
    alpha_r = 0.1
    alpha_nr = 10
    empty_intent = 0.0001

    usage = "usage: dmww_testing.py -a <inference algorithm: gibbs or pf> -n <number of samples/particles>" \
            "--alphaR <referential alpha> --alphaNR <non-referential alpha> --empty-intent <empty intent probability>"

    try:
        opts, args = getopt.getopt(argv, "ha:n:", ["alphaR=", "alphaNR=", "empty-intent="])
    except getopt.GetoptError:
        print usage
        sys.exit(2)
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
            n_samps = arg
            n_particles = arg
        elif opt == "--alphaR":
            alpha_r = arg
        elif opt == "--alphaNR":
            alpha_nr = arg
        elif opt == "--empty_intent":
            empty_intent = arg
    print "Inference algorithm:", inference_algorithm
    print "Number of samples/particles:", n_samps
    print "Referential alpha:", alpha_r
    print "Non-referential alpha:", alpha_nr
    print "Empty intent probability:", empty_intent
    print "--------------------------------------------------"

    params = Params(n_samps=n_samps,
                    alpha_r=alpha_r,
                    alpha_nr=alpha_nr,
                    empty_intent=empty_intent,
                    #n_hypermoves=5,
                    n_particles=n_particles)

    t, results = corpus_simulation('corpora/corpus.csv', inference_algorithm, params)
    p, r, f = results
    print "Threshold:", t
    print "Precision:", p
    print "Recall:", r
    print "F-score:", f

if __name__ == "__main__":
    main(sys.argv[1:])