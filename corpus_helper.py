def find(l, elem):
    for row, i in enumerate(l):
        try:
            column = i.index(elem)
        except ValueError:
            continue
        return row, column
    return -1



def get_f(lex, gs_corpus):
    gs = squeeze(asarray(gs_corpus.sents))
    gs_num_mappings = shape(gs)[0]

    links = nonzero(lex)
    obj_is =  links[0]
    word_is = links[1]
    lex_num_mappings = size(obj_is)

    # compute precision, what portion of the target lex is composed of gold pairings
    p_count = 0
    for pair in range(lex_num_mappings):
        this_obj = obj_is[pair]
        this_word = word_is[pair]

        #loop over gold standard
        if size(where((gs[:,0] == this_obj) & (gs[:,1] == this_word))) > 0:
            p_count = p_count + 1

    if (lex_num_mappings == 0): #special case
      precision = 0
    else:
      precision = float(p_count) / float(lex_num_mappings)

    # compute recall, how many of the total gold pairings are in the target lex
    recall = float(p_count) / float(gs_num_mappings)

    # now F is just the harmonic mean
    f =  stats.hmean([recall, precision])
    return f

