import Preprocessing
import Search
import featureMaker
import numpy as np
import datetime as d
import GradientAscent as g
import random

if __name__ == '__main__':
    start = d.datetime.now()
    tags, tags_count, inverse_tag_count = Preprocessing.preprocessing()
    values = sorted(tags_count.values())
    values.reverse()
    for value in values:
        print(value, inverse_tag_count[value])
        if value < 200:
            tags.discard(inverse_tag_count[value])
    little_test, pure_little_test = Preprocessing.create_little_test(0.1)
    print(little_test)
    print(pure_little_test)
    unigrams, w = Preprocessing.get_ngrams(1)
    bigrams, w = Preprocessing.get_ngrams(2)
    trigrams, words = Preprocessing.get_ngrams(3)
    fmaker = featureMaker.feature_maker("<<>>",5)
    fmaker.init_all_params(unigrams,bigrams,trigrams)
    hist = Preprocessing.histogram_of_ngrams(3)
    gascent = g.gradient_ascent(fmaker.number_of_dimensions,10,fmaker,hist)
    hope = gascent.gradient_ascent()

    my_little_viterbi = Search.Searcher(tags, gascent, hope[0])
    my_little_viterbi.viterbi_full_run(little_test, pure_little_test)
    print("The time whole program took:", d.datetime.now()-start)


