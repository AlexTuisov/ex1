import Preprocessing
import Search
import featureMaker
import numpy as np
import datetime as d
import GradientAscent as g
import random

if __name__ == '__main__':
    tags = Preprocessing.preprocessing()
    unigrams, w = Preprocessing.get_ngrams(1)
    bigrams, w = Preprocessing.get_ngrams(2)
    trigrams, words = Preprocessing.get_ngrams(3)
    fmaker = featureMaker.feature_maker("<<>>",2)
    fmaker.init_all_params(unigrams,bigrams,trigrams)
    hist = Preprocessing.histogram_of_ngrams(3)
    gascent = g.gradient_ascent(fmaker.number_of_dimensions,10,fmaker,hist)
    hope = gascent.gradient_ascent()
    pure_little_test, little_test = Preprocessing.create_little_test()
    my_little_viterbi = Search.Searcher(tags, gascent, hope)
    my_little_viterbi.viterbi_full_run(Preprocessing.get_pure_test_set() , Preprocessing.get_path_to_test_set())


