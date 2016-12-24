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
    print(tags_count)
    values.reverse()
    Preprocessing.percentage_of_unknown_words()
    little_test, pure_little_test = Preprocessing.create_little_test(1)
    unigrams, w = Preprocessing.get_ngrams(1)
    bigrams, w = Preprocessing.get_ngrams(2)
    trigrams, words = Preprocessing.get_ngrams(3)
    fmaker = featureMaker.feature_maker("<<>>",5,True,1)
    fmaker.init_all_params(unigrams,bigrams,trigrams)
    hist = Preprocessing.histogram_of_ngrams(3)
    lambda_values =[1000]
    for value in lambda_values:
        gascent = g.gradient_ascent(fmaker.number_of_dimensions,value,fmaker,hist)
        hope = gascent.gradient_ascent()
        my_little_viterbi = Search.Searcher(tags, gascent, hope[0])
        a_a =my_little_viterbi.viterbi_full_run(little_test, pure_little_test)
        print("norm of v = ",np.dot(hope[0],hope[0]))
        print("for lambda = ",value," the average accuracy is = ",a_a)
    print("The time whole program took:", d.datetime.now()-start)


