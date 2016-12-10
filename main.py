import Preprocessing
import Search
import random
if __name__ == '__main__':
    # tags = Preprocessing.preprocessing()
    #unigrams, words = Preprocessing.get_ngrams(1)
    #bigrams, words = Preprocessing.get_ngrams(2)
    #strigrams, words = Preprocessing.get_ngrams(3)

    print(Preprocessing.histogram_of_ngrams(3))

"""
    tags_as_numbers = (0, 1, 2, 3, 4)
    tags = {"start": 0, "VB": 1, "NN": 2, "DT": 3, "@@@": 4}
    invert_tags = {number : tag for tag, number in tags.items()}
    sentence = ("the", "dog", "barks", "the")*5


    my_little_viterbi = Search.Searcher(tags, None)
    print (my_little_viterbi.tags_as_tuple)
    output_as_numbers = my_little_viterbi.viterbi_run_per_sentence(sentence)
    for index, output in enumerate(output_as_numbers):
        print(sentence[index], invert_tags[output])
"""

