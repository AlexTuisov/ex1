import Preprocessing
import random
if __name__ == '__main__':
    #tags = Preprocessing.preprocessing()
    unigrams, words = Preprocessing.get_ngrams(1)
    bigrams, words = Preprocessing.get_ngrams(2)
    trigrams, words = Preprocessing.get_ngrams(3)
    print(words[22])
    print(trigrams[22])
#