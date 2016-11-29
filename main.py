import Preprocessing
import random
if __name__ == '__main__':
    unigrams = Preprocessing.get_ngrams(1)
    bigrams = Preprocessing.get_ngrams(2)
    trigrams = Preprocessing.get_ngrams(3)
    print(unigrams.popitem())
    print("\n")
    print(bigrams.popitem())
    print("\n")
    print(trigrams.popitem())