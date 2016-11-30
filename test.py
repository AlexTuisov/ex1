import featureMaker
import Preprocessing
if __name__ == '__main__':
    unigrams = Preprocessing.get_ngrams(1)
    bigrams = Preprocessing.get_ngrams(2)
    trigrams = Preprocessing.get_ngrams(3)
    index_number, paramsIndex=featureMaker.getFeatureParamsFromIndex(unigrams,bigrams,trigrams)
    print(index_number)