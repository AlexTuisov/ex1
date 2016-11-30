import featureMaker
import Preprocessing
import datetime as d
import  GradientAscent as g
if __name__ == '__main__':
    """unigrams = Preprocessing.get_ngrams(1)
    bigrams = Preprocessing.get_ngrams(2)
    trigrams = Preprocessing.get_ngrams(3)
    index_number, paramsIndex=featureMaker.getFeatureParamsFromIndex(unigrams,bigrams,trigrams)
    print(index_number)"""
    x = g.gradient_ascent(100000,10,())
    a=x.vector_v_init()
    b=x.vector_v_init()
    start = d.datetime.now()
    x.vector_multiplication(a,b)
    print("it took :",d.datetime.now() - start)