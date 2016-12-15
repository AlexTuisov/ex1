import featureMaker
import Preprocessing
import datetime as d
import GradientAscent as g
import numpy as np
import matrixTest as mt

if __name__ == '__main__':
    str = "hopefully"
    if str.endswith("ly"):
        print("works")

    """unigrams ,w= Preprocessing.get_ngrams(1)
    bigrams ,w= Preprocessing.get_ngrams(2)
    trigrams, words = Preprocessing.get_ngrams(3)
    fmaker = featureMaker.feature_maker("<<>>",44)
    fmaker.init_all_params(unigrams,bigrams,trigrams)
    hist = Preprocessing.histogram_of_ngrams(3)
    gascent = g.gradient_ascent(fmaker.number_of_dimensions,10,fmaker,hist)
    start = d.datetime.now()
    hope = gascent.gradient_ascent()
    print(hope)
    print("it took :",str(d.datetime.now()-start))
    print("v_norm: ",str(np.dot(hope[0],hope[0])))"""
    """print(index_number)
    x = g.gradient_ascent(100000,10,())
    a=x.vector_v_init()
    b=x.vector_v_init()
    start = d.datetime.now()
    x.vector_multiplication(a,b)
    print("it took :",d.datetime.now() - start)"""

