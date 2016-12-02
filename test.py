import featureMaker
import Preprocessing
import datetime as d
import  GradientAscent as g
if __name__ == '__main__':
    unigrams ,w= Preprocessing.get_ngrams(1)
    bigrams ,w= Preprocessing.get_ngrams(2)
    trigrams, words = Preprocessing.get_ngrams(3)
    ftest = featureMaker.feature_maker("###")
    index_number, paramsIndex,reverese = ftest.getFeatureParamsFromIndex(unigrams,bigrams,trigrams)
    tags = Preprocessing.preprocessing()
    gradient_ascent = g.gradient_ascent(index_number, 10, tags)
    now = d.datetime.now()
    gradient_ascent.gradient_ascent(words,trigrams,paramsIndex,reverese)
    print("it took: ", d.datetime.now() - now)
    """print(index_number)
    x = g.gradient_ascent(100000,10,())
    a=x.vector_v_init()
    b=x.vector_v_init()
    start = d.datetime.now()
    x.vector_multiplication(a,b)
    print("it took :",d.datetime.now() - start)"""