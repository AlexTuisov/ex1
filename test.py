import featureMaker
import Preprocessing
import datetime as d
import  GradientAscent as g
if __name__ == '__main__':
    unigrams ,w= Preprocessing.get_ngrams(1)
    bigrams ,w= Preprocessing.get_ngrams(2)
    trigrams, words = Preprocessing.get_ngrams(3)
    fmaker = featureMaker.feature_maker("###",2)
    fmaker.init_all_params(unigrams,bigrams,trigrams)
    gascent = g.gradient_ascent(fmaker.number_of_dimensions,10,fmaker)
    start = d.datetime.now()
    hope = gascent.gradient_ascent()
    print("it took :"+d.datetime.now()-start)
    print(hope)
    """print(index_number)
    x = g.gradient_ascent(100000,10,())
    a=x.vector_v_init()
    b=x.vector_v_init()
    start = d.datetime.now()
    x.vector_multiplication(a,b)
    print("it took :",d.datetime.now() - start)"""

