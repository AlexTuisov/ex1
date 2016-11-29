import numpy as np
import Preprocessing as pr
from scipy.sparse import bsr_matrix
def createSparseVectorOfFeatures(bigrams,unigrams,trigrams,sentenceNumber,numberOfParams):
    featureVec = bsr_matrix((1, numberOfParams))
    sentenceTags = unigrams[sentenceNumber]


    return ""


def getNumberOfDimentions():
    return 27000 #TODO :this will be infered later
