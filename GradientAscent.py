import Preprocessing as pr
from scipy.sparse import bsr_matrix
def createSparseVectorOfFeatures(bigrams,unigrams,trigrams,sentenceNumber,numberOfParams,paramIndex):
    special_delimiter = "###"#TODO: will be decided outside
    feature_vec = bsr_matrix((1, numberOfParams)).toarray()
    sentence_tags = unigrams[sentenceNumber]
    for word in sentence_tags:
        vector_index = paramIndex[word+special_delimiter+sentence_tags[word][0]]
        feature_vec[vector_index] = 1
    sentence_tags = bigrams[sentenceNumber]
    for word in sentence_tags:
        vector_index = paramIndex[word + special_delimiter + sentence_tags[word][0]+special_delimiter + sentence_tags[word][1]]
        feature_vec[vector_index] = 1
    sentence_tags = trigrams[sentenceNumber]
    for word in sentence_tags:
        vector_index = paramIndex[
            word + special_delimiter + sentence_tags[word][0] + special_delimiter + sentence_tags[word][1]+special_delimiter + sentence_tags[word][2]]
        feature_vec[vector_index] = 1
    return feature_vec


