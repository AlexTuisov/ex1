from scipy.sparse import bsr_matrix

def getFeatureParamsFromIndex(unigrams,bigrams,trigrams):
    special_delimiter = "###"
    paramsIndex = {}
    index_number = 0
    for index in unigrams:
        sentence = unigrams[index]
        for word in sentence:
            word_with_tag = word+special_delimiter+str(sentence[word][0][0])#TODO : make more generic
            if not paramsIndex.get(word_with_tag,False):
                paramsIndex[word_with_tag]=index_number
                index_number += 1
    for index in bigrams:
        sentence = bigrams[index]
        for word in sentence:
            print(sentence[word][0][1])
            word_with_tag = word+special_delimiter+str(sentence[word][0][0])+special_delimiter+str(sentence[word][0][1])#TODO : make more generic
            if not paramsIndex.get(word_with_tag,False):
                paramsIndex[word_with_tag]=index_number
                index_number += 1
    for index in trigrams:
        sentence = trigrams[index]
        for word in sentence:
            word_with_tag = word+special_delimiter+str(sentence[word][0][0])+special_delimiter+str(sentence[word][0][1])+special_delimiter+str(sentence[word][0][2])#TODO : make more generic
            if not paramsIndex.get(word_with_tag,False):
                paramsIndex[word_with_tag]=index_number
                index_number += 1
    print("number of feature dims is ",index_number)
    return index_number,paramsIndex



def createSparseVectorOfFeatures(bigrams,unigrams,trigrams,sentenceNumber,numberOfParams,paramIndex):
    special_delimiter = "###"#TODO: will be decided outside
    feature_vec = bsr_matrix((1, numberOfParams)).toarray()
    sentence_tags = unigrams[sentenceNumber]
    for word in sentence_tags:
        vector_index = paramIndex[word+special_delimiter+sentence_tags[word][0][0]]
        feature_vec[vector_index] = 1
    sentence_tags = bigrams[sentenceNumber]
    for word in sentence_tags:
        vector_index = paramIndex[word + special_delimiter + sentence_tags[word][0][0]+special_delimiter + sentence_tags[word][0][1]]
        feature_vec[vector_index] = 1
    sentence_tags = trigrams[sentenceNumber]
    for word in sentence_tags:
        vector_index = paramIndex[
            word + special_delimiter + sentence_tags[word][0][0] + special_delimiter + sentence_tags[word][0][1]+special_delimiter + sentence_tags[word][0][2]]
        feature_vec[vector_index] = 1
    return feature_vec