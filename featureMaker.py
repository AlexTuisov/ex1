import Preprocessing as prp
def getFeatureParamsFromIndex(unigrams,bigrams,trigrams):
    special_delimiter = "###"
    paramsIndex = {}
    index_number = 0
    for index in unigrams:
        sentence = unigrams[index]
        for word in sentence:
            word_with_tag = word+special_delimiter+sentence[word][0]#TODO : make more generic
            if not paramsIndex.get(word_with_tag,False):
                paramsIndex[word_with_tag]=index_number
                index_number += 1
    for index in bigrams:
        for word in sentence:
            word_with_tag = word+special_delimiter+sentence[word][0]+special_delimiter+sentence[word][1]#TODO : make more generic
            if not paramsIndex.get(word_with_tag,False):
                paramsIndex[word_with_tag]=index_number
                index_number += 1
    for index in trigrams:
        for word in sentence:
            word_with_tag = word+special_delimiter+sentence[word][0]+special_delimiter+sentence[word][1]+special_delimiter+sentence[word][2]#TODO : make more generic
            if not paramsIndex.get(word_with_tag,False):
                paramsIndex[word_with_tag]=index_number
                index_number += 1
    print("number of feature dims is ",index_number)
    return tuple(index_number,paramsIndex)