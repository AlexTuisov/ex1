from scipy.sparse import bsr_matrix

class feature_maker:

    def __init__(self,special_delimiter):
        self.special_delimiter =special_delimiter

    def getFeatureParamsFromIndex(self,unigrams,bigrams,trigrams):#TODO: create reverese index
        paramsIndex = {}
        reverese_param_index = {}
        index_number = 0
        for index in unigrams:
            sentence = unigrams[index]
            for word in sentence:
                word_with_tag = word+self.special_delimiter+str(sentence[word][0][0])
                if not paramsIndex.get(word_with_tag,False):
                    paramsIndex[word_with_tag]=[index_number,1]
                    reverese_param_index[index_number] = word_with_tag
                    index_number += 1
                else:
                    paramsIndex[word_with_tag][1] += 1
        for index in bigrams:
            sentence = bigrams[index]
            for word in sentence:
                word_with_tag = word+self.special_delimiter+str(sentence[word][0][0])+self.special_delimiter+str(sentence[word][0][1])
                if not paramsIndex.get(word_with_tag,False):
                    paramsIndex[word_with_tag] = [index_number,1]
                    reverese_param_index[index_number] = word_with_tag
                    index_number += 1
                else:
                    paramsIndex[word_with_tag][1] += 1
        for index in trigrams:
            sentence = trigrams[index]
            for word in sentence:
                word_with_tag = word+self.special_delimiter+str(sentence[word][0][0])+self.special_delimiter+str(sentence[word][0][1])+self.special_delimiter+str(sentence[word][0][2])#TODO : make more generic
                if not paramsIndex.get(word_with_tag,False):
                    paramsIndex[word_with_tag]=[index_number,1]
                    reverese_param_index[index_number] = word_with_tag
                    index_number += 1
                else:
                    paramsIndex[word_with_tag][1] += 1
        print("number of feature dims is ",index_number)
        return index_number,paramsIndex,reverese_param_index


    """"
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
        return feature_vec"""


    def create_sparse_vector_of_features(self,current_tag,previous_tag,last_tag,sentence,index,param_index,number_of_params):
        feature_vec = bsr_matrix((1, number_of_params)).toarray()
        word = sentence[index]
        unigram = word+self.special_delimiter+current_tag
        bigram = unigram+self.special_delimiter+previous_tag
        trigram = bigram+self.special_delimiter+last_tag
        unigram_index = param_index[unigram]
        feature_vec[unigram_index] = 1
        bigram_index=param_index[bigram]
        feature_vec[bigram_index] = 1
        trigram_index = param_index[trigram]
        feature_vec[trigram_index] = 1
        return feature_vec


