from scipy.sparse import bsr_matrix
import operator

class feature_maker:

    def __init__(self,special_delimiter):
        self.special_delimiter =special_delimiter
        self.tag_counter={}
        self.number_of_dimension=0
        self.number_of_sentences = 0
        self.param_index={}

    def create_feature_matrix(self,param_index):
        feature_matrix = bsr_matrix(self.number_of_sentences,self.number_of_dimensions)
        for feature in param_index:
            sentences_index =param_index[feature][2]
            for index in sentences_index:
                feature_matrix[index][param_index[feature][0]] = 1
        return feature_matrix


    def sum_of_feature_vector(self,feature_matrix):
        sum_of_feature_vector = bsr_matrix(1, self.number_of_dimensions)
        row_number = feature_matrix.shape()[0]
        for row in range(row_number):
            sum_of_feature_vector + feature_matrix[row]
        return sum_of_feature_vector



    def get_index_of_k_most_seen_tags(self,k):
        k_most_seen_tags ={}
        for word in self.tag_counter:
            tags = self.tag_counter[word]
            sorted_tags_by_count = sorted(tags.items(),key=operator.itemgetter(1))
            k_most_seen_tags[word] = list()
            tag_index = 1
            for tag in sorted_tags_by_count:
                if tag_index>k:
                    break
                k_most_seen_tags[word].append(tag)
                tag_index+=1
        return k_most_seen_tags

    def create_expected_matrix_for_sentence(self,sentence):

        print("")

    def create_tag_permutation(self,sentence,k_most_seen_tags):
        print("")



    def getFeatureParamsFromIndex(self,unigrams,bigrams,trigrams):
        params_index = {}
        reverese_param_index = {}
        index_number = 0
        feature_indexes_list = list()
        for index in unigrams:
            sentence = unigrams[index]
            for word in sentence:
                if not self.tag_counter.get(word,False):
                    self.tag_counter[word]={}
                    self.tag_counter[word][sentence[word][0][0]]=1
                elif not self.tag_counter[word].get(sentence[0][0],False):
                    self.tag_counter[word][sentence[word][0][0]] = 1
                else:
                    self.tag_counter[word][sentence[word][0][0]] += 1
                word_with_tag = word+self.special_delimiter+str(sentence[word][0][0])
                if not params_index.get(word_with_tag,False):
                    params_index[word_with_tag]=[index_number,1,list(index)]
                    reverese_param_index[index_number] = word_with_tag
                    index_number += 1
                else:
                    params_index[word_with_tag][1] += 1
                    params_index[word_with_tag][2].append(index)
        feature_indexes_list.append(index_number)
        for index in bigrams:
            sentence = bigrams[index]
            for word in sentence:
                word_with_tag = word+self.special_delimiter+str(sentence[word][0][0])+self.special_delimiter+str(sentence[word][0][1])
                if not params_index.get(word_with_tag,False):
                    params_index[word_with_tag] = [index_number,1,list(index)]
                    reverese_param_index[index_number] = word_with_tag
                    index_number += 1
                else:
                    params_index[word_with_tag][1] += 1
                    params_index[word_with_tag][2].append(index)
        feature_indexes_list.append(index_number)
        for index in trigrams:
            sentence = trigrams[index]
            for word in sentence:
                word_with_tag = word+self.special_delimiter+str(sentence[word][0][0])+self.special_delimiter+str(sentence[word][0][1])+self.special_delimiter+str(sentence[word][0][2])#TODO : make more generic
                if not params_index.get(word_with_tag,False):
                    params_index[word_with_tag]=[index_number,1,list(index)]
                    reverese_param_index[index_number] = word_with_tag
                    index_number += 1
                else:
                    params_index[word_with_tag][1] += 1
                    params_index[word_with_tag][2].append(index)
        feature_indexes_list.append(index_number)
        print("number of feature dims is ",index_number)
        number_of_sentences = len(unigrams)
        self.number_of_dimension = index_number
        self.number_of_sentences = number_of_sentences
        self.param_index = params_index
        return params_index,reverese_param_index,feature_indexes_list#TODO: arrange returns


    def create_sparse_vector_of_features(self,current_tag,previous_tag,last_tag,word,param_index,number_of_params):
        feature_vec = bsr_matrix((1, number_of_params)).toarray()
        unigram = word+self.special_delimiter+current_tag
        bigram = unigram+self.special_delimiter+previous_tag
        trigram = bigram+self.special_delimiter+last_tag
        if param_index.get(unigram,False):
            unigram_index = param_index[unigram][0]
            feature_vec[0][unigram_index] = 1
        if param_index.get(bigram,False):
            bigram_index=param_index[bigram][0]
            feature_vec[0][bigram_index] = 1
        if param_index.get(trigram,False):
            trigram_index = param_index[trigram][0]
            feature_vec[0][trigram_index] = 1
        return feature_vec


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