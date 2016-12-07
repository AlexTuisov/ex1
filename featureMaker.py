from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import operator
import itertools
from snowballstemmer import EnglishStemmer as es


class feature_maker:

    def __init__(self,special_delimiter,k):
        self.special_delimiter =special_delimiter
        self.tag_counter={}
        self.number_of_dimensions=0
        self.number_of_sentences = 0
        self.param_index={}
        self.feature_indexes_list=list()
        self.reverse_param_index ={}
        self.k_most_seen_tags={}
        self.k = k
        self.feature_matrix=0
        self.feature_stats ={}#TODO : erase

    def create_feature_matrix(self):
        feature_matrix = csr_matrix((self.number_of_sentences,self.number_of_dimensions),dtype=int).toarray()
        for feature in self.param_index:
            sentences_index =self.param_index[feature][2]
            for index in sentences_index:
                feature_matrix[index][self.param_index[feature][0]] = 1
        return feature_matrix


    def sum_of_feature_vector(self,feature_matrix):
        sum_of_feature_vector = csr_matrix(1, self.number_of_dimensions)
        row_number = feature_matrix.shape()[0]
        for row in range(row_number):
            sum_of_feature_vector + feature_matrix[row]
        return sum_of_feature_vector



    def get_index_of_k_most_seen_tags(self):
        k_most_seen_tags ={}
        for word in self.tag_counter:
            tags = self.tag_counter[word]
            sorted_tags_by_count = sorted(tags.items(),key=operator.itemgetter(1),reverse=True)
            k_most_seen_tags[word] = list()
            tag_index = 1
            for tag in sorted_tags_by_count:
                if tag_index>self.k:
                    break
                k_most_seen_tags[word].append(tag[0])
                tag_index+=1
        self.k_most_seen_tags = k_most_seen_tags

    def create_expected_matrix(self):
        print("expected matrix init")
        rows_count = 1
        current_index = 0
        index = self.feature_indexes_list[1]
        while (index <= self.number_of_dimensions-1):
            trigram = self.reverse_param_index[index]
            trigram_data = self.param_index[trigram]
            current_word = trigram.split(self.special_delimiter)[0]
            for words in trigram_data[3]:
                lists_of_k_most_seen_tags = []
                if self.k_most_seen_tags.get(current_word,False):
                    lists_of_k_most_seen_tags.append(self.k_most_seen_tags[current_word])
                if self.k_most_seen_tags.get(words[0],False):
                    lists_of_k_most_seen_tags.append(self.k_most_seen_tags[words[0]])
                if self.k_most_seen_tags.get(words[1],False):
                    lists_of_k_most_seen_tags.append(self.k_most_seen_tags[words[1]])
                for element in itertools.product(*lists_of_k_most_seen_tags):
                    rows_count += 1
            index += 1
        index = self.feature_indexes_list[1]
        expected_matrix = csr_matrix((rows_count,self.number_of_dimensions),dtype=int).toarray()
        while (index <= self.number_of_dimensions - 1):
            trigram = self.reverse_param_index[index]
            trigram_data = self.param_index[trigram]()
            current_word = trigram.split(self.special_delimiter)[0]
            for words in trigram_data[3]:
                lists_of_k_most_seen_tags = []
                if self.k_most_seen_tags.get(current_word, False):
                    lists_of_k_most_seen_tags.append(self.k_most_seen_tags[current_word])
                if self.k_most_seen_tags.get(words[0], False):
                    lists_of_k_most_seen_tags.append(self.k_most_seen_tags[words[0]])
                if self.k_most_seen_tags.get(words[1], False):
                    lists_of_k_most_seen_tags.append(self.k_most_seen_tags[words[1]])
                for element in itertools.product(*lists_of_k_most_seen_tags):
                    self.modify_expected_matrix(element[0],element[1],element[2],current_word,expected_matrix,current_index)
                    current_index+=1
            index += 1
        print(expected_matrix)





    def init_all_params(self,unigrams,bigrams,trigrams):
        print("param_index init")
        self.get_feature_params_from_index(unigrams,bigrams,trigrams)
        print("k_index init")
        self.get_index_of_k_most_seen_tags()
        print("feature matrix init")
        self.create_feature_matrix()

    def get_feature_params_from_index(self, unigrams, bigrams, trigrams):
        feature_stats = {}
        stemmer = es()
        params_index = {}
        reverse_param_index = {}
        index_number = 0
        feature_indexes_list = list()
        for index in unigrams:
            sentence = unigrams[index]
            for word in sentence:
                stemmed_word = stemmer.stemWord(word)
                stemmed_word = stemmed_word.lower()
                if not self.tag_counter.get(stemmed_word,False):
                    self.tag_counter[stemmed_word]={}
                    self.tag_counter[stemmed_word][sentence[word][0][0][0]]=1
                elif not self.tag_counter[stemmed_word].get(sentence[word][0][0][0],False):
                    self.tag_counter[stemmed_word][sentence[word][0][0][0]] = 1
                else:
                    self.tag_counter[stemmed_word][sentence[word][0][0][0]] += 1
                word_with_tag = stemmed_word+self.special_delimiter+str(sentence[word][0][0][0])
                if not params_index.get(word_with_tag,False):
                    params_index[word_with_tag]=[index_number, 1 , [index,]]
                    reverse_param_index[index_number] = word_with_tag
                    index_number += 1
                else:
                    params_index[word_with_tag][1] += 1
                    params_index[word_with_tag][2].append(index)
        feature_indexes_list.append(index_number)
        for index in bigrams:
            sentence = bigrams[index]
            for word in sentence:
                stemmed_word = stemmer.stemWord(word)
                stemmed_word = stemmed_word.lower()
                word_with_tag = stemmed_word+self.special_delimiter+str(sentence[word][0][0][0])+self.special_delimiter+str(sentence[word][0][0][1])
                if not params_index.get(word_with_tag,False):
                    params_index[word_with_tag] = [index_number,1,[index,]]
                    reverse_param_index[index_number] = word_with_tag
                    index_number += 1
                else:
                    params_index[word_with_tag][1] += 1
                    params_index[word_with_tag][2].append(index)
        feature_indexes_list.append(index_number)
        print(feature_indexes_list)
        for index in trigrams:
            sentence = trigrams[index]
            for word in sentence:
                stemmed_word = stemmer.stemWord(word)
                stemmed_word = stemmed_word.lower()
                word_with_tag = stemmed_word+self.special_delimiter+str(sentence[word][0][0][0])+self.special_delimiter+str(sentence[word][0][0][1])+self.special_delimiter+str(sentence[word][0][0][2])
                if not params_index.get(word_with_tag,False):
                    params_index[word_with_tag]=[index_number,1,[index,],[sentence[word][0][1],]]
                    reverse_param_index[index_number] = word_with_tag
                    index_number += 1
                else:
                    params_index[word_with_tag][1] += 1
                    params_index[word_with_tag][2].append(index)
                    params_index[word_with_tag][3].append(sentence[word][0][1])
        print("number of feature dims is ",index_number)
        number_of_sentences = len(unigrams)
        self.number_of_dimensions = index_number
        self.number_of_sentences = number_of_sentences
        self.param_index = params_index
        self.feature_indexes_list = feature_indexes_list
        self.reverse_param_index = reverse_param_index
        feature_stats["uni"] = {}
        feature_stats["bi"] = {}
        feature_stats["tri"] = {}
        for feature in params_index:
            if params_index[feature][0]<14400:
                feature_stats["uni"][feature] = params_index[feature][1]
            elif params_index[feature][0]<46629:
                feature_stats["bi"][feature] = params_index[feature][1]
            else:
                feature_stats["tri"][feature] = params_index[feature][1]
        target1 = open("tri.tsv",'w')
        for feature in feature_stats["tri"]:
            target1.write(feature+"\t"+str(feature_stats["tri"][feature])+"\n")
        target1.close()



    def create_sparse_vector_of_features(self,current_tag,previous_tag,last_tag,word,param_index,number_of_params):
        feature_vec = csr_matrix((1, number_of_params)).toarray()
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

    def modify_expected_matrix(self, current_tag, previous_tag, last_tag, word, expected_feature_matrix,current_index):
        unigram = word+self.special_delimiter+current_tag
        bigram = unigram+self.special_delimiter+previous_tag
        trigram = bigram+self.special_delimiter+last_tag
        if self.param_index.get(unigram,False):
            unigram_index = self.param_index[unigram][0]
            expected_feature_matrix[current_index][unigram_index] = self.param_index[unigram][1]
        if self.param_index.get(bigram,False):
            bigram_index = self.param_index[bigram][0]
            expected_feature_matrix[current_index][bigram_index] = self.param_index[bigram][1]
        if self.param_index.get(trigram,False):
            trigram_index = self.param_index[trigram][0]
            expected_feature_matrix[current_index][trigram_index] = self.param_index[trigram][1]
        return expected_feature_matrix


























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