from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import operator
import itertools
from snowballstemmer import EnglishStemmer as es


class feature_maker:

    def __init__(self,special_delimiter,k,extended):
        self.special_delimiter =special_delimiter
        self.tag_counter={}
        self.number_of_dimensions=0
        self.number_of_sentences = 0
        self.param_index={}
        self.feature_indexes_list = list()
        self.reverse_param_index = {}
        self.k_most_seen_tags = {}
        self.k = k
        self.feature_matrix=0
        self.expected_feature_matrix_index = 0
        self.pruned_feature_index = {}
        self.extended = extended
        self.special_suffixes = ["ly", "al", "or", "ing", "ful", "ism", "ist", "ion", "sion", "tion", "acy",
                                 "hood", "or", "ar", "age", "like", "once", "ness", "able", "ible", "ify",
                                 "ic", "ate", "ous", "ize", "ish"]


    def create_feature_matrix(self):
        feature_matrix = lil_matrix((self.number_of_sentences,self.number_of_dimensions))
        for feature in self.param_index:
            sentences_index =self.param_index[feature][2]
            for index in sentences_index:
                if self.pruned_feature_index.get(feature,False):
                    feature_matrix[index,self.pruned_feature_index[feature][0]] += 1
        self.feature_matrix = csr_matrix(feature_matrix)


    def sum_of_feature_vector(self):
        sum_of_feature_vector = csr_matrix((1, self.number_of_dimensions))
        for row in range(self.feature_matrix.get_shape()[0]):
            sum_of_feature_vector += self.feature_matrix.getrow(row)
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

    def create_expected_matrix_index(self):
        expected_matrix_index = {}
        for feature in self.param_index:
            if len(feature.split(self.special_delimiter))>3:
                word = feature.split(self.special_delimiter)[0]
                expected_matrix = lil_matrix((len(self.k_most_seen_tags[word]), self.number_of_dimensions))
                previous_tag = feature.split(self.special_delimiter)[2]
                last_tag = feature.split(self.special_delimiter)[3]
                current_index = 0
                for tag in self.k_most_seen_tags[word]:
                    self.modify_expected_matrix(tag,previous_tag,last_tag,word,expected_matrix,current_index)
                    current_index += 1
                expected_matrix_index[feature] = csr_matrix(expected_matrix)
        self.expected_feature_matrix_index= expected_matrix_index




    def init_all_params(self,unigrams,bigrams,trigrams):
        print("param_index init")
        self.get_feature_params_from_index(unigrams,bigrams,trigrams)
        print("k_index init")
        self.get_index_of_k_most_seen_tags()
        print("feature matrix init")
        self.create_feature_matrix()
        print("expected matrix index init")
        self.create_expected_matrix_index()


    def get_feature_params_from_index(self, unigrams, bigrams, trigrams):
        stemmer = es()
        params_index = {}
        index_number = 0
        feature_indexes_list = list()
        for index in unigrams:
            sentence = unigrams[index]
            for word in sentence:
                #stemmed_word = stemmer.stemWord(word)
                #stemmed_word = stemmed_word.lower()
                if not self.tag_counter.get(word,False):
                    self.tag_counter[word]={}
                    self.tag_counter[word][sentence[word][0][0][0]]=1
                elif not self.tag_counter[word].get(sentence[word][0][0][0],False):
                    self.tag_counter[word][sentence[word][0][0][0]] = 1
                else:
                    self.tag_counter[word][sentence[word][0][0][0]] += 1
                word_with_tag = word+self.special_delimiter+str(sentence[word][0][0][0])
                if not params_index.get(word_with_tag,False):
                    params_index[word_with_tag]=[index_number, 1 , [index,]]
                    index_number += 1
                else:
                    params_index[word_with_tag][1] += 1
                    params_index[word_with_tag][2].append(index)
        self.tag_counter["*"]={}
        self.tag_counter["*"]["*"]=1
        feature_indexes_list.append(index_number)
        for index in bigrams:
            sentence = bigrams[index]
            for word in sentence:
                #stemmed_word = stemmer.stemWord(word)
                #stemmed_word = stemmed_word.lower()
                word_with_tag = word+self.special_delimiter+str(sentence[word][0][0][0])+self.special_delimiter+str(sentence[word][0][0][1])
                if not params_index.get(word_with_tag,False):
                    params_index[word_with_tag] = [index_number,1,[index,]]
                    index_number += 1
                else:
                    params_index[word_with_tag][1] += 1
                    params_index[word_with_tag][2].append(index)
        feature_indexes_list.append(index_number)
        for index in trigrams:
            sentence = trigrams[index]
            for word in sentence:
                #stemmed_word = stemmer.stemWord(word)
                #stemmed_word = stemmed_word.lower()
                if self.extended:
                    index_number = self.add_features_for_word_extended_model(word,sentence[word][0][0][0],sentence[word][0][0][1],sentence[word][0][0][2],index_number,params_index,index)
                word_with_tag = word+self.special_delimiter+str(sentence[word][0][0][0])+self.special_delimiter+str(sentence[word][0][0][1])+self.special_delimiter+str(sentence[word][0][0][2])
                if not params_index.get(word_with_tag,False):
                    params_index[word_with_tag]=[index_number,1,[index,],[sentence[word][0][1],]]
                    index_number += 1
                else:
                    params_index[word_with_tag][1] += 1
                    params_index[word_with_tag][2].append(index)
                    params_index[word_with_tag][3].append(sentence[word][0][1])
        number_of_sentences = len(unigrams)
        self.number_of_sentences = number_of_sentences
        self.param_index = params_index
        self.feature_indexes_list = feature_indexes_list
        print("pruning dimensions of features")
        self.prune_feature_dimensions(params_index)

    def create_sparse_vector_of_features(self,current_tag,previous_tag,last_tag,word):
        feature_vec = lil_matrix((1, self.number_of_dimensions))
        unigram = word+self.special_delimiter+current_tag
        bigram = unigram+self.special_delimiter+previous_tag
        trigram = bigram+self.special_delimiter+last_tag
        if self.pruned_feature_index.get(unigram,False):
            unigram_index = self.pruned_feature_index[unigram][0]
            feature_vec[0,unigram_index] = 1
        if self.pruned_feature_index.get(bigram,False):
            bigram_index=self.pruned_feature_index[bigram][0]
            feature_vec[0,bigram_index] = 1
        if self.pruned_feature_index.get(trigram,False):
            trigram_index = self.pruned_feature_index[trigram][0]
            feature_vec[0,trigram_index] = 1
        if self.extended:
            relevant_extended_features = self.return_relevat_features(word,current_tag,previous_tag,last_tag)
            for extended_feature in relevant_extended_features:
                if self.pruned_feature_index.get(extended_feature, False):
                    extended_feature_index = self.pruned_feature_index[extended_feature][0]
                    feature_vec[0,extended_feature_index] =1
        return csr_matrix(feature_vec)

    def modify_expected_matrix(self, current_tag, previous_tag, last_tag, word, expected_feature_matrix,current_index):
        unigram = word+self.special_delimiter+current_tag
        bigram = unigram+self.special_delimiter+previous_tag
        trigram = bigram+self.special_delimiter+last_tag
        if self.pruned_feature_index.get(unigram,False):
            unigram_index = self.pruned_feature_index[unigram][0]
            expected_feature_matrix[current_index, unigram_index] = 1
        if self.pruned_feature_index.get(bigram,False):
            bigram_index = self.pruned_feature_index[bigram][0]
            expected_feature_matrix[current_index,bigram_index] = 1
        if self.pruned_feature_index.get(trigram,False):
            trigram_index = self.pruned_feature_index[trigram][0]
            expected_feature_matrix[current_index,trigram_index] = 1
        if self.extended:
            relevant_extended_features = self.return_relevat_features(word, current_tag, previous_tag,last_tag)
            for extended_feature in relevant_extended_features:
                if self.pruned_feature_index.get(extended_feature, False):
                    extended_feature_index = self.pruned_feature_index[extended_feature][0]
                    expected_feature_matrix[current_index, extended_feature_index] = 1
        return expected_feature_matrix


    def prune_feature_dimensions(self,param_index):
        pruned_index = {}
        self.reverse_param_index={}
        new_index = 0
        for feature in param_index:
            if len(feature.split(self.special_delimiter))>3:
                if self.param_index[feature][1]>=3:
                    pruned_index[feature] = param_index[feature]
                    pruned_index[feature][0] = new_index
                    self.reverse_param_index[new_index] = feature
                    new_index += 1
            elif len(feature.split(self.special_delimiter))>2:
                if  param_index[feature][1]>=2:
                    pruned_index[feature] = param_index[feature]
                    pruned_index[feature][0] = new_index
                    self.reverse_param_index[new_index] = feature
                    new_index += 1
            else:
                pruned_index[feature] = param_index[feature]
                pruned_index[feature][0] = new_index
                self.reverse_param_index[new_index]=feature
                new_index += 1
        print("number of dimensions: ",new_index)
        self.number_of_dimensions = new_index
        self.pruned_feature_index = pruned_index




    def add_features_for_word_extended_model(self,word,current_tag,previous_tag,last_tag,current_index,param_index,sentence_number):
        tags_relevant_upper_case = ["*","."]
        if self.contain_digit(word):
            feature_name = "<<ContainDigit>>"+self.special_delimiter+current_tag
            if not param_index.get(feature_name, False):
                param_index[feature_name] = [current_index,1,[sentence_number,]]
                current_index += 1
            else:
                param_index[feature_name][2].append(sentence_number)
        if (self.is_capital(word)) and (previous_tag in tags_relevant_upper_case):
            feature_name = "<<UPPERSTART>>"+self.special_delimiter+current_tag
            if not param_index.get(feature_name, False):
                param_index[feature_name] = [current_index,1,[sentence_number,]]
                current_index += 1
            else:
                param_index[feature_name][2].append(sentence_number)
        elif (self.is_capital(word)):
            feature_name = "<<UPPER>>"+self.special_delimiter+current_tag
            if not param_index.get(feature_name, False):
                param_index[feature_name] = [current_index,1,[sentence_number,]]
                current_index += 1
            else:
                param_index[feature_name][2].append(sentence_number)
        for suffix in self.special_suffixes:
            if word.endswith(suffix):
                feature_name = "<<"+suffix + "_suffix>>" + self.special_delimiter + current_tag
                if not param_index.get(feature_name,False):
                    param_index[feature_name]=[current_index,1,[sentence_number,]]
                    current_index += 1
                else:
                    param_index[feature_name][2].append(sentence_number)
        unigram_extended_feature = "<<"+current_tag+">>"
        bigram_extended_feature = "<<"+current_tag+"_"+previous_tag+">>"
        trigram_extended_feature ="<<"+current_tag+"_"+previous_tag+"_"+last_tag+">>"
        if not param_index.get(unigram_extended_feature, False):
            param_index[unigram_extended_feature] = [current_index, 1, [sentence_number, ]]
            current_index += 1
        else:
            param_index[unigram_extended_feature][2].append(sentence_number)
        if not param_index.get(bigram_extended_feature, False):
            param_index[bigram_extended_feature] = [current_index, 1, [sentence_number, ]]
            current_index += 1
        else:
            param_index[bigram_extended_feature][2].append(sentence_number)
        if not param_index.get(trigram_extended_feature, False):
            param_index[trigram_extended_feature] = [current_index, 1, [sentence_number, ]]
            current_index += 1
        else:
            param_index[trigram_extended_feature][2].append(sentence_number)
        return current_index



    def return_relevat_features(self,word,current_tag,previous_tag,last_tag):
        relevant_features=[]
        tags_relevant_upper_case = ["start","*","."]
        if self.contain_digit(word):
            relevant_features.append("<<ContainDigit>>"+self.special_delimiter+current_tag)
        if (self.is_capital(word)) and (previous_tag in tags_relevant_upper_case):
            relevant_features.append("<<UPPERSTART>>"+self.special_delimiter+current_tag)
        elif (self.is_capital(word)):
            relevant_features.append("<<UPPER>>"+self.special_delimiter+current_tag)
        for suffix in self.special_suffixes:
            if word.endswith(suffix):
                relevant_features.append("<<"+suffix + "_suffix>>" + self.special_delimiter + current_tag)
        unigram_extended_feature = "<<" + current_tag + ">>"
        bigram_extended_feature = "<<" + current_tag + "_" + previous_tag + ">>"
        trigram_extended_feature = "<<" + current_tag + "_" + previous_tag + "_" + last_tag + ">>"
        relevant_features.append(unigram_extended_feature)
        relevant_features.append(bigram_extended_feature)
        relevant_features.append(trigram_extended_feature)
        return relevant_features






    def is_capital(self,word):
        if word[0].isupper():
            return True
        return False

    def contain_digit(self,word):
        if any(c.isdigit() for c in word):
            return True
        return False




"""    def create_expected_matrix(self):
        print("expected matrix init")
        rows_count = 1
        current_index = 0
        index =0
        stemmer = es()
        while (index <= self.number_of_dimensions-1):
            feature = self.reverse_param_index[index]
            if len(feature.split(self.special_delimiter)) > 3:
                trigram_data = self.param_index[feature]
                current_word = feature.split(self.special_delimiter)[0]
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
        index = 0
        print(rows_count)
        expected_matrix = lil_matrix((rows_count,self.number_of_dimensions),dtype=int)
        while (index <= self.number_of_dimensions-1):
            feature = self.reverse_param_index[index]
            if len(feature.split(self.special_delimiter))>3:
                trigram_data = self.param_index[feature]
                current_word = feature.split(self.special_delimiter)[0]
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
                        current_index += 1
            index += 1
        self.expected_feature_matrix = csr_matrix(expected_matrix)
"""














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

"""feature_stats["uni"] = {}
   feature_stats["bi"] = {}
   feature_stats["tri"] = {}
   for feature in params_index:
       if params_index[feature][0]<14400:
           feature_stats["uni"][feature] = params_index[feature][1]
       elif params_index[feature][0]<46629:
           feature_stats["bi"][feature] = params_index[feature][1]
       else:
           feature_stats["tri"][feature] = params_index[feature][1]
   target1 = open("bi.tsv",'w')
   for feature in feature_stats["bi"]:
       target1.write(feature+"\t"+str(feature_stats["bi"][feature])+"\n")
   target1.close()"""

""" def create_expected_matrix_index(self):
        trigram_index_of_matries ={}
        index =0
        while (index <= self.number_of_dimensions-1):
            feature = self.reverse_param_index[index]
            if len(feature.split(self.special_delimiter)) > 3:
                trigram_data = self.param_index[feature]
                current_word = feature.split(self.special_delimiter)[0]
                for words in trigram_data[3]:
                    trigram_tuple = (current_word,words[0],words[1],)
                    expected_matrix_rows_number  =1
                    lists_of_k_most_seen_tags = []
                    if self.k_most_seen_tags.get(current_word,False):
                        lists_of_k_most_seen_tags.append(self.k_most_seen_tags[current_word])
                        expected_matrix_rows_number *= len(self.k_most_seen_tags[current_word])
                    if self.k_most_seen_tags.get(words[0],False):
                        lists_of_k_most_seen_tags.append(self.k_most_seen_tags[words[0]])
                        expected_matrix_rows_number *= len(self.k_most_seen_tags[words[0]])
                    if self.k_most_seen_tags.get(words[1],False):
                        lists_of_k_most_seen_tags.append(self.k_most_seen_tags[words[1]])
                        expected_matrix_rows_number *= len(self.k_most_seen_tags[words[1]])
                    expected_matrix_per_tag = lil_matrix((expected_matrix_rows_number,self.number_of_dimensions))
                    current_index = 0
                    for element in itertools.product(*lists_of_k_most_seen_tags):
                        expected_matrix_per_tag = self.modify_expected_matrix(element[0],element[1],element[2],current_word,expected_matrix_per_tag,current_index)
                        current_index += 1
                    trigram_index_of_matries[trigram_tuple]=csr_matrix(expected_matrix_per_tag)
            index += 1
        self.expected_feature_matrix_index = trigram_index_of_matries"""