import Preprocessing as pr
import numpy as np
import datetime as d
from scipy.optimize import fmin_l_bfgs_b
import featureMaker
from scipy.sparse import bsr_matrix
from multiprocessing import Pool



def vector_multiplication(vector_a, vector_b):
    return np.dot(vector_a, vector_b.transpose())

class gradient_ascent: #TODO: handle non seen tags!!!!!
    def __init__(self,number_of_dimensions,lambda_value,tags):
        self.number_of_dimensions=number_of_dimensions
        self.lambda_value = lambda_value
        self.tags = tags
        self.vector_v = self.vector_v_init()
        self.feature_index = 0
        self.feature_maker = featureMaker.feature_maker("###")

    def vector_v_init(self):
        vector_v = bsr_matrix((1, self.number_of_dimensions)).toarray()
        return vector_v

    """def process_calc(self,tag,previous_tag,third_tag,word,param_index):
        feature_vec = self.feature_maker.create_sparse_vector_of_features(tag, previous_tag, third_tag, word,
                                                                          param_index, self.number_of_dimensions)
        inner_product = vector_multiplication(self.vector_v, feature_vec)
        return np.exp(inner_product)"""

    def log_of_denominator(self,previous_tag,third_tag,word,param_index):
        total_sum = 0
        for tag in self.tags:
            feature_vec = self.feature_maker.create_sparse_vector_of_features(tag,previous_tag,third_tag,word,param_index,self.number_of_dimensions)
            inner_product = vector_multiplication(self.vector_v,feature_vec)
            total_sum+=np.exp(inner_product)
        return np.log(total_sum)

    def log_of_numerator(self,tag, previous_tag, third_tag, word,param_index):
        feature_vec = self.feature_maker.create_sparse_vector_of_features(tag, previous_tag, third_tag, word,
                                                                          param_index, self.number_of_dimensions)
        inner_product = vector_multiplication(self.vector_v, feature_vec)
        return inner_product


    def regularized_log_likelihood(self,sentences,trigrams,param_index):
        total_sum = 0
        for index in sentences:
            sentence = sentences[index]
            for i,word in enumerate(sentence):
                previous_tag = trigrams[index][word][0][1]
                third_tag = trigrams[index][word][0][2]
                current_tag = trigrams[index][word][0][0]
                log_of_numerator = self.log_of_numerator(current_tag,previous_tag,third_tag,word,param_index)
                log_of_denomenator = self.log_of_denominator(previous_tag,third_tag,word,param_index)
                total_sum += (log_of_numerator - log_of_denomenator)
        regularization = self.lambda_value*0.5*vector_multiplication(self.vector_v,self.vector_v)
        return (-total_sum+regularization)

    def gradient_of_log_likelihood_function_by_vj(self,reverese_param_index,param_index):

        total_sum = 0
        if self.feature_index > len(param_index):
            self.feature_index = 0
        feature_counter = param_index[reverese_param_index[self.feature_index]][1]
        feature = reverese_param_index[self.feature_index]
        feature_components = feature.split(self.feature_maker.special_delimiter)
        for tag in self.tags:
            if (len(feature_components) >= 3):
                new_feature = feature_components[0]+self.feature_maker.special_delimiter+tag+feature_components[2:]
            else:
                new_feature = feature_components[0]+self.feature_maker.special_delimiter+tag
            if param_index.get(new_feature,False):
                new_feature_counter = param_index[new_feature][1]
                new_feature_index = param_index[new_feature][0]
                total_sum += new_feature_counter*self.probability_calculation(new_feature_index,new_feature,param_index)
        return self.lambda_value*self.vector_v[self.feature_index]- total_sum - feature_counter

    def probability_calculation(self,index_of_feature,feature,param_index):
        if param_index.get(index_of_feature , False):
            numerator = np.exp(self.vector_v[index_of_feature])
        else:
            numerator = float(0.1/0.9)
        denominator = 0
        feature_components = feature.split(self.feature_maker.special_delimiter)
        for tag in self.tags:
            if len(feature_components) >= 3:
                new_feature = feature_components[0] + self.feature_maker.special_delimiter + tag + feature_components[
                                                                                                   2:]
            else:
                new_feature = feature_components[0] + self.feature_maker.special_delimiter + tag
            new_feature_vec = bsr_matrix((1, self.number_of_dimensions)).toarray()
            if param_index.get(new_feature,False):
                new_feature_index = param_index[new_feature]
                new_feature_vec[0][new_feature_index]=1
            denominator += np.exp(vector_multiplication(self.vector_v,new_feature_vec))
        return float(float(numerator)/denominator)*0.9


    def gradient_ascent(self,sentences,trigram,param_index,reverese_param_index):
        fmin_l_bfgs_b(self.regularized_log_likelihood(sentences,trigram,param_index), self.vector_v, fprime=self.gradient_of_log_likelihood_function_by_vj(reverese_param_index,param_index), factr=1000000000.0,maxiter=2)
