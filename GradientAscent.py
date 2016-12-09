import Preprocessing as pr
import numpy as np
import datetime as d
from scipy.optimize import fmin_l_bfgs_b
import featureMaker
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

def vector_multiplication(vector_a, vector_b):
    return csr_matrix.dot(vector_a, vector_b)

class gradient_ascent: #TODO: handle non seen tags!!!!!
    def __init__(self,number_of_dimensions,lambda_value,feature_maker):
        self.number_of_dimensions=number_of_dimensions
        self.lambda_value = lambda_value
        #self.tags = tags
        #self.vector_v = self.vector_v_init()
        self.feature_index = 0
        self.feature_maker = feature_maker
        self.sum_of_feature_vector = self.feature_maker.sum_of_feature_vector()




    def vector_v_init(self):
        vector_v = csr_matrix((1, self.number_of_dimensions)).toarray()
        return vector_v


    def log_of_numerator(self,vector_v):
        vector = self.feature_maker.feature_matrix.dot(vector_v)
        res = vector.sum()
        print("numer=",res)
        return res

    def log_of_denominator(self,vector_v):
        total_sum = 0
        for trigram in self.feature_maker.expected_feature_matrix_index:
            matrix = self.feature_maker.expected_feature_matrix_index[trigram]
            total_sum+=np.log(self.sum_of_exponential_permutations(matrix,vector_v))
        print("denom=",total_sum)
        return total_sum


    def sum_of_exponential_permutations(self,matrix,vector_v):
        """for row in range(matrix.get_shape()[0]):
            feature_sum += np.exp(vector_multiplication(self.vector_v, matrix.getrow(row).toarray()))"""
        initial=matrix.dot(vector_v)
        initial=np.exp(initial)
        return initial.sum()



    def regularized_log_likelihood(self,vector_v):
        res = self.log_of_denominator(vector_v)-self.log_of_numerator(vector_v)+5*np.dot(vector_v,vector_v)
        return res


    def gradient_of_log_likelihood(self,vector_v):
        for trigram in self.feature_maker.expected_feature_matrix_index:
            matrix = self.feature_maker.expected_feature_matrix_index[trigram]
            sum_of_exponentias = self.sum_of_exponential_permutations(matrix,vector_v)
            expected_sum = csr_matrix((1,self.number_of_dimensions))
            for row in range(matrix.get_shape()[0]):
                feature = matrix.getrow(row)
                expected_feature = feature.multiply(self.probability_calculation(feature,sum_of_exponentias,vector_v))
                expected_sum+=expected_feature
        res = (-expected_sum + self.sum_of_feature_vector - csr_matrix(vector_v).multiply(self.lambda_value)).toarray()
        return res.transpose()


    def probability_calculation(self,feature,sum_of_exp_prem,vector_v):#TODO : handle non seen words
        numerator = np.exp(feature.dot(vector_v))
        return float(float(numerator)/sum_of_exp_prem)


    def gradient_ascent(self):
        print("starting gradient ascent")
        vector_v0 = np.ndarray((1,self.number_of_dimensions))
        vector_v0.fill(0)
        print(type(vector_v0))
        fmin_l_bfgs_b(self.regularized_log_likelihood,vector_v0,fprime=self.gradient_of_log_likelihood,maxiter=2)













"""    def gradient_of_log_likelihood_function_by_vj(self,reverese_param_index,param_index):
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
        return self.lambda_value*self.vector_v[self.feature_index]- total_sum - feature_counter"""

"""def log_of_denominator(self,previous_tag,third_tag,word,param_index):
    total_sum = 0
    for tag in self.tags:
        feature_vec = self.feature_maker.create_sparse_vector_of_features(tag,previous_tag,third_tag,word,param_index,self.number_of_dimensions)
        inner_product = vector_multiplication(self.vector_v,feature_vec)
        total_sum+=np.exp(inner_product)
    return np.log(total_sum)"""

"""def log_of_numerator(self,tag, previous_tag, third_tag, word,param_index):
    feature_vec = self.feature_maker.create_sparse_vector_of_features(tag, previous_tag, third_tag, word,
                                                                      param_index, self.number_of_dimensions)
    inner_product = vector_multiplication(self.vector_v, feature_vec)
    return inner_product"""

"""    def regularized_log_likelihood(self,sentences,trigrams,param_index):
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
        return (-total_sum+regularization)"""

"""  def probability_calculation(self,index_of_feature,feature,param_index):
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
        return float(float(numerator)/denominator)*0.9"""

"""    def sum_of_feature_vector_init(self,feature_matrix):
        return self.feature_maker.sum_of_feature_vector(feature_matrix)"""