import Preprocessing as pr
import numpy as np
import threading
import datetime as d
import time
from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse import csr_matrix
def vector_multiplication(vector_a, vector_b):
    return csr_matrix.dot(vector_a, vector_b)

class gradient_ascent: #TODO: handle non seen tags!!!!!
    def __init__(self,number_of_dimensions,lambda_value,feature_maker,trigram_histogram):
        self.number_of_dimensions=number_of_dimensions
        self.lambda_value = lambda_value
        #self.tags = tags
        self.vector_v = 0
        self.feature_index = 0
        self.feature_maker = feature_maker
        self.sum_of_feature_vector = self.feature_maker.sum_of_feature_vector()
        self.trigram_histogram =trigram_histogram



    def vector_v_init(self):
        vector_v = csr_matrix((1, self.number_of_dimensions)).toarray()
        return vector_v


    def log_of_numerator(self,vector_v):
        vector = self.feature_maker.feature_matrix.dot(vector_v)
        res = vector.sum()
        return res

    def log_of_denominator(self,vector_v):
        total_sum = 0
        for trigram in self.feature_maker.expected_feature_matrix_index:
            matrix = self.feature_maker.expected_feature_matrix_index[trigram]
            total_sum+=self.feature_maker.param_index[trigram][1]*np.log(self.sum_of_exponential_permutations(matrix,vector_v))
        return total_sum


    def sum_of_exponential_permutations(self,matrix,vector_v):
        initial=matrix.dot(vector_v)
        initial=np.exp(initial)
        return initial.sum()



    def regularized_log_likelihood(self,vector_v):
        res = self.log_of_denominator(vector_v)-self.log_of_numerator(vector_v)+float(float(self.lambda_value)/2)*np.dot(vector_v,vector_v)
        print(res)
        return res


    def gradient_of_log_likelihood(self,vector_v):
        expected_sum = csr_matrix((1, self.number_of_dimensions))
        for trigram in self.feature_maker.expected_feature_matrix_index:
            matrix = self.feature_maker.expected_feature_matrix_index[trigram]
            sum_of_exponentias = self.sum_of_exponential_permutations(matrix,vector_v)
            for row in range(matrix.get_shape()[0]):
                feature = matrix.getrow(row)
                expected_feature = feature.multiply(self.probability_calculation(feature,sum_of_exponentias,vector_v))
                expected_sum+=expected_feature.multiply(self.feature_maker.param_index[trigram][1])
        res = (expected_sum - self.sum_of_feature_vector + csr_matrix(vector_v).multiply(self.lambda_value)).toarray()
        return res.transpose()


    def probability_calculation(self,feature,sum_of_exp_prem,vector_v):
        numerator = np.exp(feature.dot(vector_v))
        return float(float(numerator)/sum_of_exp_prem)


    def gradient_ascent(self):
        print("starting gradient ascent")
        vector_v0 = np.ndarray((1,self.number_of_dimensions))
        vector_v0.fill(0)
        results = fmin_l_bfgs_b(self.regularized_log_likelihood,vector_v0,self.gradient_of_log_likelihood,factr=1e10)
        self.vector_v=results[0]
        print("gradient ascent ended")
        return results










    def calculation(self,trigram,vector_v,lock):
        sum = 0
        while threading.active_count() > 10:
            #wait
            continue
        matrix = self.feature_maker.expected_feature_matrix_index[trigram]
        sum_of_exponentias = self.sum_of_exponential_permutations(matrix, vector_v)
        for row in range(matrix.get_shape()[0]):
            feature = matrix.getrow(row)
            expected_feature = feature.multiply(self.probability_calculation(feature, sum_of_exponentias, vector_v))
            sum+=expected_feature.multiply(self.feature_maker.param_index[trigram][1])
        return sum

    def test(self,vector_v):
        expected_sum = csr_matrix((1, self.number_of_dimensions))
        threads=[]
        lock = threading.Lock()
        results = []
        for trigram in self.feature_maker.expected_feature_matrix_index:
            t = threading.Thread(target=self.calculation,args=(trigram,vector_v,lock))
            threads.append(t)
            results.append(t.start())
        print(len(results))
        for thread in threads:
            thread.join()
        res = (expected_sum - self.sum_of_feature_vector + csr_matrix(vector_v).multiply(self.lambda_value)).toarray()
        return res.transpose()




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