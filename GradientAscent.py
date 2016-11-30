import Preprocessing as pr
import numpy as np
import scipy as sp
import featureMaker
from scipy.sparse import bsr_matrix

class gradient_ascent:
    def __init__(self,number_of_dimensions,lambda_value,tags):
        self.number_of_dimensions=number_of_dimensions
        self.lambda_value = lambda_value
        self.tags = tags
        self.vector_v = self.vector_v_init()
        self.feature_maker = featureMaker.feature_maker("###")

    def vector_v_init(self):
        vector_v = bsr_matrix((1, self.number_of_dimensions)).toarray()
        return vector_v


    def vector_multiplication(self,vector_a,vector_b):
        return np.dot(vector_a,vector_b.transpose())


    def log_of_denominator(self,previous_tag,third_tag,sentence,index,param_index):
        total_sum = 0
        for tag in self.tags:
            feature_vec = self.feature_maker.create_sparse_vector_of_features(tag,previous_tag,third_tag,sentence,index,param_index,self.number_of_dimensions)
            inner_product = self.vector_multiplication(self.vector_v,feature_vec)
            total_sum+=np.exp(inner_product)
        return np.log(total_sum)

    def log_of_numerator(self,tag, previous_tag, third_tag, sentence, index,
                                                                          param_index):
        feature_vec = self.feature_maker.create_sparse_vector_of_features(tag, previous_tag, third_tag, sentence, index,
                                                                          param_index, self.number_of_dimensions)
        inner_product = self.vector_multiplication(self.vector_v, feature_vec)
        return inner_product


    def get_regularized_log_likelihood_function(self):
        def regularized_log_likelihood(self,sentences,trigrams,param_index):
            total_sum = 0
            for index in sentences:
                sentence = sentences[index]
                for i,word in enumerate(sentence):
                    previous_tag = trigrams[index][word][0][1]
                    third_tag = trigrams[index][word][0][2]
                    current_tag = trigrams[index][word][0][0]
                    log_of_numerator = self.log_of_numerator(current_tag,previous_tag,third_tag,sentence,i,param_index)
                    log_of_denomenator = self.log_of_denominator(previous_tag,third_tag,sentence,index,param_index)
                    total_sum += (log_of_numerator - log_of_denomenator)

            regularization = self.lambda_value*0.5*self.vector_multiplication(self.vector_v,self.vector_v)
            return (total_sum-regularization)
        return regularized_log_likelihood


