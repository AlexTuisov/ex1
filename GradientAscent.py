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


    def log_of_denominator(self,previous_tag,last_tag,sentence,index,param_index,vector_v):
        total_sum = 0
        for tag in self.tags:
            feature_vec = self.feature_maker.create_sparse_vector_of_features(tag,previous_tag,last_tag,sentence,index,param_index,self.number_of_dimensions)
            inner_product = self.vector_multiplication(vector_v,feature_vec)
            total_sum+=np.exp(inner_product)
        return np.log(total_sum)

    def log_of_numerator(self,tag, previous_tag, last_tag, sentence, index,
                                                                          param_index,vector_v):
        feature_vec = self.feature_maker.create_sparse_vector_of_features(tag, previous_tag, last_tag, sentence, index,
                                                                          param_index, self.number_of_dimensions)
        inner_product = self.vector_multiplication(vector_v, feature_vec)
        return inner_product

    def regularized_log_liklihood(self,sentences):
        for index in sentences:
            sentence = sentences[index]


