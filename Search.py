# File that contains search algorithms once the model has been learned
from copy import deepcopy
import sys
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

from snowballstemmer import EnglishStemmer as es

import GradientAscent
import numpy as np
import random
import datetime as d
from multiprocessing import Pool


class Searcher:
    def __init__(self, tags, gradient_ascent, vector_v):
        self.tags = {tag: number+1 for number, tag in enumerate(tags)}
        self.tags["start"] = 0
        self.inverted_tags = {number: tag for tag, number in self.tags.items()}
        self.tags_as_tuple = tuple(sorted(list(self.tags.values())))
        self.gradient_ascent = gradient_ascent
        self.vector_v = vector_v
        self.feature_maker = gradient_ascent.feature_maker

    def viterbi_full_run(self, pure_test_set, test_set_with_true_tags):
        accuracy_list = []
        with Pool(4) as p:
            results_list = p.map(self.viterbi_run_per_sentence, pure_test_set)
        for number, tagged_sentence in enumerate(results_list):
            true_sentence = test_set_with_true_tags[number]
            accuracy_list.append(self.check_outcome((tagged_sentence,), (true_sentence,)))
        average_accuracy = sum(accuracy_list)/len(accuracy_list)
        """print("The average accuracy is:", average_accuracy)
        print("A sample tagged sentance:")
        print(random.choice(results_list))"""
        return average_accuracy


    def viterbi_run_per_sentence(self, sentence_as_list_of_pure_words_without_finish):
        sentence_as_list_of_pure_words = list(sentence_as_list_of_pure_words_without_finish) + ["finish", "finish"]
        length_of_sentence = len(sentence_as_list_of_pure_words)

        # first step
        previous_pi_value_table = np.full((len(self.tags), len(self.tags)), 0, dtype=np.float)
        previous_pi_value_table[self.tags["start"], self.tags["start"]] = 1
        table_of_backpointers = np.zeros((length_of_sentence-1, len(self.tags), len(self.tags)), dtype=np.int16)

        # the iterations
        current_pi_value_table = np.zeros((len(self.tags), len(self.tags)))

        #extracting the relevant tags for u and v

        for k in range(0, length_of_sentence-1):
            current_pi_value_table = np.full((len(self.tags), len(self.tags)), 0, dtype=np.float)
            relevant_tags_for_u = self.extract_relevant_tags(k-1, sentence_as_list_of_pure_words)
            relevant_tags_for_v = self.extract_relevant_tags(k, sentence_as_list_of_pure_words)
            for u in self.tags_as_tuple:
                for v in self.tags_as_tuple:
                    """if (v not in relevant_tags_for_v) or (u not in relevant_tags_for_u):
                        current_pi_value_table[u][v] = -(sys.float_info.max/1000000)
                        continue"""
                    results = self.calculate_pi_value_and_backpointer(
                        u, v, previous_pi_value_table[u], sentence_as_list_of_pure_words[k], k, sentence_as_list_of_pure_words)
                    current_pi_value_table[u][v] = results[0]
                    table_of_backpointers[k][u][v] = results[1]
            previous_pi_value_table = deepcopy(current_pi_value_table)

        # last step
        last_tags = np.unravel_index(current_pi_value_table.argmax(), current_pi_value_table.shape)
        sentence_as_tags = self.extract_backpointers(table_of_backpointers, last_tags, length_of_sentence, sentence_as_list_of_pure_words)
        tagged_sentence = self.combine_words_with_tags(sentence_as_list_of_pure_words_without_finish, sentence_as_tags)
        return tagged_sentence

    def calculate_pi_value_and_backpointer(self, u, v, previous_pi_value_row, word, index, sentence):
        pi_values = {}
        relevant_tags_for_t = self.extract_relevant_tags(index - 2, sentence)
        relevant_tags_for_v = self.extract_relevant_tags(index, sentence)
        exponential_matrix_dictionary = {}
        denominator_dictionary = {}
        for t in relevant_tags_for_t:
            permutation_matrix = self.create_exponential_permutations_matrix(relevant_tags_for_v, u, t, word)
            exponential_matrix_dictionary[t] = permutation_matrix
        for t in exponential_matrix_dictionary:
            permutation_matrix = exponential_matrix_dictionary[t]
            denominator_dictionary[t] = self.gradient_ascent.sum_of_exponential_permutations(permutation_matrix,
                                                                                             self.vector_v)
        exponential_matrix_dictionary = {}  # memmory purpose
        for t in relevant_tags_for_t:
            pi_values[previous_pi_value_row[t] + self.log_transition_probabilities(v, u, t, word,
                                                                                   denominator_dictionary[t])] = t
        best_value = max(list(pi_values.keys()))
        return (best_value, pi_values[best_value])

    def extract_backpointers(self, table_of_backpointers, last_tags, length_of_sentence, sentence):
        last_tag = int(last_tags[0])
        tag_before_last = int(last_tags[1])
        tag_before_last = self.postprocessing(tag_before_last, str.lower(sentence[length_of_sentence-3]))
        sentence_as_tags = [last_tag, tag_before_last]
        for index in reversed(range(1, length_of_sentence-1)):
            new_tag = table_of_backpointers[index][tag_before_last][last_tag]
            new_tag = self.postprocessing(new_tag, str.lower(sentence[index-2]))
            sentence_as_tags.append(int(new_tag))
            last_tag = int(new_tag)
            tag_before_last = int(last_tag)
        sentence_as_tags.reverse()
        return sentence_as_tags[1:-1]

    def log_transition_probabilities(self, self_tag, previous_tag, last_tag, word, denominator):
        # implementation of formula from slides of tirgul 4
        numerator = np.exp(self.get_feature_vector(self_tag, previous_tag, last_tag, word).dot(self.vector_v))
        value = np.log(float(numerator)/denominator)
        return value

    def get_feature_vector(self, tag, previous_tag, last_tag, word):
        new_vector = self.feature_maker.create_sparse_vector_of_features(self.inverted_tags[tag],
                                                                         self.inverted_tags[previous_tag], self.inverted_tags[last_tag], word)
        return new_vector

    def combine_words_with_tags(self, sentence, current_tag_sequence):
        tagged_sentence = []
        for index, output in enumerate(current_tag_sequence):
            to_append = sentence[index] + "_" + str(self.inverted_tags[output])
            tagged_sentence.append(to_append)
        return tagged_sentence

    def check_outcome(self, tagged_set, true_tagged_set):
        count = 0.0
        count_of_true = 0.0
        for index_of_sentence, true_tagged_sentence in enumerate(true_tagged_set):
            for index_of_word, true_tagged_word in enumerate(true_tagged_sentence):
                if true_tagged_word == tagged_set[index_of_sentence][index_of_word]:
                    count += 1
                    count_of_true += 1
                else:
                    count += 1
        return float(count_of_true/count)

    def extract_relevant_tags(self, index, sentence):
        if index < 0:
            return (self.tags["start"],)
        if sentence[index] in self.feature_maker.k_most_seen_tags.keys():
            to_return = []
            sequence = tuple(self.feature_maker.k_most_seen_tags[sentence[index]])
            for item in sequence:
                if item in self.tags.keys():
                    to_return.append(self.tags[item])
            if len(to_return) <= 0:
                to_return.append(self.tags["NN"])
            to_return = tuple(to_return)
            return to_return
        else:
            return(self.get_open_tags())

    def get_open_tags(self):
        #open parts of speech in english
        open_tags = ("NN", "NNP", "JJ", "NNS", "RB", "VBD", "VB", "VBZ", "VBN", "VBG", "VBP","CD")
        to_return = []
        for item in open_tags:
            to_return.append(self.tags[item])
        return tuple(to_return)


    def create_exponential_permutations_matrix(self,relevant_tags,previous_tag,last_tag, word):
        permutation_matrix = lil_matrix((len(relevant_tags), self.gradient_ascent.feature_maker.number_of_dimensions))
        current_index = 0
        for current_tag in relevant_tags:
            self.feature_maker.modify_expected_matrix(self.inverted_tags[current_tag], self.inverted_tags[previous_tag],self.inverted_tags[last_tag], word, permutation_matrix, current_index)
            current_index += 1
        return csr_matrix(permutation_matrix)

    def postprocessing(self, supposed_tag, word):
        actual_tag = supposed_tag
        if word in self.feature_maker.k_most_seen_tags.keys():
            if len(self.feature_maker.k_most_seen_tags[word]) >= 1:
                if self.feature_maker.k_most_seen_tags[word][0] in self.tags.keys():
                    actual_tag = self.tags[self.feature_maker.k_most_seen_tags[word][0]]
        return actual_tag
