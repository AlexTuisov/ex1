# File that contains search algorithms once the model has been learned
from copy import deepcopy

from scipy.sparse import csr_matrix

import GradientAscent
import numpy as np
import random
import datetime as d


class Searcher:
    def __init__(self, tags, gradient_descent, vector_v):
        self.tags = {tag: number+1 for number, tag in enumerate(tags)}
        self.tags["start"] = 0
        print(self.tags)
        self.inverted_tags = {number: tag for tag, number in self.tags.items()}
        self.tags_as_tuple = tuple(sorted(list(self.tags.values())))
        self.gradient_descent = gradient_descent
        self.vector_v = vector_v

    def viterbi_full_run(self, pure_test_set, test_set_with_true_tags):
        tagged_test_set = []
        for sentence in pure_test_set:
            #sentence = unprocessed_sentence.split()
            current_tag_sequence = self.viterbi_run_per_sentence(sentence)
            tagged_sentence = self.combine_words_with_tags(sentence, current_tag_sequence)
            tagged_test_set.append(tagged_sentence)
        self.check_outcome(tagged_test_set, test_set_with_true_tags)

    def viterbi_run_per_sentence(self, sentence_as_list_of_pure_words_without_finish):
        start = d.datetime.now()
        sentence_as_list_of_pure_words = list(sentence_as_list_of_pure_words_without_finish) + ["finish", "finish"]
        length_of_sentence = len(sentence_as_list_of_pure_words)

        # first step
        previous_pi_value_table = np.zeros((len(self.tags), len(self.tags)))
        previous_pi_value_table[self.tags["start"], self.tags["start"]] = 1
        list_of_pi_tables = np.zeros((length_of_sentence, len(self.tags), len(self.tags)))
        table_of_backpointers = np.zeros((length_of_sentence, len(self.tags), len(self.tags)))

        # the iterations
        current_pi_value_table = np.empty((len(self.tags), len(self.tags)))
        for k in range(0, length_of_sentence-1):
            current_pi_value_table = np.empty((len(self.tags), len(self.tags)))
            for u in self.tags_as_tuple:
                for v in self.tags_as_tuple:
                    start_inner = d.datetime.now()
                    current_pi_value_table[u][v], table_of_backpointers[k][u][v] = self.calculate_pi_value_and_backpointer(
                        u, v, previous_pi_value_table[u], sentence_as_list_of_pure_words[k])
                    print("one innermost iteration took: ", d.datetime.now() - start_inner)
            previous_pi_value_table = deepcopy(current_pi_value_table)
            list_of_pi_tables[k] = previous_pi_value_table

        # last step
        last_tags = np.unravel_index(current_pi_value_table.argmax(), current_pi_value_table.shape)
        sentence_as_tags = self.extract_backpointers(table_of_backpointers, last_tags, length_of_sentence)
        print("one iteration took: ", d.datetime.now() - start)
        return sentence_as_tags

    def calculate_pi_value_and_backpointer(self, u, v, previous_pi_value_column, word):
        pi_values = {}
        for t in self.tags_as_tuple:
            pi_values[previous_pi_value_column[t]+self.log_transition_probabilities(v, u, t, word)] = t
        best_value = max(list(pi_values.keys()))
        return best_value, pi_values[best_value]

    def extract_backpointers(self, table_of_backpointers, last_tags, length_of_sentence):
        last_tag = int(last_tags[1])
        tag_before_last = int(last_tags[0])
        sentence_as_tags = [last_tag, tag_before_last]
        for index in reversed(range(1, length_of_sentence-1)):
            new_tag = table_of_backpointers[index][last_tag][tag_before_last]
            sentence_as_tags.append(int(new_tag))
            last_tag = int(new_tag)
            tag_before_last = int(last_tag)
        sentence_as_tags.reverse()
        return sentence_as_tags[:-2]

    def log_transition_probabilities(self, self_tag, last_tag, previous_tag, word):
        # implementation of formula from slides of tirgul 4
        current_feature_vector = self.get_feature_vector(self_tag, previous_tag, last_tag, word)
        numerator = csr_matrix.dot(self.vector_v, current_feature_vector.transpose())
        denominator = 0.0
        for tag in self.tags_as_tuple:
            denominator += csr_matrix.dot(self.vector_v, self.get_feature_vector(tag, previous_tag, last_tag, word).transpose())
        return np.log(0.5)

    def get_feature_vector(self, tag, previous_tag, last_tag, word):
        local_feature_maker = self.gradient_descent.feature_maker
        new_vector = local_feature_maker.create_sparse_vector_of_features(self.inverted_tags[tag], self.inverted_tags[previous_tag], self.inverted_tags[last_tag], word)
        return new_vector

    def combine_words_with_tags(self, sentence, current_tag_sequence):
        tagged_sentence = []
        for index, output in enumerate(current_tag_sequence):
            to_append = sentence[index] + "_" + str(self.invert_tags[output])
            tagged_sentence.append(to_append)
        return tagged_sentence

    def check_outcome(self, tagged_set, true_tagged_set):
        count = 0.000001
        count_of_true = 0.000001
        for index_of_sentence, true_tagged_sentence in enumerate(true_tagged_set):
            for index_of_word, true_tagged_word in enumerate(true_tagged_sentence.split()):
                if true_tagged_word == tagged_set[index_of_sentence][index_of_word]:
                    count += 1
                    count_of_true += 1
                else:
                    count += 1
        print("the accuracy is: ", count_of_true/count)
        print("a sample tagged sentence: ", random.sample(tagged_set, 1))
        print("another one: ", random.sample(tagged_set, 1))
        return None

