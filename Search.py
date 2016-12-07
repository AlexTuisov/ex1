# File that contains search algorithms once the model has been learned
from copy import deepcopy

import numpy as np
import random


class Searcher:
    def __init__(self, tags, gradient_descent):
        # tags should be a tuple instead of set
        self.tags = tags
        self.tags_as_tuple = tuple(sorted(list(tags.values())))
        self.gradient_descent = gradient_descent

    def viterbi_full_run(self, pure_test_set, test_set_with_true_tags):
        tagged_test_set = []
        for sentence in pure_test_set:
            tags = self.viterbi_run_per_sentence(sentence)
            tagged_sentence = combine_words_with_tags(sentence, tags)
            tagged_test_set.append(tagged_sentence)
        return check_outcome(tagged_test_set, test_set_with_true_tags)

    def viterbi_run_per_sentence(self, sentence_as_list_of_pure_words):
        length_of_sentence = len(sentence_as_list_of_pure_words)
        previous_pi_value_table = np.zeros((len(self.tags), len(self.tags)))
        previous_pi_value_table[self.tags["start"], self.tags["start"]] = 1

        # first step
        table_of_backpointers = np.zeros((length_of_sentence+1, len(self.tags), len(self.tags)))

        # the iterations
        current_pi_value_table = np.empty((len(self.tags), len(self.tags)))
        for k in range(0, length_of_sentence):
            current_pi_value_table = np.empty((len(self.tags), len(self.tags)))
            for u in self.tags_as_tuple:
                for v in self.tags_as_tuple:
                    current_pi_value_table[u][v], table_of_backpointers[k][u][v] = self.calculate_pi_value_and_backpointer(
                        u, v, previous_pi_value_table[u], sentence_as_list_of_pure_words[k])
            previous_pi_value_table = deepcopy(current_pi_value_table)

        # last step
        for u in self.tags_as_tuple:
            for v in self.tags_as_tuple:
                current_pi_value_table[u][v] = (previous_pi_value_table[u][v]
                                                                    + self.log_transition_probabilities(self.tags["@@@"], u, v, "finish"))
        last_tags = np.unravel_index(current_pi_value_table.argmax(), current_pi_value_table.shape)
        sentence_as_tags = self.extract_backpointers(table_of_backpointers, last_tags, length_of_sentence)
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
        return sentence_as_tags

    def log_transition_probabilities(self, self_tag, last_tag, previous_tag, word):
        # will talk to Greg's function and calculate log
        # stub
        prob = 0.1 + random.random()*0.05
        if word == "the" and self_tag == self.tags["DT"]:
            prob += 0.5
        if word == "dog"  and self_tag == self.tags["NN"] and previous_tag == self.tags["DT"]:
            prob += 0.5
        if word == "barks" and self_tag == self.tags["VB"] and last_tag == self.tags["DT"]:
            prob += 0.5
        if word == "finish" and previous_tag == self.tags["NN"] and self_tag == self.tags["@@@"]:
            prob += 0.6
        if word == "finish" and last_tag == self.tags["NN"] and self_tag == self.tags["@@@"]:
            prob += 0.2
        return np.log(prob)


def combine_words_with_tags(sentence, tags):
    return 1


def check_outcome(tagged_set, true_tagged_set):
    return 1

