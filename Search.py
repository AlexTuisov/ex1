# File that contains search algorithms once the model has been learned
import numpy as np
import random

class Searcher:
    def __init__(self, tags, gradient_descent):
        # tags should be a tuple instead of set
        self.tags = tags
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
        table_of_backpointers = np.zeros((length_of_sentence, len(self.tags), len(self.tags)))

        # the iterations
        current_pi_value_table = np.empty((len(self.tags), len(self.tags)))
        for k in range(1, length_of_sentence):
            current_pi_value_table = np.empty((len(self.tags), len(self.tags)))
            for index_of_u, u in enumerate(self.tags):
                for index_of_v, v in enumerate(self.tags):
                    current_pi_value_table[index_of_u][index_of_v], table_of_backpointers[k, index_of_u, index_of_v] = self.calculate_pi_value_and_backpointer(
                        k, u, v, previous_pi_value_table[index_of_u], sentence_as_list_of_pure_words[k])
            previous_pi_value_table = np.copy(current_pi_value_table)

        # last step
        for index_of_u, u in enumerate(self.tags):
            for index_of_v, v in enumerate(self.tags):
                current_pi_value_table[index_of_u][index_of_v] = (previous_pi_value_table[index_of_u][index_of_v]
                                                                    + self.log_transition_probabilities("@@@", u, v, "finish"))
        last_tags = np.unravel_index(current_pi_value_table.argmax(), previous_pi_value_table.shape)
        sentence_as_tags = self.extract_backpointers(table_of_backpointers, last_tags, length_of_sentence)
        return sentence_as_tags

    def calculate_pi_value_and_backpointer(self, k, u, v, previous_pi_value_column, word):
        pi_values = {}
        for index_of_t, t in enumerate(self.tags):
            pi_values[previous_pi_value_column[index_of_t]+self.log_transition_probabilities(v, u, t, word)] = self.tags[t]
        best_value = max(list(pi_values.keys()))
        return best_value, pi_values[best_value]

    def extract_backpointers(self, table_of_backpointers, last_tags, length_of_sentence):
        last_tag = int(last_tags[1])
        tag_before_last = int(last_tags[0])
        sentence_as_tags = [last_tag, tag_before_last]
        for index in reversed(range(0, length_of_sentence-2)):
            new_tag = table_of_backpointers[index+2][last_tag][tag_before_last]
            sentence_as_tags.append(int(new_tag))
            last_tag = int(new_tag)
            tag_before_last = int(last_tag)
        sentence_as_tags.reverse()
        return sentence_as_tags

    def log_transition_probabilities(self, self_tag, previous_tag, last_tag, word):
        # will talk to Greg's function and calculate log
        # stub
        prob = 0.1
        if word == "start" and self_tag == self.tags["start"]:
            prob = 0.9
        if word == "the" and (self_tag == self.tags["DT"] or last_tag == self.tags["start"]):
            prob = 0.8
        return np.log(prob)


def combine_words_with_tags(sentence, tags):
    return 1


def check_outcome(tagged_set, true_tagged_set):
    return 1

