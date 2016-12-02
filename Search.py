#File that contains search algorithms once the model has been learned
import numpy as np

class searcher:
    def __init__(self, set_of_tags, vector_v):
        self.tags = set_of_tags
        self.transition_probabilities = self.calculate_transition_probabilities()
        self.exposition_probabilities = self.calculate_exposition_probabilities()
        return None

    def viterbi_run(self, sentence_as_list_of_pure_words):
        previous_pi_value_table = np.ones((len(self.tags), len(self.tags)))
        table_of_backpointers = np.empty((len(sentence_as_list_of_pure_words), len(self.tags), len(self.tags)))

        for k in range(1, len(sentence_as_list_of_pure_words)):
            current_pi_value_table = np.empty((len(self.tags), len(self.tags)))
            for index_of_u, u in enumerate(self.tags):
                for index_of_v, v in enumerate(self.tags):
                    current_pi_value_table[index_of_u][index_of_v], table_of_backpointers[k, index_of_u, index_of_v] = self.calculate_pi_value_and_backpointer(
                        k, u, v, previous_pi_value_table[index_of_u], sentence_as_list_of_pure_words[k])
            previous_pi_value_table = current_pi_value_table
        #sentence_as_tags = self.extract_backpointers(table_of_backpointers, current_pi_value_table)
        return None

    def calculate_pi_value_and_backpointer(self, k, u, v, previous_pi_value_column, word):
        pi_values = {}
        if k==1:
            pi_values[(self.transition_probabilities[(v, "start", u)]
                                                       *self.exposition_probabilities[word, v])] = "start"
        else:
            for index_of_w, w in enumerate(self.tags):
                pi_values[(previous_pi_value_column[index_of_w]*self.transition_probabilities[(v, w, u)]
                                                           *self.exposition_probabilities[word, v])] = w
        best_value = max(list(pi_values.keys()))
        return best_value, pi_values[best_value]

    def extract_backpointers(self, ):
        return None

    def calculate_transition_probabilities(self):
        return None

    def calculate_exposition_probabilities(self):
        return None

