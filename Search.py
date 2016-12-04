#File that contains search algorithms once the model has been learned
import numpy as np

class searcher:
    def __init__(self, tags, vector_v):
        #tags should be a tuple instead of set
        self.tags = tuple(tags)
        self.transition_probabilities = self.calculate_transition_probabilities()
        self.exposition_probabilities = self.calculate_exposition_probabilities()
        return None

    def viterbi_run(self, sentence_as_list_of_pure_words):
        length_of_sentence = len(sentence_as_list_of_pure_words)
        previous_pi_value_table = np.ones((len(self.tags), len(self.tags)))
        #first step
        table_of_backpointers = np.empty((length_of_sentence, len(self.tags), len(self.tags)))
        #the iterations
        for k in range(1, length_of_sentence):
            current_pi_value_table = np.empty((len(self.tags), len(self.tags)))
            for index_of_u, u in enumerate(self.tags):
                for index_of_v, v in enumerate(self.tags):
                    current_pi_value_table[index_of_u][index_of_v], table_of_backpointers[k, index_of_u, index_of_v] = self.calculate_pi_value_and_backpointer(
                        k, u, v, previous_pi_value_table[index_of_u], sentence_as_list_of_pure_words[k])
            previous_pi_value_table = current_pi_value_table.__deepcopy__()
        #last step
        for index_of_u, u in enumerate(self.tags):
            for index_of_v, v in enumerate(self.tags):
                current_pi_value_table[index_of_u][index_of_v] = (previous_pi_value_table[index_of_u][index_of_v]
                                                                    * self.transition_probabilities[("stop", u, v)])
        last_tag_value = max(current_pi_value_table.flat)
        last_tags = np.unravel_index(current_pi_value_table.argmax(), current_pi_value_table.shape)
        sentence_as_tags = self.extract_backpointers(table_of_backpointers, last_tags, length_of_sentence)
        return sentence_as_tags

    def calculate_pi_value_and_backpointer(self, k, u, v, previous_pi_value_column, word):
        pi_values = {}
        if k==1:
            pi_values[(self.transition_probabilities[(v, "start", u)]
                                                       *self.exposition_probabilities[(word, v)])] = "start"
        else:
            for index_of_w, w in enumerate(self.tags):
                pi_values[(previous_pi_value_column[index_of_w]*self.transition_probabilities[(v, w, u)]
                                                           *self.exposition_probabilities[(word, v)])] = w
        best_value = max(list(pi_values.keys()))
        return best_value, pi_values[best_value]

    def extract_backpointers(self, table_of_backpointers, last_tags, length_of_sentence):
        last_tag = self.tags[last_tags[1]]
        tag_before_last = self.tags[last_tags[0]]
        sentence_as_tags = [last_tag, tag_before_last]
        for index in reversed(range(0, length_of_sentence-2)):
            new_tag = table_of_backpointers[index+2][last_tag][tag_before_last]
            sentence_as_tags.append[self.tags[new_tag]]
            last_tag = new_tag
            tag_before_last = last_tag
        return sentence_as_tags.reverse()

    def calculate_transition_probabilities(self):
        return None

    def calculate_exposition_probabilities(self):
        return None

