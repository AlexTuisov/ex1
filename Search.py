#File that contains search algorithms once the model has been learned

class searcher:
    def __init__(self, set_of_tags):
        #skeleton for constructor
        self.tags = set_of_tags
        return None

    def viterbi_run(self, sentence_as_list_of_pure_words, sentence_as_tags):
        #skeleton for Viterbi algorithm
        number_of_iterations = len(sentence_as_list_of_pure_words)
        return None

    def calculate_transition_probabilities(self):
        return None

    def calculate_exposition_probabilities(self):
        return None

