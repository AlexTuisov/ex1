#This file contains some scripts for pre-processing the data
import os

def unigrams():
    train_set_as_dictionary = {}
    path = os.path.dirname(__file__)
    absolute_path = os.path.join(path, "data/train.wtag")
    count = 0
    tags = set([])
    with open(absolute_path) as raw_train_set:
        for sentence in raw_train_set:
            train_set_as_strings = sentence.split()
            for tagged_word in train_set_as_strings:
                try:
                    word, tag = tagged_word.split('_')
                    tags.add(tag)
                except:
                    count += 1
                    continue
                if word not in train_set_as_dictionary.keys():
                    train_set_as_dictionary[word] = {tag : 1}
                elif tag not in train_set_as_dictionary[word].keys():
                    train_set_as_dictionary[word][tag] = 1
                else:
                    train_set_as_dictionary[word][tag] += 1
    print(count)
    print(len(tags), tags)
    print(train_set_as_dictionary)
    return train_set_as_dictionary


def bigrams():
    path = os.path.dirname(__file__)
    absolute_path = os.path.join(path, "data/train.wtag")
    count = 0
    tags = set([])
    with open(absolute_path) as raw_train_set:
        #To be added
        None
