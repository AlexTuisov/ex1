#This file contains some scripts for pre-processing the data
import os


#first try, little bit of statistics on the input
def preprocessing():
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
                    continue
                if word not in train_set_as_dictionary.keys():
                    count += 1
                    train_set_as_dictionary[word] = {tag : 1}
                elif tag not in train_set_as_dictionary[word].keys():
                    train_set_as_dictionary[word][tag] = 1
                else:
                    train_set_as_dictionary[word][tag] += 1
    print(count)
    #print(len(tags), tags)
    #print(train_set_as_dictionary)
    return train_set_as_dictionary

#this function should take in sentence from the .wtag file,
#and output two tuples of strings: 1. tuple of untagged words in a sentence
#with *, * at the start and @@@, @@@ at the end
#2. tuple of tags corresponding to each word

def prettifying_the_tagged_sentence(sentence):
    sentence_as_strings = sentence.split()
    # "@@@" is a stop sign
    sentence_as_strings = ["*", "*"] + sentence_as_strings + ["@@@", "@@@"]
    sentence_as_tags = []
    sentence_without_tags = []
    for tagged_word in sentence_as_strings:
        try:
            pure_word, tag = tagged_word.split('_')
        except:
            if tagged_word == "*":
                tag = "start"
                pure_word = "*"
            else:
                tag = "finish"
                pure_word = "@@@"
        sentence_as_tags.append(tag)
        sentence_without_tags.append(pure_word)
    return tuple(sentence_without_tags), tuple(sentence_as_tags)

def get_path_to_training_set():
    path = os.path.dirname(__file__)
    absolute_path = os.path.join(path, "data/train.wtag")
    return absolute_path
"""
def get_unigrams():
    unigram_features_as_dictionary = {}
    with open(get_path_to_training_set()) as raw_train_set:
        for num_of_sentence, sentence in enumerate(raw_train_set):
            train_set_as_strings = sentence.split()
            unigram_features_as_dictionary[num_of_sentence] = {}
            for tagged_word in train_set_as_strings:
                try:
                    word, tag = tagged_word.split('_')
                except:
                    continue
                if word not in unigram_features_as_dictionary[num_of_sentence].keys():
                    unigram_features_as_dictionary[num_of_sentence][word] = [tag, ]
                else:
                    unigram_features_as_dictionary[num_of_sentence][word].append(tag)
    return unigram_features_as_dictionary

def get_bigrams():
    bigram_features_as_dictionary = {}
    with open(get_path_to_training_set()) as raw_train_set:
        for num_of_sentence, sentence in enumerate(raw_train_set):
            words, tags = prettifying_the_tagged_sentence(sentence)
            for index, word in enumerate(words):
                if (index < 2) or (index > len(words) - 2):
                    continue
                if word not in bigram_features_as_dictionary.keys():
                    bigram_features_as_dictionary[word] = [(tags[index], tags[index-1], num_of_sentence),]
                else:
                    bigram_features_as_dictionary[word].append((tags[index], tags[index-1], num_of_sentence),)
    return bigram_features_as_dictionary

def get_trigrams():
    trigram_features_as_dictionary = {}
    with open(get_path_to_training_set()) as raw_train_set:
        for num_of_sentence, sentence in enumerate(raw_train_set):
            words, tags = prettifying_the_tagged_sentence(sentence)
            for index, word in enumerate(words):
                if (index < 2) or (index > len(words) - 2):
                    continue
                if word not in trigram_features_as_dictionary.keys():
                    trigram_features_as_dictionary[word] = [(tags[index], tags[index-1], tags[index-2], num_of_sentence),]
                else:
                    trigram_features_as_dictionary[word].append((tags[index], tags[index-1], tags[index-2], num_of_sentence),)
    return trigram_features_as_dictionary
"""
def get_ngrams(n):
    ngrams_as_dictionary = {}
    with open(get_path_to_training_set()) as raw_train_set:
        for num_of_sentence, sentence in enumerate(raw_train_set):
            words, tags = prettifying_the_tagged_sentence(sentence)
            ngrams_as_dictionary[num_of_sentence] = {}
            for index, word in enumerate(words):
                if (index < 2) or (index > len(words) - 2):
                    continue
                feature = []
                for i in range(0, n):
                    if (index - i) >= 0:
                        feature.append(tags[index - i])
                    else:
                        feature.append("start")
                feature = tuple(feature)
                if word not in ngrams_as_dictionary[num_of_sentence].keys():
                    ngrams_as_dictionary[num_of_sentence][word] = [feature, ]
                else:
                    ngrams_as_dictionary[num_of_sentence][word].append(feature)
    return ngrams_as_dictionary

#