#This file contains some scripts for pre-processing the data
import os
import random


#first try, little bit of statistics on the input
def preprocessing():
    train_set_as_dictionary = {}
    path = os.path.dirname(__file__)
    absolute_path = os.path.join(path, "data/train2.wtag")
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
    print(tags)
    #print(train_set_as_dictionary)
    return set(tags)

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
    absolute_path = os.path.join(path, "data/train2.wtag")
    return absolute_path

def get_path_to_test_set():
    path = os.path.dirname(__file__)
    absolute_path = os.path.join(path, "data/test.wtag")
    return absolute_path

def create_little_test():
    pure_test_set = []
    tagged_little_test = []
    with open(get_path_to_test_set()) as raw_test_set:
        if random.random() < 0.002:
            for sentence in raw_test_set:
                tagged_little_test.append(sentence)
                pure_sentence = []
                for word in sentence.split():
                    pure_word = word.split("_")[0]
                    pure_sentence.append(pure_word)
                pure_test_set.append(pure_sentence)
    return pure_test_set, tagged_little_test

def get_pure_test_set():
    pure_test_set = []
    test_set_with_true_tags = []
    with open(get_path_to_test_set()) as raw_test_set:
        for sentence in raw_test_set:
            pure_sentence = []
            true_tagged_sentence = []
            for word in sentence.split():
                pure_word = word.split("_")[0]
                pure_sentence.append(pure_word)
                true_tagged_sentence.append(word)
            pure_test_set.append(pure_sentence)
            test_set_with_true_tags.append(true_tagged_sentence)
    return pure_test_set, test_set_with_true_tags

def get_ngrams(n):
    ngrams_as_dictionary = {}
    pure_sentences = {}
    with open(get_path_to_training_set()) as raw_train_set:
        for num_of_sentence, sentence in enumerate(raw_train_set):
            words, tags = prettifying_the_tagged_sentence(sentence)
            ngrams_as_dictionary[num_of_sentence] = {}
            for index, word in enumerate(words):
                if (index < 2) or (index > len(words) - 2):
                    continue
                tag_feature = []
                word_feature = []
                for i in range(0, n):
                    if (index - i) >= 0:
                        tag_feature.append(tags[index - i])
                        word_feature.append(words[index-i])
                    else:
                        tag_feature.append("start")
                        word_feature.append("*")
                feature = (tuple(tag_feature), tuple(word_feature[1:]))
                if word not in ngrams_as_dictionary[num_of_sentence].keys():
                    ngrams_as_dictionary[num_of_sentence][word] = [feature, ]
                else:
                    ngrams_as_dictionary[num_of_sentence][word].append(feature)
            pure_sentences[num_of_sentence] = words[2:-2]
    return ngrams_as_dictionary, pure_sentences

def histogram_of_ngrams(n):
    histogram_of_trigrams = {}
    with open(get_path_to_training_set()) as raw_train_set:
        for num_of_sentence, sentence in enumerate(raw_train_set):
            words, tags = prettifying_the_tagged_sentence(sentence)
            for index, word in enumerate(words):
                if (index < (n-1)) or (index > len(words) - (n-1)):
                    continue
                word_feature = []
                for i in range(0, n):
                    if (index - i) >= 0:
                        word_feature.append(words[index - i])
                    else:
                        word_feature.append("*")
                word_feature = tuple(word_feature)
                if word_feature in histogram_of_trigrams.keys():
                    histogram_of_trigrams[word_feature] += 1
                else:
                    histogram_of_trigrams[word_feature] = 1
    return histogram_of_trigrams

def longest_sentence():
    with open(get_path_to_training_set()) as train:
        longest = 0
        for sentence in train:
            if len(sentence.split()) > longest:
                longest = len(sentence.split())
    return longest

#