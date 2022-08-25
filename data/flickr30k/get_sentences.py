import os
import re
from itertools import combinations
import pandas as pd
import csv

def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset
    input:
      fn - full file path to the sentence file to parse

    output:
        a dictionary, where each image is its own key and has a list of the five sentences that belong to it
    """
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    full_annotations = {}
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        dict_key = None
        for token in sentence.split():
            # extract the ID as a key for the dict
            if re.search('#\d$', token) is not None:
                dict_key = token.split(".")[0] # truncate the whole .jpg#{0,1,2,3,4} stuff
            elif token == '.': # do not include the punctuation
                pass
            elif add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)
        # if we already have one sentence for the annotation, then append
        if dict_key in full_annotations:
            full_annotations[dict_key].append(' '.join(words))
        else:
            full_annotations[dict_key] = [' '.join(words)]

        # sentence_data = {'sentence': ' '.join(words), 'phrases': []}
        # for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
        #     sentence_data['phrases'].append({'first_word_index': index,
        #                                      'phrase': phrase,
        #                                      'phrase_id': p_id,
        #                                      'phrase_type': p_type})
        #
        # annotations.append(sentence_data)

    return full_annotations

def generate_training_data_combinations(full_annotations_dict):
    """
    This function generates all 2-tuples for the 5-sentences per image.
    Hence, 5! combinations for each image
    """
    list_of_combinations = []
    for key in full_annotations_dict.keys():
        combs = list(combinations(full_annotations_dict[key], 2)) # all combinations of length 2
        list_of_combinations.append(combs)
    flat_list = [item for sublist in list_of_combinations for item in sublist]
    df = pd.DataFrame(flat_list)
    return df


def main():
    cwd = os.getcwd()
    print(cwd)
    dicct = get_sentence_data("results_20130124.token")
    train_data = generate_training_data_combinations(dicct)
    train_data.to_csv('flickr30k.csv', index=False, quoting=csv.QUOTE_ALL, header=False)

if __name__ == '__main__':
    main()