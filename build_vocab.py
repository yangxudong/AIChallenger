#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Build vocabularies of words from datasets"""
import argparse
from collections import Counter
import pandas as pd
import json
import os
import sys
reload(sys) 
sys.setdefaultencoding('utf-8') 

parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1, help="Minimum count for words in the dataset",
                    type=int)
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")

# Hyper parameters for the vocab
NUM_OOV_BUCKETS = 1 # number of buckets (= number of ids) for unknown words
PAD_WORD = '<pad>'


def save_vocab_to_txt_file(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to the id of the token.
    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, "w") as f:
        f.write("\n".join(token for token in vocab))


def save_dict_to_json(d, json_path):
    """Saves dict to json file
    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_vocab(txt_path, vocab):
    """Update word and tag vocabulary from dataset
    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method
    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path) as f:
        for i, line in enumerate(f):
            vocab.update(line.strip().split(' '))
    return i + 1

max_sentence_len = 0
def update_sentences(doc, sentences, doc_len, vocab):
    ss = doc.split(":")
    doc_len.append(len(ss))
    global max_sentence_len
    for s in ss:
        words = s.strip().split(' ')
        ll = len(words)
        if ll > max_sentence_len:
            max_sentence_len = ll
        vocab.update(words)
    sentences.extend(ss)

def corpus_sentences(csv_path, sentences, doc_len, vocab):
    data = pd.read_csv(csv_path, encoding='utf-8')
    data.sentences.apply(lambda x:  update_sentences(x, sentences, doc_len, vocab))

if __name__ == '__main__':
    args = parser.parse_args()

    # Build word vocab with train and test datasets
    print("Building word vocabulary...")
    words = Counter()
    sentences = []
    doc_len = []
    #size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train_content_words.txt'), words)
    #size_dev_sentences = update_vocab(os.path.join(args.data_dir, 'valid_content_words.txt'), words)
    #size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test_content_words.txt'), words)
    corpus_sentences(os.path.join(args.data_dir, 'train.csv'), sentences, doc_len, words)
    corpus_sentences(os.path.join(args.data_dir, 'valid.csv'), sentences, doc_len, words)
    corpus_sentences(os.path.join(args.data_dir, 'testa.csv'), sentences, doc_len, words)
    print("- done.")

    # Only keep most frequent tokens
    words = [tok for tok, count in words.items() if count >= args.min_count_word]
    # Add pad tokens
    if PAD_WORD not in words: words.append(PAD_WORD)
    # Save vocabularies to file
    print("Saving vocabularies to file...")
    save_vocab_to_txt_file(words, os.path.join(args.data_dir, 'words.txt'))
    print("- done.")

    print("Saving sentences to file ...")
    save_vocab_to_txt_file(sentences, os.path.join(args.data_dir, 'sentences.txt'))
    print("- done.")

    print("Saving doc_len to file ...")
    doc_len_path = os.path.join(args.data_dir, 'doc_len.txt')
    with open(doc_len_path, "w") as f:
        f.write("\n".join(str(l) for l in doc_len))
    print("- done.")

    # Save datasets properties in json file
    sizes = {
    #    'train_size': size_train_sentences,
    #    'valid_size': size_dev_sentences,
    #    'test_size': size_test_sentences,
        'max_sentence_num': max(doc_len),
        'max_sentence_len': max_sentence_len,
        'vocab_size': len(words) + NUM_OOV_BUCKETS,
        'pad_word': PAD_WORD,
        'num_oov_buckets': NUM_OOV_BUCKETS
    }
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))

    # Logging sizes
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))
