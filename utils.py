"""Utility functions."""

# [ Imports ]
# [ -Python ]
import os
import json
import pickle
import argparse
import logging
from collections import Counter

class BuplicateFilter(object):
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv

LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']

def _log_level_string_to_int(log_level_string):
    if not log_level_string in LOG_LEVEL_STRINGS:
        message = 'invalid choice: {0} (choose from {1})'.format(log_level_string, LOG_LEVEL_STRINGS)
        raise argparse.ArgumentTypeError(message)

    log_level_int = getattr(logging, log_level_string, logging.INFO)
    return log_level_int

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cache", "-n", dest="no_cache", action="store_true")
    parser.add_argument("--batch_size", "-b", dest="batch_size", type=int, default=16)
    parser.add_argument("--epochs", "-e", dest="num_epochs", type=int, default=5)
    parser.add_argument("--embedding", "-v", dest="embedding_size", type=int, default=300)
    parser.add_argument("--layer_size", "-l", dest="layer_size", type=int, default=200)
    parser.add_argument("--log_level", "-d", default='INFO', dest="log_level", type=_log_level_string_to_int, help='Set the logging output level. {0}'.format(LOG_LEVEL_STRINGS))
    parser.add_argument("--dynet-autobatch")
    parser.add_argument("--dynet-gpu")
    parser.add_argument("--dynet-gpus")
    args = parser.parse_args()
    return args


class Vocab:
    def __init__(self, corpus, label=False):
        self.counts = Counter()
        for word in corpus:
            self.counts[word] += 1
        self.word_to_idx = {w: i for i, (w, _) in enumerate(self.counts.most_common())}
        if not label:
            self.word_to_idx['<UNK>'] = len(self.word_to_idx)

    def __len__(self):
        return len(self.word_to_idx)

    def __getitem__(self, word):
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        else:
            return self.word_to_idx['<UNK>']


def tokenize_from_parse(parse):
    tokens = []
    for token in parse.split():
        if token.endswith(')'):
            tokens.append(token.rstrip(')'))
    return tokens


def read(filename, length):
    sentence1 = []
    sentence2 = []
    labels = []
    i = 0
    with open(filename) as f:
        for line in f:
            example = json.loads(line)
            i += 1
            if example['gold_label'] == "-":
                continue
            sentence1_list = ['<NULL>'] + tokenize_from_parse(example['sentence1_parse'])
            sentence2_list = ['<NULL>'] + tokenize_from_parse(example['sentence2_parse'])
            sentence1.append(sentence1_list)
            sentence2.append(sentence2_list)
            labels.append(example['gold_label'])
            if i % 100 == 0:
                print("\x1b[2K\r{}/{}".format(i, length), end="")
    print()
    return sentence1, sentence2, labels
