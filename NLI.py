"""Natural Language Entailment using https://aclweb.org/anthology/D16-1244."""

# [ -Imports ]
import os
import json
import pickle
import logging
from itertools import chain
from collections import Counter
# [ -Third Party ]
from sklearn.utils import shuffle
import dynet as dy


logging.basicConfig(level=logging.INFO)


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

    def get(self, word):
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        else:
            return self.word_to_idx['<UNK>']


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
            print(example['sentence1'])
            print(example['sentence2'])
            print(example['sentence1_parse'])
            print(example['sentence2_parse'])
            exit()
            sentence1.append(['<NULL>'] + word_tokenize(example['sentence1'].lower()))
            sentence2.append(['<NULL>'] + word_tokenize(example['sentence2'].lower()))
            labels.append(example['gold_label'])
            if i % 100 == 0:
                print("\x1b[2K\r{}/{}".format(i, length), end="")
    print()
    return sentence1, sentence2, labels


train_data = "snli_1.0/snli_1.0_train.jsonl"
dev_data = "snli_1.0/snli_1.0_dev.jsonl"
test_data = "snli_1.0/snli_1.0_test.jsonl"

if not os.path.exists("cache"):
    logging.info("Cache not found, reprocessing data.")
    train_sentence1, train_sentence2, train_labels = read(train_data, 550152)
    dev_sentence1, dev_sentence2, dev_labels = read(dev_data, 10000)
    test_sentence1, test_sentence2, test_labels = read(test_data, 10000)
    raw_data = chain(*train_sentence1, *train_sentence2)
    vocab = Vocab(raw_data)
    pickle.dump(vocab, open("cache/vocab.p", "wb"))
    pickle.dump([train_sentence1, train_sentence2, train_labels], open("cache/train.p", "wb"))
    pickle.dump([test_sentence1, test_sentence2, test_labels], open("cache/test.p", "wb"))
    pickle.dump([dev_sentence1, dev_sentence2, dev_labels], open("cache/dev.p", "wb"))
else:
    logging.info("Loading data from the cache.")
    vocab = pickle.load(open("cache/vocab.p", "rb"))
    train_sentence1, train_sentence2, train_labels = pickle.load(open("cache/train.p", "rb"))
    dev_sentence1, dev_sentence2, dev_labels = pickle.load(open("cache/dev.p", "rb"))
    test_sentence1, test_sentence2, test_labels = pickle.load(open("cache/test.p", "rb"))

label_vocab = Vocab(train_labels, label=True)

vocab_size = len(vocab)
logging.info("Vocab size: " + str(vocab_size))
num_classes = len(label_vocab)
logging.info("Number of Classes: " + str(num_classes))
embedding_size = 100
logging.info("Embedding size: " + str(embedding_size))
layer_size = 200
logging.info("Layer Size: " + str(layer_size))

model = dy.ParameterCollection()
trainer = dy.AdamTrainer(model)

EMBEDDING_MATRIX = model.add_lookup_parameters((vocab_size, embedding_size))

transform_w1 = model.add_parameters((layer_size, embedding_size))
transform_b1 = model.add_parameters(layer_size)
transform_w2 = model.add_parameters((layer_size, layer_size))
transform_b2 = model.add_parameters(layer_size)

pair_w1 = model.add_parameters((layer_size, layer_size * 2))
pair_b1 = model.add_parameters(layer_size)
pair_w2 = model.add_parameters((layer_size, layer_size))
pair_b2 = model.add_parameters(layer_size)

decide_w1 = model.add_parameters((layer_size, layer_size * 2))
decide_b1 = model.add_parameters(layer_size)
decide_w2 = model.add_parameters((num_classes, layer_size))
decide_b2 = model.add_parameters(num_classes)


def transform(sentence):
    w1 = dy.parameter(transform_w1)
    b1 = dy.parameter(transform_b1)
    w2 = dy.parameter(transform_w2)
    b2 = dy.parameter(transform_b2)

    sentence_transformed = dy.colwise_add(w1 * sentence, b1)
    sentence_transformed = dy.colwise_add(w2 * sentence_transformed, b2)

    return sentence_transformed


def decomposable_attention(sentence_a, sentence_b):
    similarity_scores = dy.transpose(sentence_a) * sentence_b
    logging.info("Similarity Matrix size: " + str(similarity_scores.dim()))

    sentence_a_softmax = dy.softmax(similarity_scores)
    logging.info("Sentence a softmax size: " + str(sentence_a_softmax.dim()))
    sentence_b_softmax = dy.softmax(dy.transpose(similarity_scores))
    logging.info("Sentence b softmax size: " + str(sentence_b_softmax.dim()))

    sentence_b_attended = sentence_b * dy.transpose(sentence_a_softmax)
    sentence_a_attended = sentence_a * dy.transpose(sentence_b_softmax)

    return sentence_a_attended, sentence_b_attended


def pair(sentence, sentence_other_attended):
    w1 = dy.parameter(pair_w1)
    b1 = dy.parameter(pair_b1)
    w2 = dy.parameter(pair_w2)
    b2 = dy.parameter(pair_b2)

    sentence_pair = dy.concatenate(
        [sentence, sentence_other_attended], d=0
    )
    logging.info("Sentence paired with Attended shape: " + str(sentence_pair.dim()))

    pair_transformed = dy.colwise_add(w1 * sentence_pair, b1)
    pair_transformed = dy.colwise_add(w2 * pair_transformed, b2)

    return pair_transformed


def decide(sentence_a, sentence_b):
    w1 = dy.parameter(decide_w1)
    b1 = dy.parameter(decide_b1)
    w2 = dy.parameter(decide_w2)
    b2 = dy.parameter(decide_b2)

    combined = dy.concatenate([sentence_a, sentence_b])
    logging.info("Combined representations shape: " + str(combined.dim()))

    x = (w1 * combined) + b1
    logits = (w2 * x) + b2

    return logits


def calc_loss(sentence_a, sentence_b, label):

    logging.info("la: " + str(len(sentence1)))
    logging.info("lb: " + str(len(sentence2)))

    sentence_a_embedded = [EMBEDDING_MATRIX[vocab.get(w)] for w in sentence_a]
    sentence_a_embedded = dy.concatenate(sentence_a_embedded, d=1)
    logging.info("Sentence a embedded shape: " + str(sentence_a_embedded.dim()))

    sentence_b_embedded = [EMBEDDING_MATRIX[vocab.get(w)] for w in sentence_b]
    sentence_b_embedded = dy.concatenate(sentence_b_embedded, d=1)
    logging.info("Sentence b embedded shape: " + str(sentence_b_embedded.dim()))

    sentence_a_transformed = transform(sentence_a_embedded)
    logging.info("Sentence a transformed shape: " + str(sentence_a_transformed.dim()))
    sentence_b_transformed = transform(sentence_b_embedded)
    logging.info("Sentence b transformed shape: " + str(sentence_b_transformed.dim()))

    sentence_a_attended, sentence_b_attended = decomposable_attention(
        sentence_a_transformed,
        sentence_b_transformed
    )
    logging.info("Sentence a attended shape: " + str(sentence_a_attended.dim()))
    logging.info("Sentence b attended shape: " + str(sentence_b_attended.dim()))

    sentence_a_pair = pair(sentence_a_transformed, sentence_b_attended)
    sentence_b_pair = pair(sentence_b_transformed, sentence_a_attended)
    logging.info("Sentence and attention transformed shape: " + str(sentence_a_pair.dim()))
    logging.info("Sentence and attention transformed shape: " + str(sentence_b_pair.dim()))

    sentence_a = dy.sum_dim(sentence_a_pair, [1])
    logging.info("Sentence a reduction shape: " + str(sentence_a.dim()))
    sentence_b = dy.sum_dim(sentence_b_pair, [1])
    logging.info("Sentence b reduction shape: " + str(sentence_b.dim()))

    logits = decide(sentence_a, sentence_b)
    logging.info("Logits shape: " + str(logits.dim()))

    encoded_label = label_vocab.get(label)

    loss = dy.pickneglogsoftmax(logits, encoded_label)
    return loss


if __name__ == "__main__":
    num_epochs = 5
    train_sentences = 0
    train_loss = 0
    batch_size = 64
    for epoch in range(num_epochs):
        train_sentence1, train_sentence2, train_labels = shuffle(
            train_sentence1,
            train_sentence2,
            train_labels
        )
        offset = 0
        i = 1
        while offset < len(train_sentence1):
            losses = []
            dy.renew_cg()
            sent1_batch = train_sentence1[offset:offset + batch_size]
            sent2_batch = train_sentence2[offset:offset + batch_size]
            label_batch = train_labels[offset:offset + batch_size]
            for sentence1, sentence2, label in zip(sent1_batch, sent2_batch, label_batch):
                loss_exp = calc_loss(sentence1, sentence2, label)
                losses.append(loss_exp)

            batch_loss = dy.esum(losses) / batch_size
            train_loss += batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()
            offset += batch_size
            i += 1
            if i % (500 // batch_size) == 0:
                trainer.status()
                print(train_loss / (500 // batch_size))
                train_loss = 0
            if i % (10000 // batch_size) == 0 or i // batch_size == len(train_sentence1) - 1:
                dev_loss = 0
                dev_losses = []
                dy.renew_cg()
                for dev_1, dev_2, dev_label in zip(dev_sentence1, dev_sentence2, dev_labels):
                    dev_loss_exp = calc_loss(dev_1, dev_2, dev_label)
                    dev_losses.append(dev_loss_exp)
                total_dev_loss = dy.esum(dev_losses) / len(dev_sentence1)
                dev_loss = total_dev_loss.scalar_value()
                print("Dev Loss: {:.4f}".format(dev_loss))

        print("Epoch {} finished.".format(epoch + 1))
    test_loss = 0
    test_sentences = 0
    test_losses = []
    dy.renew_cg()
    for test_1, test_2, test_label in zip(test_sentence1, test_sentence2, test_labels):
        loss_exp = calc_loss(test_1, test_2, test_label)
        test_losses.append(loss_exp)
    total_test_loss = dy.esum(test_losses) / len(test_sentence1)
    test_loss = total_test_loss.scalar_value()
    print("Test Loss: {:.4f}".format(test_loss))
