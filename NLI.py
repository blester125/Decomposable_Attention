"""Natural Language Entailment using https://aclweb.org/anthology/D16-1244."""

# [ Imports ]
# [ -Python ]
import os
import json
import pickle
import logging
from itertools import chain
from collections import Counter
# [ -Third Party ]
from sklearn.utils import shuffle
import numpy as np
import dynet as dy
# [ -Project ]
from utils import Vocab, read, parse_args


args = parse_args()

logging.basicConfig(level=args.log_level)

train_data = "snli_1.0/snli_1.0_train.jsonl"
dev_data = "snli_1.0/snli_1.0_dev.jsonl"
test_data = "snli_1.0/snli_1.0_test.jsonl"

if args.no_cache or not os.path.exists("cache"):
    logging.info("Cache not found, reprocessing data.")
    train_sentence1, train_sentence2, train_labels = read(train_data, 550152)
    dev_sentence1, dev_sentence2, dev_labels = read(dev_data, 10000)
    test_sentence1, test_sentence2, test_labels = read(test_data, 10000)
    raw_data = chain(*train_sentence1, *train_sentence2)
    vocab = Vocab(raw_data)
    if not os.path.exists("cache"):
        os.makedirs("cache")
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
embedding_size = args.embedding_size
logging.info("Embedding size: " + str(embedding_size))
layer_size = args.layer_size
logging.info("Layer Size: " + str(layer_size))
batch_size = args.batch_size
logging.info("Batch Size: " + str(batch_size))
num_epochs = args.num_epochs
logging.info("Num Epochs: " + str(num_epochs))

model = dy.ParameterCollection()
trainer = dy.AdamTrainer(model)

EMBEDDING_MATRIX = model.add_lookup_parameters((vocab_size, embedding_size))

transform_w1 = model.add_parameters((layer_size, embedding_size))
transform_b1 = model.add_parameters(layer_size)
transform_w2 = model.add_parameters((layer_size, layer_size))
transform_b2 = model.add_parameters(layer_size)

combine_w1 = model.add_parameters((layer_size, layer_size * 2))
combine_b1 = model.add_parameters(layer_size)
combine_w2 = model.add_parameters((layer_size, layer_size))
combine_b2 = model.add_parameters(layer_size)

decide_w1 = model.add_parameters((layer_size, layer_size * 2))
decide_b1 = model.add_parameters(layer_size)
decide_w2 = model.add_parameters((num_classes, layer_size))
decide_b2 = model.add_parameters(num_classes)


def embed(sentence):
    sentence_embedded = [EMBEDDING_MATRIX[vocab[w]] for w in sentence]
    sentence_embedded = dy.concatenate(sentence_embedded, d=1)
    return sentence_embedded


def transform(sentence):
    w1 = dy.parameter(transform_w1)
    b1 = dy.parameter(transform_b1)
    w2 = dy.parameter(transform_w2)
    b2 = dy.parameter(transform_b2)

    sentence_transformed = dy.colwise_add(w1 * sentence, b1)
    sentence_transformed = dy.rectify(sentence_transformed)
    sentence_transformed = dy.colwise_add(w2 * sentence_transformed, b2)
    sentence_transformed = dy.rectify(sentence_transformed)

    return sentence_transformed


def attend(sentence_a, sentence_b):
    similarity_scores = dy.transpose(sentence_a) * sentence_b
    logging.debug("Similarity Matrix size: " + str(similarity_scores.dim()))

    sentence_a_softmax = dy.softmax(similarity_scores)
    logging.debug("Sentence a softmax size: " + str(sentence_a_softmax.dim()))
    sentence_b_softmax = dy.softmax(dy.transpose(similarity_scores))
    logging.debug("Sentence b softmax size: " + str(sentence_b_softmax.dim()))

    sentence_b_attended = sentence_b * dy.transpose(sentence_a_softmax)
    sentence_a_attended = sentence_a * dy.transpose(sentence_b_softmax)

    return sentence_a_attended, sentence_b_attended


def combine(sentence, sentence_other_attended):
    w1 = dy.parameter(combine_w1)
    b1 = dy.parameter(combine_b1)
    w2 = dy.parameter(combine_w2)
    b2 = dy.parameter(combine_b2)

    sentence_combine = dy.concatenate(
        [sentence, sentence_other_attended], d=0
    )
    logging.debug("Sentence combined with Attended shape: " + str(sentence_combine.dim()))

    combine_transformed = dy.colwise_add(w1 * sentence_combine, b1)
    combine_transformed = dy.rectify(combine_transformed)
    combine_transformed = dy.colwise_add(w2 * combine_transformed, b2)
    combine_transformed = dy.rectify(combine_transformed)

    return combine_transformed


def aggregate(sentence_a, sentence_b):
    w1 = dy.parameter(decide_w1)
    b1 = dy.parameter(decide_b1)
    w2 = dy.parameter(decide_w2)
    b2 = dy.parameter(decide_b2)

    sentence_a = dy.sum_dim(sentence_a, [1])
    logging.debug("Sentence a reduction shape: " + str(sentence_a.dim()))
    sentence_b = dy.sum_dim(sentence_b, [1])
    logging.debug("Sentence b reduction shape: " + str(sentence_b.dim()))

    combined = dy.concatenate([sentence_a, sentence_b])
    logging.debug("Combined representations shape: " + str(combined.dim()))

    x = (w1 * combined) + b1
    x = dy.rectify(x)
    logits = (w2 * x) + b2

    return logits


def forward_prop(sentence_a, sentence_b):

    logging.debug("la: " + str(len(sentence1)))
    logging.debug("lb: " + str(len(sentence2)))

    sentence_a_embedded = embed(sentence_a)
    logging.debug("Sentence a embedded shape: " + str(sentence_a_embedded.dim()))

    sentence_b_embedded = embed(sentence_b)
    logging.debug("Sentence b embedded shape: " + str(sentence_b_embedded.dim()))

    sentence_a_transformed = transform(sentence_a_embedded)
    logging.debug("Sentence a transformed shape: " + str(sentence_a_transformed.dim()))
    sentence_b_transformed = transform(sentence_b_embedded)
    logging.debug("Sentence b transformed shape: " + str(sentence_b_transformed.dim()))

    sentence_a_attended, sentence_b_attended = attend(
        sentence_a_transformed,
        sentence_b_transformed
    )
    logging.debug("Sentence a attended shape: " + str(sentence_a_attended.dim()))
    logging.debug("Sentence b attended shape: " + str(sentence_b_attended.dim()))

    sentence_a_combine = combine(sentence_a_transformed, sentence_b_attended)
    sentence_b_combine = combine(sentence_b_transformed, sentence_a_attended)
    logging.debug("Sentence and attention transformed shape: " + str(sentence_a_combine.dim()))
    logging.debug("Sentence and attention transformed shape: " + str(sentence_b_combine.dim()))

    logits = aggregate(sentence_a_combine, sentence_b_combine)
    logging.debug("Logits shape: " + str(logits.dim()))

    return logits


def calc_loss(sentence_a, sentence_b, label):
    logits = forward_prop(sentence_a, sentence_b)
    encoded_label = label_vocab[label]
    loss = dy.pickneglogsoftmax(logits, encoded_label)
    return loss


if __name__ == "__main__":
    train_sentences = 0
    train_loss = 0
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
    preds_exp = []
    dy.renew_cg()
    for test_1, test_2, test_label in zip(test_sentence1, test_sentence2, test_labels):
        preds_exp.append(forward_prop(test_1, test_2))
    preds = dy.concatenate(preds_exp, d=1).npvalue()
    preds = np.argmax(preds, axis=0)
    correct = 0
    total = 0
    for pred, label in zip(preds, test_labels):
        if pred == label_vocab[label]:
            correct += 1
        total += 1
    print("Test Accuracy: {:.2f}".format((correct / total) * 100))
