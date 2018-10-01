import sys
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-g', '--gpus', help='number of gpus to use [0,1], default=0', type=int, default=0, choices=[0,1])
ap.add_argument('-m', '--memory', help='set dynet memory, default 8192',  default=8192)
ap.add_argument('-s', '--seed', help='dynet random seed, pick any integer you like, default=3016748844', default=3016748844)
ap.add_argument('--num_hidden_layers', help='number of hidden layers to use', type=int, default=0)
ap.add_argument('--num_epochs', help='number of epochs to train', type=int, default=5)
ap.add_argument('corpus_prefix', help='path to the corpus resource')
ap.add_argument('dataset_prefix', help='path to the train/test/val/rel data')
ap.add_argument('model_prefix_file', help='where to store the result')
ap.add_argument('embeddings_file', help='path to word embeddings file')

args = ap.parse_args()

sys.path.append('../common/')

from lstm_common import *
from itertools import count
from evaluation_common import *
from collections import defaultdict
from knowledge_resource import KnowledgeResource
from paths_lstm_classifier import PathLSTMClassifier

EMBEDDINGS_DIM = 50


def main():

    np.random.seed(133)

    # Load the relations
    with codecs.open(args.dataset_prefix + '/relations.txt', 'r', 'utf-8') as f_in:
        relations = [line.strip() for line in f_in]
        relation_index = { relation : i for i, relation in enumerate(relations) }

    # Load the datasets
    print 'Loading the dataset...'
    train_set = load_dataset(args.dataset_prefix + '/train.tsv', relations)
    val_set = load_dataset(args.dataset_prefix + '/val.tsv', relations)
    test_set = load_dataset(args.dataset_prefix + '/test.tsv', relations)
    y_train = [relation_index[label] for label in train_set.values()]
    y_val = [relation_index[label] for label in val_set.values()]
    y_test = [relation_index[label] for label in test_set.values()]
    dataset_keys = train_set.keys() + val_set.keys() + test_set.keys()
    print 'Done!'

    # Load the resource (processed corpus)
    print 'Loading the corpus...'
    corpus = KnowledgeResource(args.corpus_prefix)
    print 'Done!'

    # Get the vocabulary
    vocabulary = get_vocabulary(corpus, dataset_keys)

    # Load the word embeddings
    print 'Initializing word embeddings...'
    word_vectors, lemma_index = load_embeddings(args.embeddings_file, vocabulary)
    lemma_inverted_index = { i : w for w, i in lemma_index.iteritems() }

    # Load the paths and create the feature vectors
    print 'Loading path files...'
    dataset_instances, pos_index, dep_index, dir_index, pos_inverted_index, dep_inverted_index, \
    dir_inverted_index = load_paths(corpus, dataset_keys, lemma_index)
    print 'Number of words %d, number of pos tags: %d, number of dependency labels: %d, number of directions: %d' % \
          (len(lemma_index), len(pos_index), len(dep_index), len(dir_index))

    X_train = dataset_instances[:len(train_set)]
    X_val = dataset_instances[len(train_set):len(train_set)+len(val_set)]
    X_test = dataset_instances[len(train_set)+len(val_set):]

    # Tune the hyper-parameters using the validation set
    alphas = [0.001]
    word_dropout_rates = [0.0, 0.2, 0.4]
    f1_results = []
    models = []
    descriptions = []

    for alpha in alphas:
        for word_dropout_rate in word_dropout_rates:

            # Create the classifier
            classifier = PathLSTMClassifier(num_lemmas=len(lemma_index), num_pos=len(pos_index),
                                            num_dep=len(dep_index), num_directions=len(dir_index),
                                            n_epochs=args.num_epochs,
                                            num_relations=len(relations), lemma_embeddings=word_vectors,
                                            dropout=word_dropout_rate, alpha=alpha, use_xy_embeddings=False,
                                            num_hidden_layers=args.num_hidden_layers)

            print 'Training with learning rate = %f, dropout = %f...' % (alpha, word_dropout_rate)
            classifier.fit(X_train, y_train)

            pred = classifier.predict(X_val)
            precision, recall, f1, support = evaluate(y_val, pred, relations, do_full_reoprt=False)
            print 'Learning rate = %f, dropout = %f, Precision: %.3f, Recall: %.3f, F1: %.3f' % \
                  (alpha, word_dropout_rate, precision, recall, f1)
            f1_results.append(f1)
            models.append(classifier)

            # Save intermediate model
            classifier.save_model(args.model_prefix_file + '.' + str(word_dropout_rate),
                                  [lemma_index, pos_index, dep_index, dir_index])
            descriptions.append('Learning rate = %f, dropout = %f' % (alpha, word_dropout_rate))

    best_index = np.argmax(f1_results)
    classifier = models[best_index]
    description = descriptions[best_index]
    print 'Best hyper-parameters: ' + description

    # Save the best model to a file
    print 'Saving the model...'
    classifier.save_model(args.model_prefix_file, [lemma_index, pos_index, dep_index, dir_index])

    # Evaluate on the test set
    print 'Evaluation:'
    pred = classifier.predict(X_test)
    precision, recall, f1, support = evaluate(y_test, pred, relations, do_full_reoprt=True)
    print 'Precision: %.3f, Recall: %.3f, F1: %.3f' % (precision, recall, f1)

    # Write the predictions to a file
    output_predictions(args.model_prefix_file + '.predictions', relations, pred, test_set.keys(), y_test)

    # Retrieve k-best scoring paths for each class
    all_paths = unique([path for path_list in dataset_instances for path in path_list])
    top_k = classifier.get_top_k_paths(all_paths, relation_index, 0.7)

    for i, relation in enumerate(relations):
        with codecs.open(args.model_prefix_file + '.paths.' + relation, 'w', 'utf-8') as f_out:
            for path, score in top_k[i]:
                path_str = '_'.join([reconstruct_edge(edge, lemma_inverted_index, pos_inverted_index,
                                                      dep_inverted_index, dir_inverted_index) for edge in path])
                print >> f_out, '\t'.join([path_str, str(score)])


def get_vocabulary(corpus, dataset_keys):
    """
    Get all the words in paths
    :param corpus: the corpus object
    :param dataset_keys: the word pairs in the dataset
    :return: a set of distinct words appearing as x or y or in a path
    """
    keys = [(get_id(corpus, x), get_id(corpus, y)) for (x, y) in dataset_keys]
    
    path_lemmas = set([edge.split('/')[0]
                       for (x_id, y_id) in keys
                       for path in get_paths(corpus, x_id, y_id).keys()
                       for edge in path.split('_')])
    print(list(path_lemmas))
    return list(path_lemmas)


def load_paths(corpus, dataset_keys, lemma_index):
    """
    Load the paths for this dataset
    :param corpus: the corpus object
    :param dataset_keys: the word pairs in the dataset
    :return: the index of words for the word embeddings
    """

    # Define the dictionaries
    pos_index = defaultdict(count(0).next)
    dep_index = defaultdict(count(0).next)
    dir_index = defaultdict(count(0).next)

    dummy = pos_index['#NOPATH#']
    dummy = dep_index['#NOPATH#']
    dummy = dir_index['#NOPATH#']

    keys = [(get_id(corpus, x), get_id(corpus, y)) for (x, y) in dataset_keys]
    paths_x_to_y = [{ vectorize_path(path, lemma_index, pos_index, dep_index, dir_index) : count
                      for path, count in get_paths(corpus, x_id, y_id).iteritems() }
                    for (x_id, y_id) in keys]
    paths = [ { p : c for p, c in paths_x_to_y[i].iteritems() if p is not None } for i in range(len(keys)) ]

    empty = [dataset_keys[i] for i, path_list in enumerate(paths) if len(path_list.keys()) == 0]
    print 'Pairs without paths:', len(empty), ', all dataset:', len(dataset_keys)

    pos_inverted_index = { i : p for p, i in pos_index.iteritems() }
    dep_inverted_index = { i : p for p, i in dep_index.iteritems() }
    dir_inverted_index = { i : p for p, i in dir_index.iteritems() }

    return paths, pos_index, dep_index, dir_index, pos_inverted_index, dep_inverted_index, dir_inverted_index


if __name__ == '__main__':
    main()
