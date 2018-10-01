import sys
import argparse
import pprint

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
MAX_PATHS_PER_PAIR = -1 # Set to K > 0 if you want to limit the number of path per pair (for memory reasons)


def main():
    np.random.seed(133)

    # Load the relations
    with codecs.open(args.dataset_prefix + '/relations.txt', 'r', 'utf-8') as f_in:
        relations = [line.strip() for line in f_in]
        relation_index = { relation : i for i, relation in enumerate(relations) }
        print('relation_index :')
        pprint.pprint(relation_index)

    # Load the datasets
    print 'Loading the dataset...'
    train_set = load_dataset(args.dataset_prefix + '/train.tsv', relations)
    val_set = load_dataset(args.dataset_prefix + '/val.tsv', relations)
    test_set = load_dataset(args.dataset_prefix + '/test.tsv', relations)

    print("test_set ", test_set)
    print('\n')

    y_train = [relation_index[label] for label in train_set.values()]
    y_val = [relation_index[label] for label in val_set.values()]
    y_test = [relation_index[label] for label in test_set.values()]

    print("y_test ", y_test)
    print('\n')

    print("test_set.keys() ", test_set.keys())
    dataset_keys = train_set.keys() + val_set.keys() + test_set.keys()

    print 'Done!'
    print('\n')

    # Load the resource (processed corpus)
    print 'Loading the corpus...'
    corpus = KnowledgeResource(args.corpus_prefix)
    print("corpus", corpus)
    print 'Done!'

    # Get the vocabulary
    vocabulary = get_vocabulary(corpus, dataset_keys)

    # Load the word embeddings
    print 'Initializing word embeddings...'
    word_vectors, word_index = load_embeddings(args.embeddings_file, vocabulary)
    # print('word_vectors', word_vectors)
    # print('word_index', word_index)
    # 61, u'anti-terrorism': 77116, u'sugino': 378214, u'v-p': 192698, u"l'association": 183544, u'nicolino': 284460, u'paskal': 134561, u'wons': 344069, u'jianming': 147054, u'paskah': 250517, u'paskai': 389896, u'sivarasa': 206828, u'cutlet': 142874, u'cutler': 18017, u'b\xe9liveau': 270700, u'2213': 275550, u'92.29': 344139, u"l'eglise": 248592, u'farhud': 397788, u'hollen': 59631, u'5150': 153632, u'birthmark': 105028, u'namadi': 291246, u'uebber': 235170, u'holler': 53382, u'holles': 137285, u'libretti': 92423, u'holley': 38966, u'suiciders': 297045, u'leaden': 65314, u'blue-and-white': 199303, u'text-types': 365053, u'csaa':
    # 352872, u'csac': 280893, u'leaded': 55631, u'maranville': 311808, u'reza\xef': 213221, u'csas': 177125, u'csar': 120796, u'csat': 240101, u'haywood': 35298, u'deciliter': 118490, u'yannitsis': 390408, u'thoroughfare': 23669, u'rivonia': 149865, u'ultratop': 133395, u'148-member': 256295, u'kashagan': 100228, u'wesley': 10467, u'matrixx': 207821, u'phylis': 272842, u'fiumicino': 95253, u'pailan': 161723, u'first-of-its-kind': 225091, u'genets': 296893, u'goldwork': 338499, u'neo-babylonian': 195998, u'interventionists': 171769, u'clojure': 303098, u'adhaim': 374678, u'bugle': 41652, u'skaha': 397963, u'aldaco': 396559, u'myisha': 185566, u'frolunda': 195939, u'basilisks': 346164, u'bhatkar': 379085, u'jenine': 184185, u'long-winged': 327407, u'arden-arcade': 341140, u'relaci\xf3n': 267608, u'blockhouses': 114387, u'poove': 367186, u'maize': 14908, u'mojahedin': 224956, u'gerstel': 311743, u'brujas': 128695, u'cracroft': 301781, u'ajna': 377586, u'lederer': 37662, u'100-person': 200702, u'dieguito': 217408, u'quez\xf3n': 302127, u'footwork': 31682, u'kojima': 64072, u'cl\xe9ment': 69390, u'cisowski': 260671, u'jerrys': 247111, u'clampdown': 24779, u'near-fatal': 112364, u'third-world': 163091, u'brashness': 99576, u'adhunik': 398217, u'witherell':
    # 256863, u'chelyabinsk-70': 321375, u'achmat': 110656, u'short-acting': 262297, u'p19': 320735, u'chromate': 136926, u'worrywarts': 340377, u'achmad': 76861, u'semi-nomadic': 109866, u'chalonnaise': 357270, u'flaminius': 202827, u'hirschbiegel': 215738, u'pandyan': 94815, u'd.n.c.': 367506, u'polevanov': 276819, u'spanks': 280699, u'pandyas': 111459, u'morihiro': 103633, u'121.26': 262184, u'spanky': 85069, u'flameouts': 226637, u'15km': 53708, u'50cm': 358620, u'campton': 154447, u'sancton': 349215, u'beguelin': 345069, u'1.218': 398596, u'50cc': 163840, u'hickling': 105165, u'dogtown': 89630, u'league-record': 362391, u'performance-based': 123474, u'guadagno': 185339, u'guadagni': 190095, u'whitesand': 279029, u'rowshan': 393443, u'bruschi': 45649, u'shyatt': 214189, u'profanity-laced': 388047, u'psone': 234041, u'gencon': 353806,
    # u'torpey': 144547, u'daum': 57164, u'gao': 9308, u'ipfw': 203997, u'expands': 14320, u'gam': 17423, u'scratchcards': 348767, u'wassermann': 215028, u'sheinbein': 48976, u'clinopyroxene': 356820, u'sd-6': 241100, u'zinka': 274365})
    word_inverted_index = { i : w for w, i in word_index.iteritems() }

    # Load the paths and create the feature vectors
    print 'Loading path files...'
    x_y_vectors, dataset_instances, pos_index, dep_index, dir_index, pos_inverted_index, dep_inverted_index, \
    dir_inverted_index = load_paths_and_word_vectors(corpus, dataset_keys, word_index)
    print 'Number of words %d, number of pos tags: %d, number of dependency labels: %d, number of directions: %d' % \
          (len(word_index), len(pos_index), len(dep_index), len(dir_index))

    X_train = dataset_instances[:len(train_set)]
    X_val = dataset_instances[len(train_set):len(train_set)+len(val_set)]
    X_test = dataset_instances[len(train_set)+len(val_set):]
    print('X_test', X_test)

    x_y_vectors_train = x_y_vectors[:len(train_set)]
    x_y_vectors_val = x_y_vectors[len(train_set):len(train_set)+len(val_set)]
    x_y_vectors_test = x_y_vectors[len(train_set)+len(val_set):]
    print('x_y_vectors_test', x_y_vectors_test)

    # Tune the hyper-parameters using the validation set
    alphas = [0.001]
    word_dropout_rates = [0.0] # [0.0, 0.2, 0.4]
    f1_results = []
    models = []
    descriptions = []

    for alpha in alphas:
        for word_dropout_rate in word_dropout_rates:

            # Create the classifier
            classifier = PathLSTMClassifier(num_lemmas=len(word_index), num_pos=len(pos_index),
                                            num_dep=len(dep_index), num_directions=len(dir_index),
                                            n_epochs=args.num_epochs,
                                            num_relations=len(relations), lemma_embeddings=word_vectors,
                                            dropout=word_dropout_rate, alpha=alpha, use_xy_embeddings=True,
                                            num_hidden_layers=args.num_hidden_layers)

            print 'Training with learning rate = %f, dropout = %f...' % (alpha, word_dropout_rate)
            classifier.fit(X_train, y_train, x_y_vectors=x_y_vectors_train)

            pred = classifier.predict(X_val, x_y_vectors=x_y_vectors_val)
            precision, recall, f1, support = evaluate(y_val, pred, relations, do_full_reoprt=False)
            print 'Learning rate = %f, dropout = %f, Precision: %.3f, Recall: %.3f, F1: %.3f' % \
                  (alpha, word_dropout_rate, precision, recall, f1)
            f1_results.append(f1)
            models.append(classifier)

            # Save intermediate models
            classifier.save_model(args.model_prefix_file + '.' + str(word_dropout_rate),
                                  [word_index, pos_index, dep_index, dir_index])
            descriptions.append('Learning rate = %f, dropout = %f' % (alpha, word_dropout_rate))

    best_index = np.argmax(f1_results)
    classifier = models[best_index]
    description = descriptions[best_index]
    print 'Best hyper-parameters: ' + description

    # Save the best model to a file
    print 'Saving the model...'
    classifier.save_model(args.model_prefix_file, [word_index, pos_index, dep_index, dir_index])

    # Evaluate on the test set
    print 'Evaluation:'
    pred = classifier.predict(X_test, x_y_vectors=x_y_vectors_test)
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
                path_str = '_'.join([reconstruct_edge(edge, word_inverted_index, pos_inverted_index,
                                                      dep_inverted_index, dir_inverted_index) for edge in path])
                print >> f_out, '\t'.join([path_str, str(score)])


def get_vocabulary(corpus, dataset_keys):
    '''
    Get all the words in the dataset and paths
    :param corpus: the corpus object
    :param dataset_keys: the word pairs in the dataset
    :return: a set of distinct words appearing as x or y or in a path
    '''
    keys = [(get_id(corpus, x), get_id(corpus, y)) for (x, y) in dataset_keys]

    print('keys inside get_vocabulry :')
    print(keys)

    path_lemmas = set([edge.split('/')[0]
                       for (x_id, y_id) in keys
                       for path in get_paths(corpus, x_id, y_id).keys()
                       for edge in path.split('_')
                       if x_id > 0 and y_id > 0])
                       
    x_y_words = set([x for (x, y) in dataset_keys]).union([y for (x, y) in dataset_keys])
    return list(path_lemmas.union(x_y_words))


def load_paths_and_word_vectors(corpus, dataset_keys, lemma_index):
    '''
    Load the paths and the word vectors for this dataset
    :param corpus: the corpus object
    :param dataset_keys: the word pairs in the dataset
    :param word_index: the index of words for the word embeddings
    :return:
    '''

    # Define the dictionaries
    pos_index = defaultdict(count(0).next)
    print('inside load_paths_and_word_vectors method: ')
    print('pos_index: ')
    pprint.pprint(pos_index)
    dep_index = defaultdict(count(0).next)
    dir_index = defaultdict(count(0).next)

    _ = pos_index['#UNKNOWN#']
    _ = dep_index['#UNKNOWN#']
    _ = dir_index['#UNKNOWN#']

    # Vectorize tha paths
    # check for valid utf8 GB
    # keys = [(corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))) for (x, y) in dataset_keys]
    keys = [(get_id(corpus, x), get_id(corpus, y)) for (x, y) in dataset_keys]

    string_paths = [get_paths(corpus, x_id, y_id).items() for (x_id, y_id) in keys]
 
    # Limit number of paths
    if MAX_PATHS_PER_PAIR > 0:
        string_paths = [curr_paths[:MAX_PATHS_PER_PAIR] for curr_paths in string_paths]

    paths_x_to_y = [{ vectorize_path(path, lemma_index, pos_index, dep_index, dir_index) : count
                      for path, count in curr_paths }
                    for curr_paths in string_paths]
    paths = [ { p : c for p, c in paths_x_to_y[i].iteritems() if p is not None } for i in range(len(keys)) ]

    empty = [dataset_keys[i] for i, path_list in enumerate(paths) if len(path_list.keys()) == 0]
    print 'Pairs without paths:', len(empty), ', all dataset:', len(dataset_keys)

    # Get the word embeddings for x and y (get a lemma index)
    print 'Getting word vectors for the terms...'
    x_y_vectors = [(lemma_index.get(x, 0), lemma_index.get(y, 0)) for (x, y) in dataset_keys]

    pos_inverted_index = { i : p for p, i in pos_index.iteritems() }
    dep_inverted_index = { i : p for p, i in dep_index.iteritems() }
    dir_inverted_index = { i : p for p, i in dir_index.iteritems() }

    print 'Done loading corpus data!'

    return x_y_vectors, paths, pos_index, dep_index, dir_index, pos_inverted_index, dep_inverted_index, \
           dir_inverted_index


if __name__ == '__main__':
    main()