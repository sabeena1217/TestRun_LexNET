import codecs
import itertools

import numpy as np

np.random.seed(133)


def get_id(corpus, key):
    """
    Get the corpus ID of a word
    (handle utf-8 encoding errors, following change
    https://github.com/vered1986/LexNET/pull/2 from @gossebouma)
    :param corpus: the corpus' resource object
    :param key: the word
    :return: the ID of the word or -1 if not found
    """
    id = -1  # the index of the unknown word
    try:
        id = corpus.get_id_by_term(key.encode('utf-8'))
    except UnicodeEncodeError:
        pass
    return id


def vectorize_path(path, lemma_index, pos_index, dep_index, dir_index):
    """
    Return a vector representation of the path
    :param path: the string representation of a path
    :param lemma_index: index to lemma dictionary
    :param pos_index: index to POS dictionary
    :param dep_index: index to dependency label dictionary
    :param dir_index: index to edge direction dictionary
    :return:
    """
    path_edges = [vectorize_edge(edge, lemma_index, pos_index, dep_index, dir_index) for edge in path.split('_')]

    if None in path_edges:
        return None

    return tuple(path_edges)


def vectorize_edge(edge, lemma_index, pos_index, dep_index, dir_index):
    """
    Return a vector representation of the edge: concatenate lemma/pos/dep and add direction symbols
    :param edge: the string representation of an edge
    :param lemma_index: index to lemma dictionary
    :param pos_index: index to POS dictionary
    :param dep_index: index to dependency label dictionary
    :param dir_index: index to edge direction dictionary
    :return:
    """
    try:
        lemma, pos, dep, direction = edge.split('/')
        lemma, pos, dep, direction = lemma_index.get(lemma, 0), pos_index[pos], dep_index[dep], dir_index[direction]
    except:
        return None

    return tuple([lemma, pos, dep, direction])


def reconstruct_edge((lemma, pos, dep, direction),
                     lemma_inverted_index, pos_inverted_index, dep_inverted_index, dir_inverted_index):
    """
    Return a string representation of the edge
    :param lemma_inverted_index: lemma to index dictionary
    :param pos_inverted_index: POS to index dictionary
    :param dep_inverted_index: dependency label to index dictionary
    :param dir_inverted_index: edge direction to index dictionary
    :return: The string representation of the edge
    """
    edge = '/'.join([lemma_inverted_index[lemma], pos_inverted_index[pos], dep_inverted_index[dep],
                     dir_inverted_index[direction]])
    return edge


def load_embeddings(file_name, vocabulary):
    """
    Load the pre-trained embeddings from a file
    :param file_name: the embeddings file
    :param vocabulary: limited vocabulary to load vectors for
    :return: the vocabulary and the word vectors
    """
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        words, vectors = zip(*[line.strip().split(' ', 1) for line in f_in])
    wv = np.loadtxt(vectors)



    # # setting the max parameter to 1, will return a list with 2 elements!
    # txt = "apple#banana#cherry#orange"
    # x = txt.split("#", 1)
    # ['apple', 'banana#cherry#orange']

    # a = ("John", "Charles", "Mike")
    # b = ("Jenny", "Christy", "Monica", "Vicky")
    # x = zip(a, b)
    # print(tuple(x))
    # (('John', 'Jenny'), ('Charles', 'Christy'), ('Mike', 'Monica'))


    # Add the unknown words
    unknown_vector = np.random.random_sample((wv.shape[1],))
    print("unknown_vector")
    print(unknown_vector)

    # >>> np.random.random_sample()
    # 0.47108547995356098
    # Return random floats in the half-open interval [0.0, 1.0).

    # >>> 5 * np.random.random_sample((3, 2)) - 5
    # array([[-3.99149989, -0.52338984],
    #        [-2.99091858, -0.79479508],
    #        [-1.23204345, -1.75224494]])
    # Three-by-two array of random numbers from [-5, 0):


    word_set = set(words)
    # A set, in Python, is just like the mathematical set.
    # It does not hold duplicate values, and is unordered.
    # tuple is immutable, set is mutable.


    unknown_words = list(set(vocabulary).difference(set(words)))
    print("unknown_words")
    print(unknown_words)

    # Create vectors for MWEs - sum of word embeddings, and OOV words
    unknown_word_vectors = [np.add.reduce([wv[words.index(w)] if w in word_set else unknown_vector
                                           for w in word.split(' ')])
                            for word in unknown_words]
    print("unknown_word_vectors")
    print(unknown_word_vectors)

    wv = np.vstack((wv, unknown_word_vectors)) #error
    # ValueError: all the input array dimensions except for the concatenation axis must match exactly
    words = list(words) + unknown_words

    # Stack arrays in sequence vertically (row wise).
    # >>> a = np.array([[1], [2], [3]])
    # >>> b = np.array([[2], [3], [4]])
    # >>> np.vstack((a,b))
    # array([[1],
    #        [2],
    #        [3],
    #        [2],
    #        [3],
    #        [4]])

    print 'Known lemmas:', len(vocabulary) - len(unknown_words), '/', len(vocabulary)

    # Normalize each row (word vector) in the matrix to sum-up to 1
    row_norm = np.sum(np.abs(wv) ** 2, axis=-1) ** (1. / 2)
    wv /= row_norm[:, np.newaxis]

    word_index = {w: i for i, w in enumerate(words)}

    return wv, word_index


def load_dataset(dataset_file, relations):
    """
    Loads a dataset file
    :param dataset_file: the file path
    :return: a list of dataset instances, (x, y, relation)
    """
    with codecs.open(dataset_file, 'r', 'utf-8') as f_in:
        dataset = [tuple(line.strip().split('\t')) for line in f_in]
        print("dataset", dataset)
        dataset = {(x.lower(), y.lower()): relation for (x, y, relation) in dataset if relation in relations}

    return dataset


def unique(lst):
    """
    :param lst: a list of lists
    :return: a unique list of items appearing in those lists
    """
    indices = sorted(range(len(lst)), key=lst.__getitem__)
    indices = set(next(it) for k, it in
                  itertools.groupby(indices, key=lst.__getitem__))
    return [x for i, x in enumerate(lst) if i in indices]


def get_paths(corpus, x, y):
    """
    Get the paths that connect x and y in the corpus
    :param corpus: the corpus' resource object
    :param x: x
    :param y: y
    :return:
    """
    x_to_y_paths = corpus.get_relations(x, y)
    print (x,y)
    print ("x_to_y_paths: ")
    print (x_to_y_paths)

    y_to_x_paths = corpus.get_relations(y, x)
    print ("y_to_x_paths: ")
    print (y_to_x_paths)

    paths = {corpus.get_path_by_id(path): count for (path, count) in x_to_y_paths.iteritems()}
    print("paths b4 update: ")
    print(paths)

    paths.update({corpus.get_path_by_id(path).replace('X/', '@@@').replace('Y/', 'X/').replace('@@@', 'Y/'): count
                  for (path, count) in y_to_x_paths.iteritems()})
    print("paths after update: ")
    print(paths)

    return paths
