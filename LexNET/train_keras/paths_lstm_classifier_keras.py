from keras.layers import Dense, Input, Sequential
from keras.layers import Conv2D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
import numpy as np

from lstm_common import *
from sklearn.base import BaseEstimator

NUM_LAYERS = 2
LSTM_HIDDEN_DIM = 60
LEMMA_DIM = 50
POS_DIM = 4
DEP_DIM = 5
DIR_DIM = 1

EMPTY_PATH = ((0, 0, 0, 0),)
LOSS_EPSILON = 0.0 # 0.01
MINIBATCH_SIZE = 100



class PathLSTMClassifierKeras:
    def __init__(self, num_lemmas, num_pos, num_dep, num_directions=5, n_epochs=10, num_relations=2,
                 alpha=0.01, lemma_embeddings=None, dropout=0.0, use_xy_embeddings=False, num_hidden_layers=0):

        """'
                Initialize the LSTM
                :param num_lemmas Number of distinct lemmas
                :param num_pos Number of distinct part of speech tags
                :param num_dep Number of distinct depenedency labels
                :param num_directions Number of distinct path directions (e.g. >,<)
                :param n_epochs Number of training epochs
                :param num_relations Number of classes (e.g. binary = 2)
                :param alpha Learning rate
                :param lemma_embeddings Pre-trained word embedding vectors for the path-based component
                :param dropout Dropout rate
                :param use_xy_embeddings Whether to concatenate x and y word embeddings to the network input
                :param num_hidden_layers The number of hidden layers for the term-pair classification network
                """
        self.n_epochs = n_epochs
        self.num_lemmas = num_lemmas
        self.num_pos = num_pos
        self.num_dep = num_dep
        self.num_directions = num_directions
        self.num_relations = num_relations
        self.alpha = alpha
        self.dropout = dropout
        self.use_xy_embeddings = use_xy_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.update = True

        self.lemma_vectors = None
        if lemma_embeddings is not None:
            self.lemma_vectors = lemma_embeddings
            self.lemma_embeddings_dim = lemma_embeddings.shape[1]
        else:
            self.lemma_embeddings_dim = LEMMA_DIM

        # Create the network
        print 'Creating the network...'
        self.builder, self.model, self.model_parameters = create_computation_graph(self.num_lemmas, self.num_pos,
                                                                                   self.num_dep, self.num_directions,
                                                                                   self.num_relations,
                                                                                   self.lemma_vectors,
                                                                                   use_xy_embeddings,
                                                                                   self.num_hidden_layers,
                                                                                   self.lemma_embeddings_dim)
        print 'Done!'

    def fit(self.model, X_train, y_train, x_y_vectors=None):
        """
        Train the model
        """
        print 'Training the model...'
        # train(self.builder, self.model, self.model_parameters, X_train, y_train, self.n_epochs, self.alpha, self.update,
        #       self.dropout, x_y_vectors, self.num_hidden_layers)

        model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=3, batch_size=64)
        print 'Done!'


def create_computation_graph(num_lemmas, num_pos, num_dep, num_directions, num_relations,
                             wv=None, use_xy_embeddings=False, num_hidden_layers=0, lemma_dimension=50):
    """
    Initialize the model
    :param num_lemmas Number of distinct lemmas
    :param num_pos Number of distinct part of speech tags
    :param num_dep Number of distinct depenedency labels
    :param num_directions Number of distinct path directions (e.g. >,<)
    :param num_relations Number of classes (e.g. binary = 2)
    :param wv Pre-trained word embeddings file
    :param use_xy_embeddings Whether to concatenate x and y word embeddings to the network input
    :param num_hidden_layers The number of hidden layers for the term-pair classification network
    :param lemma_dimension The dimension of the lemma embeddings
    :return:
    """
    embeddings_index = dict()
    f = open('glove.6B/glove.6B.50d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
vocabulary_size = 5000;
    embedding_matrix = np.zeros((vocabulary_size, 100))
    for word, index in tokenizer.word_index.items():
        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector


X_train, y_train, X_test, y_test, W, W2 = mr_data.load_data(fold=0)
maxlen = X_train.shape[1]
max_features = len(W)
embedding_dims = len(W[0])

print('Train...')
accs = []
first_run = True
for i in xrange(folds):
    X_train, y_train, X_test, y_test, W, W2 = mr_data.load_data(fold=i)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    rand_idx = np.random.permutation(range(len(X_train)))
    X_train = X_train[rand_idx]
    y_train = y_train[rand_idx]

    def build_model():
        print('Build model...%d of %d' % (i + 1, folds))
        main_input = Input(shape=(maxlen, ), dtype='int32', name='main_input')
        embedding  = Embedding(max_features, embedding_dims,
                      weights=[np.matrix(W)], input_length=maxlen,
                      name='embedding')(main_input)

        embedding = Dropout(0.50)(embedding)


        conv = Convolution2D(nb_filter=nb_filter,
                              filter_length=2,
                              border_mode='valid',
                              activation='relu',
                              subsample_length=1,
                              name='conv4')(embedding)

        pool = MaxPooling1D(pool_length=2,
                                name='maxConv5')(conv5)

        x = merge([conv, pool], mode='concat')

        x = Dropout(0.15)(x)

        x = RNN(rnn_output_size)(x)

        x = Dense(hidden_dims, activation='relu', init='he_normal',
                  W_constraint = maxnorm(3), b_constraint=maxnorm(3),
                  name='mlp')(x)

        x = Dropout(0.10, name='drop')(x)

        output = Dense(1, init='he_normal',
                       activation='sigmoid', name='output')(x)

        model = Model(input=main_input, output=output)
        model.compile(loss={'output':'binary_crossentropy'},
                    optimizer=Adadelta(lr=0.95, epsilon=1e-06),
                    metrics=["accuracy"])
        return model

    model = build_model()
    if first_run:
        first_run = False
        print(model.summary())

    best_val_acc = 0
    best_test_acc = 0
    for j in xrange(nb_epoch):
        a = time.time()
        his = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        validation_split=0.1,
                        shuffle=True,
                        nb_epoch=1, verbose=0)
        print('Fold %d/%d Epoch %d/%d\t%s' % (i + 1,
                                          folds, j + 1, nb_epoch, str(his.history)))
        if his.history['val_acc'][0] >= best_val_acc:
            score, acc = model.evaluate(X_test, y_test,
                                        batch_size=batch_size,
                                        verbose=2)
            best_val_acc = his.history['val_acc'][0]
            best_test_acc = acc
            print('Got best epoch  best val acc is %f test acc is %f' %
                  (best_val_acc, best_test_acc))
            if len(accs) > 0:
                print('Current avg test acc:', str(np.mean(accs)))
        b = time.time()
        cost = b - a
        left = (nb_epoch - j - 1) + nb_epoch * (folds - i - 1)
        print('One round cost %ds, %d round %ds %dmin left' % (cost, left,
                                                               cost * left,
                                                               cost * left / 60.0))
    accs.append(best_test_acc)
    print('Avg test acc:', str(np.mean(accs)))
    print("Bidirectional LSTM")
    model.summary()

    # model = Model() -- gives error? tried to fix by looking at dynet tutorial examples -- GB
    dy.renew_cg()
    # Renew the computation graph.
    # Call this before building any new computation graph

    model = dy.ParameterCollection()
    # ParameterCollection to hold the parameters

    network_input = LSTM_HIDDEN_DIM

    builder = dy.LSTMBuilder(NUM_LAYERS, lemma_dimension + POS_DIM + DEP_DIM + DIR_DIM, network_input, model)

    # Concatenate x and y
    if use_xy_embeddings:
        network_input += 2 * lemma_dimension

    #  'the optimal size of the hidden layer is usually between the size of the input and size of the output layers'
    hidden_dim = int((network_input + num_relations) / 2)

    model_parameters = {}

    if num_hidden_layers == 0:
        # model_parameters['W_cnn'] = model.add_parameters((1, WIN_SIZE, EMB_SIZE, FILTER_SIZE))  # cnn weights
        # model_parameters['b_cnn'] = model.add_parameters((FILTER_SIZE))  # cnn bias

        model_parameters['W1'] = model.add_parameters((num_relations, network_input))
        model_parameters['b1'] = model.add_parameters((num_relations, 1))
    # A ParameterCollection is a container for Parameters and LookupParameters.
    # dynet.Trainer objects take ParameterCollection objects that define which parameters are being trained.

    elif num_hidden_layers == 1:

        model_parameters['W1'] = model.add_parameters((hidden_dim, network_input))
        model_parameters['b1'] = model.add_parameters((hidden_dim, 1))
        model_parameters['W2'] = model.add_parameters((num_relations, hidden_dim))
        model_parameters['b2'] = model.add_parameters((num_relations, 1))

    else:
        raise ValueError('Only 0 or 1 hidden layers are supported')

    model_parameters['lemma_lookup'] = model.add_lookup_parameters((num_lemmas, lemma_dimension))
    #LookupParameters represents a table of parameters.
    # They are used to embed a set of discrete objects (e.g. word embeddings). These are sparsely updated.


    # Pre-trained word embeddings
    if wv is not None:
        model_parameters['lemma_lookup'].init_from_array(wv)

    model_parameters['pos_lookup'] = model.add_lookup_parameters((num_pos, POS_DIM))
    model_parameters['dep_lookup'] = model.add_lookup_parameters((num_dep, DEP_DIM))
    model_parameters['dir_lookup'] = model.add_lookup_parameters((num_directions, DIR_DIM))

    return builder, model, model_parameters

def process_one_instance(keras.model model):