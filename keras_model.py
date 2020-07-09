import os, sys, argparse
import pandas
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_val_score
from immunomodeling_dataset import ScoreDataset

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--quiet', action='store_true', dest='quiet', required=False, default=False, help='')
parser.add_argument('--train', action='store', dest='train_file', required=True, help='Training data in csv format with binary immunogenicity in column 2 and input features in columns 3:end')
parser.add_argument('--test', action='store', dest='test_file', required=True, help='Validation/test data in csv format with binary immunogenicity in column 2 and input features in columns 3:end')
parser.add_argument('-n', '--hidden_size', action='store', dest='h_nodes', required=False, default='5', help='Number(s) of nodes in the hidden layer to train with. If more than one, argument should be a comma-separated list with no whitespace.')
#parser.add_argument('-', '--', action='', dest='', required=, default='', help='')
args = vars(parser.parse_args())

if args['quiet']: 
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
else:
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

trdata = pandas.read_csv(args['train_file'])
tr_X = trdata.iloc[:, 3:].values
tr_Y = trdata.iloc[:, 2].values

tedata = pandas.read_csv(args['test_file'])
te_X = tedata.iloc[:, 3:].values
te_Y = tedata.iloc[:, 2].values

assert tr_X.shape[1] == te_X.shape[1], "Training and test data do not have the same number of features!"

train_data = ScoreDataset(tr_X, tr_Y, oversample=True)
test_data = ScoreDataset(te_X, te_Y, oversample=False)

def buildmodel(n_hidden):
	model = tf.keras.models.Sequential([tf.keras.layers.Dense(n_hidden, input_shape=(tr_X.shape[1],),activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.keras.initializers.glorot_normal),
	tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True)])
	model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
	return(model)

h_nodes = [int(x) for x in args['h_nodes'].split(',')]

training_perf = []
for hidden_size in h_nodes:
	print(f'Hidden layer size: {hidden_size}')
	cvscores = []
	i = 0
	for train_X, train_Y, val_X, val_Y in train_data.get_training_batch():
		model = None
		model = buildmodel(hidden_size)
		model.fit(train_X, train_Y, epochs=500, batch_size=64, callbacks=[callback], shuffle=True, verbose=0, validation_data=(val_X, val_Y))
		print(f'Leaf {i+1} '+', '.join([f'{model.metrics_names[i]}:{scores[i]:2f}' for i in range(len(scores))]))
		cvscores.append(scores)
		i += 1
	performance = np.mean(cvscores, axis=0)
	error = np.std(cvscores, axis=0)
	print(', '.join([f'{model.metrics_names[i]}: {performance[i]:2f} +/- {error[i]:2f}' for i in range(len(performance))]))


