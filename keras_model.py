import os, sys, argparse
import pandas
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_curve, auc
from immunomodeling_dataset import ScoreDataset
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--quiet', action='store_true', dest='quiet', required=False, default=False, help='Switch to disable tensorflow warning/log information.')
#parser.add_argument('-i', '--datfile', action='store', dest='data_file', required=True, help='Training+test data in csv format with binary immunogenicity in column index 1 and input features in columns 2:end')
parser.add_argument('-t','--test_size', action='store', dest='test_set_size', required=False, default=0.10, type=float, help='Proportion of total data to use in the test set (default 0.10).')
parser.add_argument('-n', '--hidden_size', action='store', dest='h_nodes', required=False, default=18, type=int, help='Number(s) of nodes in the hidden layer to train with.')
parser.add_argument('-k', '--kfold', action='store', dest='folds', required=False, type=int, default=5, help='Number of folds to use for kfold cross validation (default 5).')
parser.add_argument('-s', '--oversample', action='store', dest='os', required=False, default=False, type=float, help='Desired ratio of minority:majority class. If specified, randomly oversamples training set only to this ratio.')
parser.add_argument('-o', '--outfile', action='store', dest='outfile', required=False, default=None, help='Save ROC AUC plots as an image file e.g. asdf.png')
args = vars(parser.parse_args())

# Whether to print out a LOT of tf output or not.
if args['quiet']:
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	v = 0
else:
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
	v = 1

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for guppy in gpus:
	tf.config.experimental.set_memory_growth(guppy, True)
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# Right now, training is to 3000 epochs. This callback stops training early if the loss to the validation fold does not decrease for 100 consecutive epochs.
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='max', restore_best_weights=True)

def buildmodel(n_in):
	# Function to generate new ANN with the same architecture for training on new leafs.
	model = tf.keras.models.Sequential([tf.keras.layers.Dense(args['h_nodes'], input_shape=(n_in,),activation=tf.nn.leaky_relu, use_bias=True, kernel_initializer=tf.keras.initializers.glorot_normal),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(2, activation=tf.nn.softmax, use_bias=True)])
	model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
	return(model)
	
# Containerized training loop for consistent training/evaluation of different types of data. For now, returns only FPR, TPR, thresholds for training and test data.
def train_anal(train_data, test_data):
	# Stores CV fold performance data (for printing only).
	training_perf = {'binary_crossentropy' : [], 'binary_accuracy' : [], 'auc' : [],
			'val_binary_crossentropy' : [], 'val_binary_accuracy' : [], 'val_auc' : [],
			'test_binary_crossentropy' : [], 'test_binary_accuracy' : [], 'test_auc' : []}

	cvscores = []
	i = 0
	# This loop trains one leaf.
	for train_X, train_Y, val_X, val_Y in train_data.get_training_batch():
		model = None
		model = buildmodel(train_data.X.shape[1])
		model.fit(train_X, train_Y, epochs=3000, batch_size=128, callbacks=[early_stop], shuffle=True, verbose=v, validation_data=(val_X, val_Y))
		# Evaluates performance of model to training leaves.
		train_scores = model.evaluate(train_X, train_Y, verbose=v)
		# Evaluates performance of model to validation leaf.
		val_scores = model.evaluate(val_X, val_Y, verbose=v)
		# Evaluates performance of model to test data not included in any CV training.
		test_scores = model.evaluate(test_data.X, train_data.enc.transform(test_data.Y).toarray(), verbose=v)
		if v:
			# If you aren't running "--quiet"ly, this'll be printed out.
			print(f'\tTraining Leaf {i+1} '+', '.join([f'{model.metrics_names[i]}:{train_scores[i]:.3f}' for i in range(len(train_scores))]))
			print(f'\tValidation Leaf {i+1} '+', '.join([f'{model.metrics_names[i]}:{val_scores[i]:.3f}' for i in range(len(val_scores))]))
			print(f'\tTest set '+', '.join([f'{model.metrics_names[i]}:{test_scores[i]:.3f}' for i in range(len(test_scores))]))
		cvscores.append(train_scores+val_scores+test_scores)
		i += 1
	# Regardless of --quiet, this will print out the final cross-validated performance of training.
	performance = np.mean(cvscores, axis=0)
	error = np.std(cvscores, axis=0)
	
	# Prints out mean ± stdev of kfold cross validation performance to training sets. This is the average performance to the (k-1)/k% of data.
	training_perf['binary_crossentropy'].append(performance[0])
	training_perf['binary_accuracy'].append(performance[1])
	training_perf['auc'].append(performance[2])
	print(', '.join([f'{model.metrics_names[i]}: {performance[i]:.3f} +/- {error[i]:.3f}' for i in (0, 1, 2)]))
	
	# Prints out mean ± stdev of kfold cross validation performance to validation sets. This is the average performance to the 1/k % of data.
	training_perf['val_binary_crossentropy'].append(performance[3])
	training_perf['val_binary_accuracy'].append(performance[4])
	training_perf['val_auc'].append(performance[5])
	print(', '.join([f'val_{model.metrics_names[i]}: {performance[i+3]:.3f} +/- {error[i+3]:.3f}' for i in (0, 1, 2)]))
	
	# Prints out mean ± stdev of kfold cross validation performance to test set.
	training_perf['test_binary_crossentropy'].append(performance[6])
	training_perf['test_binary_accuracy'].append(performance[7])
	training_perf['test_auc'].append(performance[8])
	print(', '.join([f'test_{model.metrics_names[i]}: {performance[i+6]:.3f} +/- {error[i+6]:.3f}' for i in (0, 1, 2)]))

	# Calculation of ROC fpr and tpr to entire training set.
	fpr_tr, tpr_tr, thresholds_tr = roc_curve(train_data.Y, [x[1] for x in model.predict(train_data.X)])
	# Calculation of ROC fpr and tpr to test set.
	fpr_te, tpr_te, thresholds_te = roc_curve(test_data.Y, [x[1] for x in model.predict(test_data.X)])
	return (fpr_tr, tpr_tr, thresholds_tr, fpr_te, tpr_te, thresholds_te)
	
def plot(old_rc, lin_rc, rad_rc, score_rc, linR_rc, radR_rc, scoreR_rc, linRa_rc, radRa_rc, scoreRa_rc):
	# This plots ROC AUC curves using the fpr+tpr ranges from train_anal
	plt.figure()
	plt.subplot(1,2,1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(old_rc[0], old_rc[1], label=f"Old data: {auc(old_rc[0], old_rc[1]):.3f}, {auc(old_rc[3], old_rc[4]):.3f}")
	
	plt.plot(lin_rc[0], lin_rc[1], label=f"linSVR: {auc(lin_rc[0], lin_rc[1]):.3f}, {auc(lin_rc[3], lin_rc[4]):.3f}")
	plt.plot(linR_rc[0], linR_rc[1], label=f"linSVR (RMSF): {auc(linR_rc[0], linR_rc[1]):.3f}, {auc(linR_rc[3], linR_rc[4]):.3f}")
	plt.plot(linRa_rc[0], linRa_rc[1], label=f"linSVR (RMSF ave): {auc(linRa_rc[0], linRa_rc[1]):.3f}, {auc(linRa_rc[3], linRa_rc[4]):.3f}")
	
	plt.plot(rad_rc[0], rad_rc[1], label=f"radSVR: {auc(rad_rc[0], rad_rc[1]):.3f}, {auc(rad_rc[3], rad_rc[4]):.3f}")
	plt.plot(radR_rc[0], radR_rc[1], label=f"radSVR (RMSF): {auc(radR_rc[0], radR_rc[1]):.3f}, {auc(radR_rc[3], radR_rc[4]):.3f}")
	plt.plot(radRa_rc[0], radRa_rc[1], label=f"radSVR (RMSF ave): {auc(radRa_rc[0], radRa_rc[1]):.3f}, {auc(radRa_rc[3], radRa_rc[4]):.3f}")
	
	plt.plot(score_rc[0], score_rc[1], label=f"Score: {auc(score_rc[0], score_rc[1]):.3f}, {auc(score_rc[3], score_rc[4]):.3f}")
	plt.plot(scoreR_rc[0], scoreR_rc[1], label=f"Score (RMSF): {auc(scoreR_rc[0], scoreR_rc[1]):.3f}, {auc(scoreR_rc[3], scoreR_rc[4]):.3f}")
	plt.plot(scoreRa_rc[0], scoreRa_rc[1], label=f"Score (RMSF ave): {auc(scoreRa_rc[0], scoreRa_rc[1]):.3f}, {auc(scoreRa_rc[3], scoreRa_rc[4]):.3f}")
	
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.title('Training Set');
	plt.legend(loc='best')
	
	plt.subplot(1,2,2)
	plt.plot([0, 1], [0, 1], 'k--')
	
	plt.plot(old_rc[3], old_rc[4])
	
	plt.plot(lin_rc[3], lin_rc[4])
	plt.plot(linR_rc[3], linR_rc[4])
	plt.plot(linRa_rc[3], linRa_rc[4])
	
	plt.plot(rad_rc[3], rad_rc[4])
	plt.plot(radR_rc[3], radR_rc[4])
	plt.plot(radRa_rc[3], radRa_rc[4])
	
	plt.plot(score_rc[3], score_rc[4])
	plt.plot(scoreR_rc[3], scoreR_rc[4])
	plt.plot(scoreRa_rc[3], scoreRa_rc[4])
	
	
	plt.xlabel('FPR')
	plt.title('Test Set')
	return plt

# Old data - has a set training/test set.
old_train = pandas.read_csv('SBNN-feature_matrix_old.csv.gz') ; old_test = pandas.read_csv('SBNN-feature_matrix_old-TEST.csv.gz')
old_traindata = ScoreDataset(old_train.iloc[:,2:].values, old_train.iloc[:,1].values,oversample=args['os'], kfold=args['folds'], norm=True)
old_testdata = ScoreDataset(old_test.iloc[:,2:].values, old_test.iloc[:,1].values,oversample=False, kfold=args['folds'], norm=True)
# for the following datasets, the % of training/test set is specified by the -t/--test_size argument
# linSVR-selected data - average of top 3 decoys selected using this method
lin_data = pandas.read_csv('linSVR-avg.csv.gz');x_train, x_test, y_train, y_test = train_test_split(lin_data.iloc[:,2:].values, lin_data.iloc[:,1].values, test_size=args['test_set_size'], shuffle=True)
lin_traindata = ScoreDataset(x_train, y_train, oversample=args['os'], kfold=args['folds'], norm=True) ; lin_testdata = ScoreDataset(x_test, y_test, oversample=False, kfold=args['folds'], norm=True)
# same as previous, but with per-residue RMSF columns
linR_data = pandas.read_csv('linSVR-avg-RMSF.csv.gz');x_train, x_test, y_train, y_test = train_test_split(linR_data.iloc[:,2:].values, linR_data.iloc[:,1].values, test_size=args['test_set_size'], shuffle=True)
linR_traindata = ScoreDataset(x_train, y_train, oversample=args['os'], kfold=args['folds'], norm=True) ; linR_testdata = ScoreDataset(x_test, y_test, oversample=False, kfold=args['folds'], norm=True)
# same as previous, but with average peptide RMSF column
linRa_data = pandas.read_csv('linSVR-avg-RMSFave.csv.gz');x_train, x_test, y_train, y_test = train_test_split(linRa_data.iloc[:,2:].values, linRa_data.iloc[:,1].values, test_size=args['test_set_size'], shuffle=True)
linRa_traindata = ScoreDataset(x_train, y_train, oversample=args['os'], kfold=args['folds'], norm=True) ; linRa_testdata = ScoreDataset(x_test, y_test, oversample=False, kfold=args['folds'], norm=True)
# radSVR-selected data - average of top 3 decoys selected using this method
rad_data = pandas.read_csv('radSVR-avg.csv.gz');x_train, x_test, y_train, y_test = train_test_split(rad_data.iloc[:,2:].values, rad_data.iloc[:,1].values, test_size=args['test_set_size'], shuffle=True)
rad_traindata = ScoreDataset(x_train, y_train, oversample=args['os'], kfold=args['folds'], norm=True) ; rad_testdata = ScoreDataset(x_test, y_test, oversample=False, kfold=args['folds'], norm=True)
# same as previous, but with per-residue RMSF columns
radR_data = pandas.read_csv('radSVR-avg-RMSF.csv.gz');x_train, x_test, y_train, y_test = train_test_split(radR_data.iloc[:,2:].values, radR_data.iloc[:,1].values, test_size=args['test_set_size'], shuffle=True)
radR_traindata = ScoreDataset(x_train, y_train, oversample=args['os'], kfold=args['folds'], norm=True) ; radR_testdata = ScoreDataset(x_test, y_test, oversample=False, kfold=args['folds'], norm=True)
# same as previous, but with average peptide RMSF column
radRa_data = pandas.read_csv('radSVR-avg-RMSFave.csv.gz');x_train, x_test, y_train, y_test = train_test_split(radRa_data.iloc[:,2:].values, radRa_data.iloc[:,1].values, test_size=args['test_set_size'], shuffle=True)
radRa_traindata = ScoreDataset(x_train, y_train, oversample=args['os'], kfold=args['folds'], norm=True) ; radRa_testdata = ScoreDataset(x_test, y_test, oversample=False, kfold=args['folds'], norm=True)
# Rosetta score-selected data - average of top 3 decoys selected using this method
score_data = pandas.read_csv('total_score-avg.csv.gz');x_train, x_test, y_train, y_test = train_test_split(score_data.iloc[:,2:].values, score_data.iloc[:,1].values, test_size=args['test_set_size'], shuffle=True)
score_traindata = ScoreDataset(x_train, y_train, oversample=args['os'], kfold=args['folds'], norm=True) ; score_testdata = ScoreDataset(x_test, y_test, oversample=False, kfold=args['folds'], norm=True)
# same as previous, but with per-residue RMSF columns
scoreR_data = pandas.read_csv('total_score-avg-RMSF.csv.gz');x_train, x_test, y_train, y_test = train_test_split(scoreR_data.iloc[:,2:].values, scoreR_data.iloc[:,1].values, test_size=args['test_set_size'], shuffle=True)
scoreR_traindata = ScoreDataset(x_train, y_train, oversample=args['os'], kfold=args['folds'], norm=True) ; scoreR_testdata = ScoreDataset(x_test, y_test, oversample=False, kfold=args['folds'], norm=True)
# same as previous, but with average peptide RMSF column
scoreRa_data = pandas.read_csv('total_score-avg-RMSFave.csv.gz');x_train, x_test, y_train, y_test = train_test_split(scoreRa_data.iloc[:,2:].values, scoreRa_data.iloc[:,1].values, test_size=args['test_set_size'], shuffle=True)
scoreRa_traindata = ScoreDataset(x_train, y_train, oversample=args['os'], kfold=args['folds'], norm=True) ; scoreRa_testdata = ScoreDataset(x_test, y_test, oversample=False, kfold=args['folds'], norm=True)

# Train an ANN (same architecture, same training method) to each of these datasets.
old_rc = train_anal(old_traindata, old_testdata)
lin_rc = train_anal(lin_traindata, lin_testdata)
rad_rc = train_anal(rad_traindata, rad_testdata)
score_rc = train_anal(score_traindata, score_testdata)

linR_rc = train_anal(linR_traindata, linR_testdata)
radR_rc = train_anal(radR_traindata, radR_testdata)
scoreR_rc = train_anal(scoreR_traindata, scoreR_testdata)

linRa_rc = train_anal(linRa_traindata, linRa_testdata)
radRa_rc = train_anal(radRa_traindata, radRa_testdata)
scoreRa_rc = train_anal(scoreRa_traindata, scoreRa_testdata)

# Plot performance - in most frameworks this should? display the plot.
plt = plot(old_rc, lin_rc, rad_rc, score_rc, linR_rc, radR_rc, scoreR_rc, linRa_rc, radRa_rc, scoreRa_rc)

# optionally, save the plot as an image
if args['outfile']:
	plt.savefig(args['outfile'])
