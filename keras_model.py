import pandas
import numpy as np
import tensorflow as tf
from sklearn.model_selection import RepeatedKFold, cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from immunomodeling_dataset import ScoreDataset

'''
Implement oversampling (of training set not val) prior to training.
'''
hidden_size = 5

#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 12} )
#sess = tf.Session(config=config)
#tf.keras.backend.set_session(sess)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

trdata = pandas.read_csv('SBNN-feature_matrix_old.csv')
tr_X = trdata.iloc[:, 3:].values
tr_Y = trdata.iloc[:, 2].values

tedata = pandas.read_csv('SBNN-feature_matrix_old-TEST.csv')
te_X = tedata.iloc[:, 3:].values
te_Y = tedata.iloc[:, 2].values

assert tr_X.shape[1] == te_X.shape[1], "Training and test data do not have the same number of features!"

train_data = ScoreDataset(tr_X, tr_Y, oversample=True)
test_data = ScoreDataset(te_X, te_Y, oversample=False)

def buildmodel(n_hidden):
	model = tf.keras.models.Sequential([tf.keras.layers.Dense(n_hidden, input_shape=(81,),activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.keras.initializers.glorot_normal),
	tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True)])
	model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
	return(model)

cvscores = []
i = 0
for train_X, train_Y, val_X, val_Y in train_data.get_training_batch():
	model = buildmodel(hidden_size)
	model.fit(train_X, train_Y, epochs=100, batch_size=128, callbacks=[callback], shuffle=True, verbose=0)
	scores = model.evaluate(val_X, val_Y, verbose=0)
	print(f'Leaf {i+1} '+', '.join([f'{model.metrics_names[i]}:{scores[i]:2f}' for i in range(len(scores))]))
	cvscores.append(scores)
	i += 1
performance = np.mean(cvscores, axis=0)
error = np.std(cvscores, axis=0)
print(', '.join([f'{model.metrics_names[i]}: {performance[i]:2f} +/- {error[i]:2f}' for i in range(len(performance))]))
