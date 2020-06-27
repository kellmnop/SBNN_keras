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


#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 12} )
#sess = tf.Session(config=config)
#tf.keras.backend.set_session(sess)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6, restore_best_weights=True)

trdata = pandas.read_csv('SBNN-feature_matrix_old.csv')
train_X = data.iloc[:, 3:].values
train_Y = data.iloc[:, 2].values
train_data = ScoreDataset(train_X, train_Y, oversample=True)

tedata = pandas.read_csv('SBNN-feature_matrix_old-TEST.csv')
test_X = test_data.iloc[:, 3:].values
test_Y = test_data.iloc[:, 2].values
test_data = ScoreDataset(test_X, test_Y, oversample=False)

def buildmodel():
	model = tf.keras.models.Sequential([tf.keras.layers.Dense(5,input_shape=(81,),activation=tf.nn.relu,use_bias=True,kernel_initializer=tf.keras.initializers.glorot_normal),
	tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=True)])
	model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
	return(model)

estimator = KerasClassifier(build_fn=buildmodel, epochs=100, batch_size=64, verbose=1)
kfold = RepeatedKFold(n_splits=5, n_repeats=100)
results = cross_val_score(estimator, X, Y, cv=kfold, n_jobs=12)  # 2 cpus
results.mean()

history = model.fit(X, Y, epochs=100, batch_size=128, callbacks=[callback], shuffle=True)
