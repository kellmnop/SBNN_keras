import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class ScoreDataset:
	'''
	Basic structure implementing data access and behavior useful for training ANN in keras/tensorflow.
	'''
	def __init__(self, features, labels, oversample=None, kfold=5, norm=False):
		assert len(features) == len(labels), "Labels (Y) and features (X) do not have the same number of observations!"
		self.X = np.array(features).astype('float64')
		if norm:
			scaler = MinMaxScaler(feature_range=(-1, 1), copy=False).fit(self.X)
			scaler.transform(self.X)
		self.Y = np.array(labels).astype('int32').reshape(len(labels),-1)
		self.leaves = []
		self.num_leaves = 1
		self.n_batches = 1
		self.oversample = oversample
		self.kfold = kfold
		self.enc = OneHotEncoder() ; self.enc.fit(self.Y)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, i):
		return [self.X[i], self.Y[i]]

	def get_x_dims(self):
		return self.X.shape[1]

	def get_y_dims(self):
		return self.Y.shape[1]

	def _oversample(self, idx, ratio=0.5):
		'''
		Random resampling of immunogenic peptides to equal ratio
		of non-immunogenic (training set only).
		'''
		# Might be better ways to do this than just random oversampling -- e.g. SMOTE?
		idx1 = [i for i in range(len(idx)) if self.Y[i] == 1]
		idx0 = [i for i in range(len(idx)) if self.Y[i] == 0]
		if len(idx1)/len(idx) < ratio:
			#print('Oversampling class 1.')
			# number of class 1 to resample
			target = int(round(ratio*len(idx0)/(1.0-ratio)))
			resample_idx = np.random.choice(idx1, size=target, replace=True)
		elif len(idx0)/len(idx) < ratio:
			#print('Oversampling class 0.')
			# number of class 0 to resample
			target = int(round(ratio*len(idx1)/(1.0-ratio)))
			resample_idx = np.random.choice(idx0, size=target, replace=True)
		else:
			#print('Class sizes already equal (no oversampling).')
			return
		return resample_idx

	def get_splits(self, folds):
		'''
		Defines k training/validation leaves.
		'''
		skf = StratifiedKFold(n_splits=folds, shuffle=True)
		assert type(folds) == int, "Folds must be type int."
		assert folds>1, "Number of folds must be greater than 1."
		for train_idx, val_idx in skf.split(self.X, self.Y):
			if self.oversample:
				train_idx = np.append(train_idx, self._oversample(train_idx, self.oversample), axis=0)
			self.leaves.append((train_idx, val_idx))
		self.num_leaves = folds

	def get_training_batch(self):
		'''
		Method generating data for training and validation leaves.
		'''
		self.get_splits(self.kfold)
		for leaf in self.leaves:
			yield self.X[leaf[0]], self.enc.transform(self.Y[leaf[0]]).toarray(), self.X[leaf[1]], self.enc.transform(self.Y[leaf[1]]).toarray()

