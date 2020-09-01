# SBNN_keras
Python3 Tensorflow 2.0/Keras script for training artificial neural networks to predict broad immunogenicity of HLA-A*02:01-presented nonameric peptide epitopes from Rosetta score function features.
### Contains:
- training data from Riley et al. 2019 (SBNN-feature_matrix_old.csv)
- test/validation data from Riley et al. 2019 (SBNN-feature_matrix_old-TEST.csv)
- neural network training script intended to faithfully mirror the Matlab ANN from Riley et al. 2019.

```
usage: keras_model.py [-h] [-q] [-t TEST_SET_SIZE] [-n H_NODES] [-k FOLDS]
                      [-s OS] [-o OUTFILE]

optional arguments:
  -h, --help            show this help message and exit
  -q, --quiet           Switch to disable tensorflow warning/log information.
  -t TEST_SET_SIZE, --test_size TEST_SET_SIZE
                        Proportion of total data to use in the test set
                        (default 0.10).
  -n H_NODES, --hidden_size H_NODES
                        Number(s) of nodes in the hidden layer to train with.
  -k FOLDS, --kfold FOLDS
                        Number of folds to use for kfold cross validation
                        (default 5).
  -s OS, --oversample OS
                        Desired ratio of minority:majority class. If
                        specified, randomly oversamples training set only to
                        this ratio.
  -o OUTFILE, --outfile OUTFILE
                        Save ROC AUC plots as an image file e.g. asdf.png
```
