# SBNN_keras
Python3 Tensorflow 2.0/Keras script for training artificial neural networks to predict broad immunogenicity of HLA-A*02:01-presented nonameric peptide epitopes from Rosetta score function features.
### Contains:
- training data from Riley et al. 2019 (SBNN-feature_matrix_old.csv)
- test/validation data from Riley et al. 2019 (SBNN-feature_matrix_old-TEST.csv)
- neural network training script intended to faithfully mirror the Matlab ANN from Riley et al. 2019.

usage: keras_model.py [-h] [-q] --train TRAIN_FILE --test TEST_FILE
                      [-n H_NODES] [-k FOLDS] [-o]

optional arguments:
  -h, --help            show this help message and exit
  -q, --quiet           Switch to disable tensorflow warning/log information.
  --train TRAIN_FILE    Training data in csv format with binary immunogenicity
                        in column index 1 and input features in columns 2:end
  --test TEST_FILE      Test data in csv format with binary immunogenicity in
                        column index 1 and input features in columns 2:end
  -n H_NODES, --hidden_size H_NODES
                        Number(s) of nodes in the hidden layer to train with.
                        If more than one, argument should be a comma-separated
                        list with no whitespace.
  -k FOLDS, --kfold FOLDS
                        Number of folds to use for kfold cross validation
                        (default 5).
  -o, --output          Switch to save training/test performance metrics to
                        files. Type is vanilla .csv, path is current
                        directory, default filename is same as input files
                        with ".perf" extension.
