# ModelForge
Low-code framework that simplifies model setup and requirements with config files. Improves productivity and saves time for ML practitioners by allowing them to plug in parameters and utilize different models according to their needs.

## Project Structure:
-**preprocessor.py**: contains all preprocessing steps and definitions of classes for model architectures

-parallelCNN, StackedCNN, custom implementation of Transformer model, RNN, categorical & numerical encoder-decoder architectures

-**.yaml files**: config.yaml, pcnn.yaml, rnn.yaml, transformer.yaml, modelarch.yaml

-config files with specified parameters for each particular model

-**torch-v**: pytorch implementation 

-**preprocessed-data file**: consists of test,training and validation dataset files in hdf5 format
