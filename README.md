# ModelForge
Low-code framework that simplifies model setup and requirements with config files. Improves productivity and saves time for ML practitioners by allowing them to plug in parameters and utilize different models according to their needs.

## To run the project, execute the following command
```bash
 python main.py <filename>.yaml
```
  
## Project Structure:
**YAML files** 
 to know the parameters used, check the `examples/` directory 
* `roberta.yaml` - Roberta, pretrained model
* `pcnn.yaml` - Parallel CNN encoder parameters
* `rnn-params.yaml` - RNN encoder parameters
* `rnn.yaml` - RNN parameters used for text translation
* `transformer.yaml` - transformer model parameters
* `modelarch.yaml` - includes RNN encoder + combiner + RNNdecoder  model architecture
* `class-news.yaml` - parameters used for news classification


**preprocessed-data**: `preprocessed-data/` directory consists of preprocessed data of test, train and validation dataset files in `.hdf5` format
