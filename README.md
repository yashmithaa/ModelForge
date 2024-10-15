```
  __  __           _      _ _____                    
 |  \/  | ___   __| | ___| |  ___|__  _ __ __ _  ___ 
 | |\/| |/ _ \ / _` |/ _ \ | |_ / _ \| '__/ _` |/ _ \
 | |  | | (_) | (_| |  __/ |  _| (_) | | | (_| |  __/
 |_|  |_|\___/ \__,_|\___|_|_|  \___/|_|  \__, |\___|
                                          |___/      
A low code Machine Learning Platform.
```
# ModelForge
ModelForge is a powerful and user-friendly platform that simplifies the machine learning process, allowing you to build and deploy models without extensive coding. It simplifies model setup and requirements with config files. Improves productivity and saves time for ML practitioners by allowing them to plug in parameters and utilize different models according to their needs.

## How it Works
<img src="https://raw.githubusercontent.com/yashmithaa/ModelForge/refs/heads/main/assets/flowchart.png">

- **Load Your Dataset:** Simply provide the path to your dataset CSV file and the corresponding YAML configuration file.

- **Specify Model Requirements:** In the YAML file, define the desired model architecture, input and output features, preprocessing steps, and evaluation metrics.

- **Let ModelForge Do the Rest:** ModelForge will automatically handle the following tasks:

    - **Data Cleaning and Preprocessing:** It will clean and preprocess your data, addressing issues such as missing values, duplicates, and outliers.
    - **Dataset Splitting:** ModelForge will split your data into training, validation, and test sets for effective model training and evaluation.
    - **Model Selection and Training:** It will select the most suitable model architecture based on your specified requirements and train the model on the training dataset.
    - **Model Evaluation:** ModelForge will evaluate the trained model's performance using the specified evaluation metrics on the validation set.
    
## Key Features:

- **Low-Code Approach:** No extensive coding is required. Simply provide the necessary information in the YAML file.
- **Flexibility:** Customize your models by specifying different architectures, features, and preprocessing steps.
- **Efficiency:** ModelForge automates many time-consuming tasks, saving you valuable time.
## Getting Started

Create a virtual environment, run the following command after cloning the repo
```bash
python -m venv venv
```

Activate the virtual environment
  - Windows:
    ```bash
    venv\Scripts\activate
    ```
  - MacOS:
    ```bash
    source venv/bin/activate
    ```
Install required python packages [Make sure virtual environment is running]
  ```bash
  pip install -r requirements.txt
  ```

To run the file
```bash
 python main.py <filename>.yaml
```
  
## Project Structure:

```
ModelForge/
├── datasets/        
│   └── ...         #sample datasets we have worked on
|
├── examples/       
│   ├── class-news.yaml  # Example configuration file for news classificantion
│   ├── modelarch.yaml   # model architecture
│   └── transformer.yaml  # transformer model
|
├── preprocessed_data/  # stores preprocessed data
│   ├── dataset.test.hdf5      
│   ├── dataset.training.hdf5     
│   └── dataset.validation.hdf5
| 
├── samples/          # ML problems modelforge worked on
│   └── ...     
│    
├── readme.md        
├── main.py            # Main script for running the project
└── requirements.txt   
```

## Yaml File
```yaml
dataset:
  path: "datasets/dataset.csv"
  format: csv
  delimiter: ","

input_features:
  - name: text
    type: text
    encoder: roberta

output_features:
  - name: sentiment
    type: category
    num_classes: 3

training:
  epochs: 5
  batch_size: 32


preprocessing:
  split:
    train: 0.7
    test: 0.2
    validation: 0.1
    random_seed: 42
  text:
    lower_case: true
    remove_punctuation: true
    remove_stopwords: true
    stemming: false
    tokenization:
      method: word
```
