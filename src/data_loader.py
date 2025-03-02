import pandas as pd
import logging

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_dataset(self):
        dataset_config = self.config['dataset']
        logging.info(f"Loading dataset from {dataset_config['path']}")
        data = pd.read_csv(dataset_config['path'], delimiter=dataset_config['delimiter'], encoding='utf-8', encoding_errors='ignore')
        logging.info("Dataset loaded successfully")
        return data
