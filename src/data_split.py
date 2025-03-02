import logging
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, config):
        self.config = config['preprocessing']['split']
        self.train_data = None
        self.test_data = None
        self.validation_data = None

    def split_data(self, data):
        train_percent = self.config['train']
        test_percent = self.config['test']
        validation_percent = self.config['validation']
        random_seed = self.config.get('random_seed', None)

        # Calculate the sizes for each split
        test_size = test_percent / (test_percent + validation_percent)
        validation_size = validation_percent / (test_percent + validation_percent)

        
        data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # First split: into training and remaining data (test + validation)
        self.train_data, remaining_data = train_test_split(data, test_size=(test_percent + validation_percent), random_state=random_seed)

        # Second split: remaining data into test and validation sets
        self.test_data, self.validation_data = train_test_split(remaining_data, test_size=test_size, random_state=random_seed)

        self.save_hdf5()

        logging.info("Data split successfully")
        return self.train_data, self.test_data, self.validation_data
    
    def save_hdf5(self):
        logging.info("Saving datasets to HDF5 files")
        
        self.train_data.to_hdf(f'preprocessed-data/dataset.training.hdf5', key='train', mode='w')
        print("\nWriting preprocessed training set to preprocessed-data/dataset.training.hdf5")

        self.test_data.to_hdf(f'preprocessed-data/dataset.test.hdf5', key='test', mode='w')
        print("Writing preprocessed test set to preprocessed-data/dataset.test.hdf5")

        self.validation_data.to_hdf(f'preprocessed-data/dataset.validation.hdf5', key='validation', mode='w')
        print("Writing preprocessed validation set to preprocessed-data/dataset.validation.hdf5\n")
