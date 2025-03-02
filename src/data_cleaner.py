import logging

class DataCleaner:
    def __init__(self, config):
        self.config = config

    def clean_data(self, data):
        logging.info("Cleaning data")
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        logging.info("Data cleaned successfully")
        return data
