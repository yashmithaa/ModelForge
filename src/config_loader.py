import yaml
import logging

class ConfigLoader:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path):
        logging.info(f"Loading config from {config_path}")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info("Config loaded successfully")
        return config