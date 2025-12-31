import logging

def setup_logger():
    logging.basicConfig(filename="logs/pipeline.log", filemode='w', 
                        level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')