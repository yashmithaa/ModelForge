import sys
import logging

from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint
from rich.table import Table


from src.config_loader import ConfigLoader
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.preprocess_text import TextPreprocessor
from src.data_split import DataSplitter
from src.roberta import Model
from src.transformer import TransformerModel
from src.modelarch import ModelArch


from utils.logger import setup_logger

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    console = Console()

    # Setup logging
    setup_logger()
    logging.info("Starting main function")

    # Load config
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config(config_path)
    console.print("\nUser specified config file\n", style="bold")
    pprint(config)

    # Load data
    loader = DataLoader(config)
    data = loader.load_dataset()

    # Clean data
    cleaner = DataCleaner(config)
    data = cleaner.clean_data(data)

    # Preprocess data
    console.print(Markdown("# Preprocessing"))
    preprocessor = TextPreprocessor(config)
    data = preprocessor.preprocess_dataset(data)

    # Split data
    splitter = DataSplitter(config)
    train_set, validation_set, test_set = splitter.split_data(data)

    table = Table(title=f"Dataset statistics\nTotal datset: {len(train_set)+len(validation_set)+len(test_set)}")
    table.add_column("Dataset", style = "Cyan")
    table.add_column("Size (in Rows)")
    table.add_column("Size (in memeory)")
    table.add_row("Train set", str(len(train_set)), f"{(sys.getsizeof(train_set) / (1024 * 1024)):.2f} Mb")
    table.add_row("Validation set", str(len(validation_set)), f"{(sys.getsizeof(validation_set) / (1024 * 1024)):.2f} Mb")
    table.add_row("Test set", str(len(test_set)), f"{(sys.getsizeof(test_set) / (1024 * 1024)):.2f} Mb")

    
    console.print(table)

    for feature in config['input_features']:
        if feature['encoder'] == 'roberta':
            model = Model(config)
            results = model.roberta(test_set,num_samples=5)
            model.print_results(results)
        elif feature['encoder'] == 'transformer':
            model = TransformerModel(vocab_size=config['model']['vocab_size'],
                    d_model=config['model']['d_model'],
                    num_heads=config['model']['num_heads'],
                    num_layers=config['model']['num_layers'],
                    dim_feedforward=config['model']['dim_feedforward'],
                    max_seq_len=config['model']['max_seq_len'],
                    num_classes=config['model']['num_classes'],
                    dropout=config['model']['dropout'])
            print(model)
        else:
            model = ModelArch(config)
            print(model)
    

if __name__ == "__main__":
    main()
