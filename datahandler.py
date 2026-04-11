"""
Backwards-compatibility shim. Canonical location: data/handler.py
"""
from data.handler import DataHandler, CompactArrayEncoder

if __name__ == "__main__":
    # Keep the original __main__ block for direct script usage
    dataset_path = r"C:\Users\knutj\OneDrive - Universitaet Bern\Datasets\FeasabilityStudy"
    data_handler = DataHandler("DataloaderGeneva", dataset_path, dataset_name='Geneva')
    data_handler.load_data(load_from_pkl=False, save_as_pkl=True, save_as_csv=True, reload_metadata=True)