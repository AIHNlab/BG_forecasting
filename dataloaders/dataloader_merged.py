import os
import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as etree
import joblib
from dataloaders.dataloader import Dataloader

class DataloaderMerged(Dataloader):
    def __init__(self, directory_path):
        super().__init__(directory_path)

    def load_data(self):
        pass
  
    def get_metadata(self):
        pass

    def _get_dataframe(self, file):
        pass
    
    def _get_dataset_specific_metadata(self):
        pass
