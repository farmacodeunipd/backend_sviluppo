from abc import ABC, abstractmethod
import pandas as pd

class BaseFileInfo(ABC):
    def __init__(self, model_file):
        self.model_file = model_file

    @abstractmethod
    def load_data(self):
        pass
    
class BaseModel(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def topN_1UserNItem(self, user_id, n=5):
        pass

    @abstractmethod
    def topN_1ItemNUser(self, item_info, n=5):
        pass