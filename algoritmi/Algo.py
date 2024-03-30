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
    
class ModelContext:
    def __init__(self, model_info):
        self.model_info = model_info

    def set_model_info(self, model_info):
        self.model_info = model_info

    def process_data(self, data):
        self.model_info.load_data(data)

    def load_model(self):
        return self.model_info.load_model()

    def save_model(self):
        return self.model_info.save_model()

    def train_model(self):
        return self.model_info.train_model()

    def topN_1UserNItem(self, user_id, n=5):
        return self.model_info.topN_1UserNItem(user_id, n)

    def topN_1ItemNUser(self, item_info, n=5):
        return self.model_info.topN_1ItemNUser(item_info, n)