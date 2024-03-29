import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.metrics import Accuracy

import os
import pickle

from Algo import BaseFileInfo
from Algo import BaseModel

def ratings_float2int(float_ratings, float_ratingMax = 2, float_ratingMin = 0, int_ratingMax = 5, int_ratingMin = 1):
    int_ratings = []
    for prediction in float_ratings:
        int_ratings.append(((prediction - float_ratingMax) / (float_ratingMin - float_ratingMax)) * (int_ratingMax - int_ratingMin) + int_ratingMin)
    return int_ratings

class NN_FileInfo(BaseFileInfo):
    def __init__(self, model_file, model_state_file, wide_preprocessor_file, tab_preprocessor_file, dataset_path, user_dataset_path, item_dataset_path):
        super().__init__(model_file)
        self.model_state_file = model_state_file
        self.wide_preprocessor_file = wide_preprocessor_file
        self.tab_preprocessor_file = tab_preprocessor_file
        self.dataset_path = dataset_path
        self.user_dataset_path = user_dataset_path
        self.item_dataset_path = item_dataset_path

    def load_data(self):
        data = pd.read_csv(self.dataset_path)
        return pd.DataFrame(data)

class Model:
    def __init__(self, file_info, epochs_n = 10, batch_size = 64):
        self.file_info = file_info
        self.epochs_n = epochs_n
        self.batch_size = batch_size
        self.model = None
        self.wide_preprocessor = None
        self.tab_preprocessor = None
        self.wide_component = None
        self.deep_component = None
        self.target = None
        self.state_dict_model = None
        self.trainer = None

    def save_preprocessors(self):
        with open(self.file_info.wide_preprocessor_file, "wb") as f:
            pickle.dump(self.wide_preprocessor, f)
        with open(self.file_info.tab_preprocessor_file, "wb") as f:
            pickle.dump(self.tab_preprocessor, f)

    def define_model(self):
        wide_cols = ["cod_art", "cod_linea_comm", "cod_sett_comm"]
        crossed_cols = [("cod_art", "cod_linea_comm"), ("cod_linea_comm", "cod_sett_comm")]

        cat_embed_cols = ["cod_cli", "cod_linea_comm", "cod_sett_comm", "cod_fam_comm"]
        continuous_cols = None
        self.target = "rating"

        df_train = self.file_info.load_data()
        self.target = df_train[self.target].values

        self.wide_preprocessor = WidePreprocessor(wide_cols = wide_cols, crossed_cols = crossed_cols)
        self.tab_preprocessor = TabPreprocessor(cat_embed_cols = cat_embed_cols, continuous_cols = continuous_cols)

        self.wide_component = self.wide_preprocessor.fit_transform(df_train)
        self.deep_component = self.tab_preprocessor.fit_transform(df_train)

        self.save_preprocessors()

        wide = Wide(input_dim = np.unique(self.wide_component).shape[0], pred_dim = 1)
        tab_mlp = TabMlp(column_idx = self.tab_preprocessor.column_idx, cat_embed_input = self.tab_preprocessor.cat_embed_input)
        
        self.model = WideDeep(wide = wide, deeptabular = tab_mlp)

    def load_model(self):
        if os.path.exists(self.file_info.model_file):
            self.model = torch.load(self.file_info.model_file)
            self.model.load_state_dict(torch.load(self.file_info.model_state_file))
            self.state_dict_model = self.model.state_dict()
            with open(self.file_info.wide_preprocessor_file, "rb") as file:
                self.wide_preprocessor = pickle.load(file)
            with open(self.file_info.tab_preprocessor_file, "rb") as file:
                self.tab_preprocessor = pickle.load(file)
        else: 
            self.define_model()
              
    def save_model(self):
        if self.model is not None:
            torch.save(self.model.state_dict(), self.file_info.model_state_file)
            torch.save(self.model, self.file_info.model_file)

    def train_model(self):
        self.load_model()
        
        if not os.path.exists(self.file_info.model_file):
            self.trainer = Trainer(self.model, objective = "regression", metrics = [Accuracy])
            self.trainer.fit(X_wide = self.wide_component, X_tab = self.deep_component, target = self.target, n_epochs = self.epochs_n, batch_size = self.batch_size, val_split = 0.3)
        else:
            self.trainer = Trainer(self.model, objective = "regression", metrics = [Accuracy])

        self.save_model()
    
    def topN_1ItemNUser(self, item_infos, n = 5):
        users_df = pd.DataFrame(pd.read_csv(self.file_info.user_dataset_path))
        
        users_df['cod_art'] = item_infos[0]
        users_df['cod_linea_comm'] = item_infos[1]
        users_df['cod_sett_comm'] = item_infos[2]
        users_df['cod_fam_comm'] = item_infos[3]
        
        X_user_wide = self.wide_preprocessor.transform(users_df)
        X_user_tab = self.tab_preprocessor.transform(users_df)
        
        user_rating_predictions = self.trainer.predict(X_wide = X_user_wide, X_tab = X_user_tab, batch_size = self.batch_size)

        user_rating_predictions = abs(user_rating_predictions)
        
        converted_ratings = ratings_float2int(user_rating_predictions, float_ratingMax = max(user_rating_predictions))
        
        user_ratings = list(zip(users_df['cod_art'], converted_ratings))

        top_n_users = sorted(user_ratings, key=lambda x: x[1], reverse = True)[:n]
        
        return top_n_users
    
    def topN_1UserNItem(self, user_id, n = 5):
        products_df = pd.read_csv(self.file_info.item_dataset_path)

        products_df['cod_cli'] = user_id
        
        X_product_wide = self.wide_preprocessor.transform(products_df)
        X_product_tab = self.tab_preprocessor.transform(products_df)         
        
        product_rating_predictions = self.trainer.predict(X_wide = X_product_wide, X_tab = X_product_tab, batch_size = self.batch_size)

        product_rating_predictions = abs(product_rating_predictions)
        
        converted_ratings = ratings_float2int(product_rating_predictions, float_ratingMax = max(product_rating_predictions))
        
        product_ratings = list(zip(products_df['cod_art'], converted_ratings))
        
        top_n_products = sorted(product_ratings, key=lambda x: x[1], reverse = True)[:n] 
        
        return top_n_products

