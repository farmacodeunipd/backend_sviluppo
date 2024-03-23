import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import WidePreprocessor, TabPreprocessor
from pytorch_widedeep.models import Wide, TabMlp, WideDeep
from pytorch_widedeep.metrics import Accuracy

import multiprocessing
import os
import pickle

class FileInfo:
    def __init__(self, model_file, wide_preprocessor_file, tab_preprocessor_file, file_path):
        self.model_file = model_file
        self.wide_preprocessor_file = wide_preprocessor_file
        self.tab_preprocessor_file = tab_preprocessor_file
        self.file_path = file_path

    def load_data(self):
        data = pd.read_csv(self.file_path)
        return pd.DataFrame(data)

class Model:
    def __init__(self, file_info, epochs_n = 5, batch_size = 64):
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

    def save_preprocessors(self):
        with open(self.file_info.wide_preprocessor_file, "wb") as f:
            pickle.dump(self.wide_preprocessor, f)
        with open(self.file_info.tab_preprocessor_file, "wb") as f:
            pickle.dump(self.tab_preprocessor, f)

    def define_model(self):
        wide_cols = ["cod_cli", "cod_art", "cod_linea_comm", "cod_sett_comm", "cod_fam_comm"]
        crossed_cols = [("cod_linea_comm", "cod_sett_comm")]

        cat_embed_cols = ["cod_cli", "cod_art", "cod_linea_comm", "cod_fam_comm"]
        continuous_cols = ["rating"]
        target = "rating"

        df_train = self.file_info.load_data()
        target = df_train[target].values

        self.wide_preprocessor = WidePreprocessor(wide_cols = wide_cols, crossed_cols = crossed_cols)
        self.tab_preprocessor = TabPreprocessor(cat_embed_cols = cat_embed_cols, continuous_cols = continuous_cols)
        self.save_preprocessors()

        self.wide_component = self.wide_preprocessor.fit_transform(df_train)
        self.deep_component = self.tab_preprocessor.fit_transform(df_train)

        wide = Wide(input_dim = np.unique(self.wide_component).shape[0], pred_dim = 1)
        tab_mlp = TabMlp(column_idx = self.tab_preprocessor.column_idx, cat_embed_input = self.tab_preprocessor.cat_embed_input)
        
        self.model = WideDeep(wide = wide, deeptabular = tab_mlp)

    def load_model(self):
        #todo
        return 1
              
    def save_model(self):
        if self.model is not None:
            torch.save(self.model.state_dict(), "./algoritmi/ptwidedeep/wd_model.pt")
            #salvare anche definizione di modello (self.model) oltre che allo stato (self.model.state_dict())?

    def train_model(self):
        trainer = Trainer(self.model, objective = "regression", metrics = [Accuracy])
        trainer.fit(X_wide = self.wide_component, X_tab = self.deep_component, target = self.target, n_epochs = self.epochs_n, batch_size = self.batch_size)

        self.save_model()

    def TopN_1UserNItem(self, user_id, n = 5):
        # Read the CSV file containing product IDs
        products_df = pd.read_csv(product_ids_csv)
        
        # Add the user ID column to the DataFrame
        products_df['cod_cli'] = user_id
        products_df['cod_linea_comm'] = 'NULL'  # Default value, modify as needed
        products_df['cod_sett_comm'] = 'NULL'   # Default value, modify as needed
        products_df['cod_fam_comm'] = 'NULL'    # Default value, modify as needed
        
        # Preprocess the product information
        X_product_wide = wide_preprocessor.transform(products_df)
        X_product_tab = tab_preprocessor.transform(products_df)
        
        # Make predictions for the products and user
        product_rating_predictions = trained_model.predict(X_wide=X_product_wide, X_tab=X_product_tab, batch_size=64)
        
        # Combine product IDs with their predicted ratings
        product_ratings = list(zip(products_df['cod_art'], product_rating_predictions))
        
        # Sort the products by predicted ratings in descending order
        top_n_products = sorted(product_ratings, key=lambda x: x[1], reverse=True)[:N]
        
        # Return the top-N products
        return top_n_products
    
    def TopN_1ItemNUser(self, item_id, n = 5):
        # Read the CSV file containing user IDs
        users_df = pd.read_csv(user_ids_csv)
        
        # Add the product ID column to the DataFrame
        users_df['cod_art'] = product_id
        users_df['cod_linea_comm'] = '11'  # Default value, modify as needed
        users_df['cod_sett_comm'] = '2V'   # Default value, modify as needed
        users_df['cod_fam_comm'] = 'G2'    # Default value, modify as needed
        
        # Preprocess the user information
        X_user_wide = wide_preprocessor.transform(users_df)
        X_user_tab = tab_preprocessor.transform(users_df)
        
        # Make predictions for the users and product
        user_rating_predictions = trained_model.predict(X_wide=X_user_wide, X_tab=X_user_tab, batch_size=64)
        
        # Combine user IDs with their predicted ratings
        user_ratings = list(zip(users_df['cod_cli'], user_rating_predictions))
        
        # Sort the users by predicted ratings in descending order
        top_n_users = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:N]
        
        # Return the top-N users
        return top_n_users