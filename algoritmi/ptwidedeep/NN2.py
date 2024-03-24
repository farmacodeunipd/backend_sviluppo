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
    def __init__(self, model_file, model_state_file, wide_preprocessor_file, tab_preprocessor_file, dataset_path, user_dataset_path, item_dataset_path):
        self.model_file = model_file
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
        self.trainer = None

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
        self.target = "rating"

        df_train = self.file_info.load_data()
        self.target = df_train[self.target].values

        self.wide_preprocessor = WidePreprocessor(wide_cols = wide_cols, crossed_cols = crossed_cols)
        self.tab_preprocessor = TabPreprocessor(cat_embed_cols = cat_embed_cols, continuous_cols = continuous_cols)
        self.save_preprocessors()

        self.wide_component = self.wide_preprocessor.fit_transform(df_train)
        self.deep_component = self.tab_preprocessor.fit_transform(df_train)

        wide = Wide(input_dim = np.unique(self.wide_component).shape[0], pred_dim = 1)
        tab_mlp = TabMlp(column_idx = self.tab_preprocessor.column_idx, cat_embed_input = self.tab_preprocessor.cat_embed_input)
        
        self.model = WideDeep(wide = wide, deeptabular = tab_mlp)

    def load_model(self):
        if os.path.exists(self.file_info.model_file):
            self.model = torch.load(self.file_info.model_file)
            self.model.load_state_dict(torch.load(self.file_info.model_state_file))
            self.state_dict_model = self.model.state_dict()
            with open(self.file_info.wide_preprocessor_file, 'rb') as file:
                self.wide_preprocessor = pickle.load(file)
            with open(self.file_info.tab_preprocessor_file, 'rb') as file:
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
            self.trainer.fit(X_wide = self.wide_component, X_tab = self.deep_component, target = self.target, n_epochs = self.epochs_n, batch_size = self.batch_size)
        else:
            self.trainer = Trainer(self.model, objective = "regression", metrics = [Accuracy])

        self.save_model()

    def TopN_1UserNItem(self, user_id, n=5):
        # Read the CSV file containing product IDs
        products_df = pd.read_csv(self.file_info.item_dataset_path)
        
        # Remove the 'rating' column if it exists
        if 'rating' in products_df.columns:
            products_df.drop(columns=['rating'], inplace=True)
        
        # Add the user ID column to the DataFrame
        products_df['cod_cli'] = user_id
        products_df['cod_linea_comm'] = 'NULL'
        products_df['cod_sett_comm'] = 'NULL'
        products_df['cod_fam_comm'] = 'NULL'
        
        # Fit and transform the WidePreprocessor
        X_product_wide = self.wide_preprocessor.fit_transform(products_df)
        
        # Fit and transform the TabPreprocessor
        X_product_tab = self.tab_preprocessor.fit_transform(products_df)
        
        # Make predictions for the products and user
        product_rating_predictions = self.trainer.predict(X_wide=X_product_wide, X_tab=X_product_tab, batch_size=64)
        
        # Combine product IDs with their predicted ratings
        product_ratings = list(zip(products_df['cod_art'], product_rating_predictions))
        
        # Sort the products by predicted ratings in descending order
        top_n_products = sorted(product_ratings, key=lambda x: x[1], reverse=True)[:n]
        
        # Return the top-N products
        return top_n_products
    
    def TopN_1ItemNUser(self, item_id, n = 5):
        # Read the CSV file containing user IDs
        users_df = pd.read_csv(self.file_info.user_dataset_path)
        
        # Add the product ID column to the DataFrame
        users_df['cod_art'] = item_id
        users_df['cod_linea_comm'] = '11'   #TODO queries
        users_df['cod_sett_comm'] = '2V'
        users_df['cod_fam_comm'] = 'G2'
        
        # Preprocess the user information
        X_user_wide = self.wide_preprocessor.transform(users_df)
        X_user_tab = self.tab_preprocessor.transform(users_df)
        
        # Make predictions for the users and product
        user_rating_predictions = self.trainer.predict(X_wide=X_user_wide, X_tab=X_user_tab, batch_size=64)
        
        # Combine user IDs with their predicted ratings
        user_ratings = list(zip(users_df['cod_cli'], user_rating_predictions))
        
        # Sort the users by predicted ratings in descending order
        top_n_users = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:n]
        
        # Return the top-N users
        return top_n_users
    
if __name__ == '__main__':
    multiprocessing.freeze_support()

    file_infos = FileInfo("./algoritmi/ptwidedeep/model.pt", "./algoritmi/ptwidedeep/wd_model.pt", "./algoritmi/ptwidedeep/WidePreprocessor.pkl", "./algoritmi/ptwidedeep/TabPreprocessor.pkl", "./algoritmi/ptwidedeep/data_preprocessed_NN.csv", "./db/dataset/anacli.csv", "./db/dataset/anaart.csv")
    neural_network = Model(file_infos)
    neural_network.train_model()

    top_items = neural_network.TopN_1UserNItem(20, 20)
    print(f"Top {len(top_items)} possible items for user {20}:")
    for rank, (item_id, rating) in enumerate(top_items, start=1):
        print(f"Rank {rank}: User ID: {item_id}, Predicted Rating: {rating}")

    top_users = neural_network.TopN_1ItemNUser(20, 20)
    print(f"Top {len(top_users)} possible users for product {20}:")
    for rank, (user_id, rating) in enumerate(top_users, start=1):
        print(f"Rank {rank}: User ID: {user_id}, Predicted Rating: {rating}")

