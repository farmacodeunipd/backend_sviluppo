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

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # Loading the dataset
    if not os.path.exists('./algoritmi/ptwidedeep/data_preprocessed_NN.csv'):
        print("Preprocessed Data are not ready or do not exist, please wait or start preprocessing the data.")
    else:
        data = pd.read_csv('./algoritmi/ptwidedeep/data_preprocessed_NN.csv')
        df = pd.DataFrame(data)
        print("Preprocessed Data have been loaded successfully.")

    # Split the data into train and test sets
    df_train, df_test = train_test_split(df, test_size=0.2)
    # df_test = df chiedi a matteo "perchè?"

    # Define the column setup
    wide_cols = [
        "cod_cli",
        "cod_art",
        "cod_linea_comm",
        "cod_sett_comm",
        "cod_fam_comm",
    ]
    crossed_cols = [("cod_linea_comm", "cod_sett_comm")]

    cat_embed_cols = [
        "cod_cli",
        "cod_art",
        "cod_linea_comm",
        "cod_fam_comm",
    ]
    continuous_cols = ["rating"]
    target = "rating"
    target = df_train[target].values

    # Prepare the data
    wide_preprocessor = WidePreprocessor(wide_cols=wide_cols)
    X_wide = wide_preprocessor.fit_transform(df_train)

    tab_preprocessor = TabPreprocessor(cat_embed_cols=cat_embed_cols)
    X_tab = tab_preprocessor.fit_transform(df_train)

    # Save the preprocessors
    with open("./algoritmi/ptwidedeep/WidePreprocessor.pkl", "wb") as f:
        pickle.dump(wide_preprocessor, f)
    with open("./algoritmi/ptwidedeep/TabPreprocessor.pkl", "wb") as f:
        pickle.dump(tab_preprocessor, f)

    # Build the model
    wide = Wide(input_dim=np.unique(X_wide).shape[0], pred_dim=1)
    tab_mlp = TabMlp(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
    )
    model = WideDeep(wide=wide, deeptabular=tab_mlp)

    # Train and validate
    trainer = Trainer(model, objective="regression", metrics=[Accuracy])
    trainer.fit(
        X_wide=X_wide,
        X_tab=X_tab,
        target=target,
        n_epochs=1,
        batch_size=64,
    )

    # Predict on test
    X_wide_te = wide_preprocessor.transform(df_test)
    X_tab_te = tab_preprocessor.transform(df_test)
    preds = trainer.predict(X_wide=X_wide_te, X_tab=X_tab_te)

    print(preds)

    # Save
    torch.save(model.state_dict(), "./algoritmi/ptwidedeep/wd_model.pt")

    # 1. Build the model
    model_new = WideDeep(wide=wide, deeptabular=tab_mlp)
    model_new.load_state_dict(torch.load("./algoritmi/ptwidedeep/wd_model.pt"))

    def top_n_users_for_product(trained_model, product_id, user_ids_csv, wide_preprocessor, tab_preprocessor, N=10):
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
    
    def top_n_products_for_user(trained_model, user_id, product_ids_csv, wide_preprocessor, tab_preprocessor, N=10):
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

    from NN2 import FileInfo, Model
    file_infos = FileInfo("./algoritmi/ptwidedeep/model.pt", "./algoritmi/ptwidedeep/wd_model.pt", "./algoritmi/ptwidedeep/WidePreprocessor.pkl", "./algoritmi/ptwidedeep/TabPreprocessor.pkl", "./algoritmi/ptwidedeep/data_preprocessed_NN.csv", "./preprocessor/exported_csv/anacli.csv", "./preprocessor/exported_csv/anaart.csv")
    neural_network = Model(file_infos)
    neural_network.train_model()
    
    # Example usage:
    product_id = 1102055  # Example product ID
    user_ids_csv = 'preprocessor/exported_csv/anacli.csv'  # Path to the CSV file containing user IDs
    top_users = top_n_users_for_product(trained_model=neural_network.trainer, product_id=product_id, 
                                        user_ids_csv=user_ids_csv,
                                        wide_preprocessor=neural_network.wide_preprocessor, 
                                        tab_preprocessor=tab_preprocessor, N=10)
    print(f"Top {len(top_users)} possible users for product {product_id}:")
    for rank, (user_id, rating) in enumerate(top_users, start=1):
        print(f"Rank {rank}: User ID: {user_id}, Predicted Rating: {rating}")
        
    # Example usage:
    user_id = 22596  # Example user ID
    product_ids_csv = 'preprocessor/exported_csv/anaart.csv'  # Path to the CSV file containing product IDs
    top_products = top_n_products_for_user(trained_model=trainer, user_id=user_id, 
                                            product_ids_csv=product_ids_csv,
                                            wide_preprocessor=wide_preprocessor, 
                                            tab_preprocessor=tab_preprocessor, N=10)
    print(f"Top {len(top_products)} recommended products for user {user_id}:")
    for rank, (product_id, rating) in enumerate(top_products, start=1):
        print(f"Rank {rank}: Product ID: {product_id}, Predicted Rating: {rating}")