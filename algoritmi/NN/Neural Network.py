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
    if not os.path.exists('./NN/data_preprocessed_NN.csv'):
        print("Preprocessed Data are not ready or do not exist, please wait or start preprocessing the data.")
    else:
        data = pd.read_csv('./NN/data_preprocessed_NN.csv')
        df = pd.DataFrame(data)
        print("Preprocessed Data have been loaded successfully.")

    # Split the data into train and test sets
    df_train, df_test = train_test_split(df, test_size=0.2)
    df_test = df

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
    os.makedirs("./NN/model_weights", exist_ok=True)
    with open("./NN/model_weights/WidePreprocessor.pkl", "wb") as f:
        pickle.dump(wide_preprocessor, f)
    with open("./NN/model_weights/TabPreprocessor.pkl", "wb") as f:
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
        n_epochs=5,
        batch_size=64,
    )

    # Predict on test
    X_wide_te = wide_preprocessor.transform(df_test)
    X_tab_te = tab_preprocessor.transform(df_test)
    preds = trainer.predict(X_wide=X_wide_te, X_tab=X_tab_te)

    print(preds)

    # Save
    torch.save(model.state_dict(), "./NN/model_weights/wd_model.pt")

    # 1. Build the model
    model_new = WideDeep(wide=wide, deeptabular=tab_mlp)
    model_new.load_state_dict(torch.load("./NN/model_weights/wd_model.pt"))

    # TOPN #
    # needs to load model, get all userids and all productids, create dataframes accordingly and then act with the topN
    # Prepare user information and products of interest
    user_info = {
        "user_id": 32,
        "age": 23,
        "gender": "Female",
        "occupation": "Professional",
        "zip_code": 69938,
        "product_id": 733,
    }
    products_of_interest = [821, 733, 43, 100, 5, 6]  # List of product IDs

    # Convert user info to a DataFrame
    user_df = pd.DataFrame({"user_id": 32,
        "age": 23,
        "gender": "Female",
        "occupation": "Professional",
        "zip_code": 69938,
        "product_id": products_of_interest,})

    # Preprocess the user's information
    X_user_wide = wide_preprocessor.transform(user_df)
    X_user_tab = tab_preprocessor.transform(user_df)

    # Preprocess the products of interest
    products_df = pd.DataFrame({"product_id": products_of_interest, "user_id": 32, "age": None,
        "gender": None,
        "occupation": None,
        "zip_code": None,})
    X_products_wide = wide_preprocessor.transform(products_df)
    X_products_tab = tab_preprocessor.transform(products_df)

    # Make predictions for the user and products
    user_ratings_predictions = trainer.predict(X_wide=X_user_wide, X_tab=X_user_tab, batch_size=64)
    product_ratings_predictions = trainer.predict(X_wide=X_products_wide, X_tab=X_products_tab, batch_size=64)

    # Combine product IDs with their predicted ratings
    product_ratings = list(zip(products_of_interest, product_ratings_predictions))

    # Sort the products by predicted ratings in descending order
    N = 10
    top_n_ratings = sorted(product_ratings, key=lambda x: x[1], reverse=True)[:N]

    print("Top N user ratings:")
    for product_id, rating in top_n_ratings:
        print(f"Product ID: {product_id}, Predicted Rating: {rating}")