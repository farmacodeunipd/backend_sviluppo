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

    # TOPN #
    # needs to load model, get all userids and all productids, create dataframes accordingly and then act with the topN
    # Prepare user information and products of interest
    #user_info = {
    #    "cod_cli": 13,
    #    "cod_art": None,
    #    "cod_linea_comm": None,
    #    "cod_sett_comm": None,
    #    "cod_fam_comm": None,
    #}
    users_of_interest = [1112834, 1112835, 1211025, 1211071, 1211098, 1211099]  # List of product IDs

    # Convert user info to a DataFrame
    product_df = pd.DataFrame({"cod_cli": users_of_interest,
        "cod_art": 1112834,
        "cod_linea_comm": "11",
        "cod_sett_comm": "3I",
        "cod_fam_comm": "D4",})

    # Preprocess the user's information
    X_product_wide = wide_preprocessor.transform(product_df)
    X_product_tab = tab_preprocessor.transform(product_df)

    # Preprocess the products of interest
    products_df = pd.DataFrame({"cod_cli": users_of_interest, 
        "cod_art": 32, 
        "cod_linea_comm": None,
        "cod_sett_comm": None,
        "cod_fam_comm": None,})
    X_users_wide = wide_preprocessor.transform(products_df)
    X_users_tab = tab_preprocessor.transform(products_df)

    # Make predictions for the users and product
    product_ratings_predictions = trainer.predict(X_wide=X_product_wide, X_tab=X_product_tab, batch_size=64)
    user_ratings_predictions = trainer.predict(X_wide=X_users_wide, X_tab=X_users_tab, batch_size=64)

    # Combine product IDs with their predicted ratings
    user_ratings = list(zip(users_of_interest, product_ratings_predictions))

    # Sort the products by predicted ratings in descending order
    N = 10
    top_n_ratings = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:N]

    print("Top N user ratings:")
    for user_id, rating in top_n_ratings:
        print(f"Product ID: {user_id}, Predicted Rating: {rating}")