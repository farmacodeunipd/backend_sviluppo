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
    
    # Creating a Dataset
    if not os.path.exists('synthetic_data.csv'):
        data = {
            "user_id": np.random.randint(1, 100, size=100),
            "product_id": np.random.randint(1, 1000, size=100),
            "rating": np.random.randint(1, 6, size=100),  # Assuming ratings are integers from 1 to 5
            "age": np.random.randint(18, 70, size=100),
            "gender": np.random.choice(["Male", "Female"], size=100),
            "occupation": np.random.choice(["Student", "Professional", "Retired"], size=100),
            "zip_code": np.random.randint(10000, 99999, size=100),  # Assuming zip codes are integers
        }
        df = pd.DataFrame(data)
        df.to_csv('synthetic_data.csv', index=False)
        print("created Data")
    else:
        data = pd.read_csv('synthetic_data.csv')
        df = pd.DataFrame(data)
        print("loaded Data")

    # Split the data into train and test sets
    df_train, df_test = train_test_split(df, test_size=0.2)
    df_test = df

    # Define the column setup
    wide_cols = [
        "gender",
        "occupation",
        "age",
        "zip_code"
    ]
    # crossed_cols = [""]

    cat_embed_cols = [
        "product_id",
        "user_id",
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
    os.makedirs("model_weights", exist_ok=True)
    with open("model_weights/WidePreprocessor.pkl", "wb") as f:
        pickle.dump(wide_preprocessor, f)
    with open("model_weights/TabPreprocessor.pkl", "wb") as f:
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
    torch.save(model.state_dict(), "model_weights/wd_model.pt")

    # 1. Build the model
    model_new = WideDeep(wide=wide, deeptabular=tab_mlp)
    model_new.load_state_dict(torch.load("model_weights/wd_model.pt"))


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