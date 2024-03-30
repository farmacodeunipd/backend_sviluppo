from surprisedir.Matrix import SVD_FileInfo, SVD_Model
from ptwidedeep.NN2 import NN_FileInfo, NN_Model
from Algo import ModelContext

# Create SVD model and file info
svd_file_info = SVD_FileInfo(model_file='./algoritmi/surprisedir/trained_model.pkl', file_path="./algoritmi/surprisedir/data_preprocessed_matrix.csv", column_1='cod_cli', column_2='cod_art', column_3='rating')
svd_model = SVD_Model(file_info=svd_file_info)

# Create NN model and file info
nn_file_info = NN_FileInfo("./algoritmi/ptwidedeep/model.pt", "./algoritmi/ptwidedeep/wd_model.pt", "./algoritmi/ptwidedeep/WidePreprocessor.pkl", "./algoritmi/ptwidedeep/TabPreprocessor.pkl", "./algoritmi/ptwidedeep/data_preprocessed_NN.csv", "./algoritmi/preprocessor/exported_csv/anacli.csv", "./algoritmi/preprocessor/exported_csv/anaart.csv")
nn_model = NN_Model(file_info=nn_file_info, epochs_n=5)

# Create a single model context
context = ModelContext(svd_model)

# Train and use SVD model
context.train_model()
top_items = context.topN_1UserNItem(13, 20)
print(f"Top {len(top_items)} possible items for user {13} using SVD:")
for rank, (item_id, rating) in enumerate(top_items, start=1):
    print(f"Rank {rank}: Item ID: {item_id}, Predicted Rating: {rating}")

# Switch to NN model
context.set_model_info(nn_model)

# Train and use NN model
context.train_model()
top_items = context.topN_1UserNItem(13, 20)
print(f"Top {len(top_items)} possible items for user {13} using NN:")
for rank, (item_id, rating) in enumerate(top_items, start=1):
    print(f"Rank {rank}: Item ID: {item_id}, Predicted Rating: {rating}")
