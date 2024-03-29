from surprisedir.Matrix import SVD_FileInfo, SVD_Model
from ptwidedeep.NN2 import NN_FileInfo, Model

#SVD
print("SVD\n")
svd = SVD_Model(file_info=SVD_FileInfo(model_file='./algoritmi/surprisedir/trained_model.pkl', file_path="./algoritmi/surprisedir/data_preprocessed_matrix.csv", column_1='cod_cli', column_2='cod_art', column_3='rating'))
svd.train_model()

top_items = svd.topN_1UserNItem(13, 20)
print(f"Top {len(top_items)} possible items for user {13}:")
for rank, (item_id, rating) in enumerate(top_items, start=1):
    print(f"Rank {rank}: User ID: {item_id}, Predicted Rating: {rating}")

top_users = svd.topN_1ItemNUser(1215051, 20)
print(f"Top {len(top_users)} possible users for product {1215051}:")
for rank, (user_id, rating) in enumerate(top_users, start=1):
    print(f"Rank {rank}: User ID: {user_id}, Predicted Rating: {rating}")
    
#NN
print("NN\n")
file_infos = NN_FileInfo("./algoritmi/ptwidedeep/model.pt", "./algoritmi/ptwidedeep/wd_model.pt", "./algoritmi/ptwidedeep/WidePreprocessor.pkl", "./algoritmi/ptwidedeep/TabPreprocessor.pkl", "./algoritmi/ptwidedeep/data_preprocessed_NN.csv", "./preprocessor/exported_csv/anacli.csv", "./preprocessor/exported_csv/anaart.csv")
neural_network = Model(file_infos, epochs_n=5)
neural_network.train_model()

top_items = neural_network.topN_1UserNItem(13, 20)
print(f"Top {len(top_items)} possible items for user {13}:")
for rank, (item_id, rating) in enumerate(top_items, start=1):
    print(f"Rank {rank}: User ID: {item_id}, Predicted Rating: {rating}")

top_users = neural_network.topN_1ItemNUser([1215051,"12","5Z","LP"], 20)
print(f"Top {len(top_users)} possible users for product {1215051}:")
for rank, (user_id, rating) in enumerate(top_users, start=1):
    print(f"Rank {rank}: User ID: {user_id}, Predicted Rating: {rating}")