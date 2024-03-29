import pandas as pd
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.model_selection import cross_validate
from collections import defaultdict
import pickle
import os

from Algo import BaseFileInfo 
from Algo import BaseModel

def rating_float2int (float_rating, float_ratingMax = 2, int_ratingMax = 5, float_ratingMin = 0, int_ratingMin = 1):
      return int((float_rating - float_ratingMin) / (float_ratingMax - float_ratingMin) * (int_ratingMax - int_ratingMin)) + int_ratingMin

# classe contenitore che contiene info relative al file ed ai dati
# (model_file: path del file che memorizza il modello, file_path: nome del file contenente i dati di training, columns: nome delle colonne del file di training, scale: scala di rating (da 0 a 2 perché propria per il logaritmo usato))
class SVD_FileInfo(BaseFileInfo):
    def __init__(self, model_file, file_path, column_1, column_2, column_3, scale_min=0, scale_max=2):
        super().__init__(model_file)
        self.file_path = file_path
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.column_1 = column_1
        self.column_2 = column_2
        self.column_3 = column_3

    def load_data(self):
        df = pd.read_csv(self.file_path)
        # Assuming you have imported Dataset and Reader from the appropriate library
        reader = Reader(rating_scale=(self.scale_min, self.scale_max))
        return Dataset.load_from_df(df[[self.column_1, self.column_2, self.column_3]], reader)

# implementazione del modello e di tutti i metodi relativi al suo training ed alle operazioni effettuabili su di esso, permetto inoltre di scegliere se usare SVD o kNN e quanti k-fold, di default viene usato SVD e k=5
class SVD_Model(BaseModel):
    def __init__(self, file_info, algo='SVD', cv_n=5):
        self.file_info = file_info
        self.algo = algo
        self.cv_n = cv_n
        self.model = None
        self.data = None
        self.trainset = None
        self.testset = None

    # metodo che controlla se il file contenente la memoria del modello esiste, se esiste lo carica, altrimenti carica il modello selezionato vuoto
    def load_model(self):
        if os.path.exists(self.file_info.model_file):
            with open(self.file_info.model_file, 'rb') as file:
                self.model = pickle.load(file)
                print(f"Existing model loading successful")
        else:
            if self.algo == 'SVD':
                self.model = SVD()
                print(f"SVD loading successful")
            elif self.algo == 'kNN':
                kNN_details = {
                    'name': 'msd',
                    'user_based': True 
                }  
                self.model = KNNBasic(sim_options=kNN_details)
                print(f"kNN loading successful")

    # metodo che salva il modello nel file di memoria del modello
    def save_model(self):
        if self.model is not None:
            with open(self.file_info.model_file, 'wb') as file:
                pickle.dump(self.model, file)

    # metodo che effettua il training del modello, prima carica i dati, poi il modello, poi se il file con il modello memorizzato non esiste, effettua il training sui dati caricati e salva il modello, prepara anche un testset su cui eseguire predizioni
    def train_model(self):
        self.data = self.file_info.load_data()
        self.load_model()  

        if not os.path.exists(self.file_info.model_file):
            cross_validate(self.model, self.data, measures=['RMSE', 'MAE'], cv=self.cv_n, verbose=True)
            self.save_model()
        
        self.trainset = self.data.build_full_trainset()
        self.testset = self.trainset.build_anti_testset(fill=None)
    
    # metodo che dato ID user (NUMERICO) e n, ritorna n ID item migliori per quell'user
    def topN_1UserNItem(self, user_id, n=5):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train_model() first.")
        
        testset_filteredUI = filter(lambda x: x[0] == user_id, self.testset)
        
        predictions = self.model.test(testset_filteredUI)

        top_n = defaultdict(int)
        for uid, iid, true_r, est, _ in predictions:
            top_n[iid] = rating_float2int(est)

        top_n = sorted(top_n.items(), key=lambda x: x[1], reverse=True)[:n]
        return top_n
    
    # metodo che dato ID item (NUMERICO) e n, ritorna n ID user migliori per quell'item
    def topN_1ItemNUser(self, item_id, n=5):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train_model() first.")
        
        testset_filteredIU = filter(lambda x: x[1] == item_id, self.testset)
        
        predictions = self.model.test(testset_filteredIU)

        top_n = defaultdict(float)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid] = rating_float2int(est)

        top_n = sorted(top_n.items(), key=lambda x: x[1], reverse=True)[:n]
        return top_n