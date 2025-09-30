from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import numpy as np
import numpy as np
import pandas as pd
import os
from sklearn.datasets import fetch_openml
import pickle

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None
import sys
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def process_dataframe(df):
    """
    Process a dataframe with a mix of numerical, binary, and categorical variables.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of the target column (if any). If provided, it will be excluded from processing.
        binary_threshold (int): Threshold for treating numerical columns as binary.

    Returns:
        pd.DataFrame: Processed dataframe with one-hot encoded categorical variables.
        dict: Dictionary mapping original features to relevant columns.
    """
    df = df.copy()

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    numerical_columns = [col for col in df.columns if col not in categorical_columns]

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    df_encoded_columns = list(df_encoded.columns)

    mapping = {}

    for col in numerical_columns:
        mapping[col] = [df_encoded_columns.index(col)]
    
    
    for col in categorical_columns:
        mapping[col] = [df_encoded_columns.index(c) for c in df_encoded_columns if c.startswith(f"{col}_")]


    # Return processed dataframe, target, and mapping
    return df_encoded, mapping

class DataFactory_clf:
    def __init__(self, 
                 dataset,
                 train_ratio = 0.7,
                 val_ratio = 1.0/3.0, # results in a 70/10/20 train/val/test split
                 seeds = [42, 63, 84],
                 cache = True):
        self.dataset = dataset
        self.cache = cache
        fetch_func = self.fetch_function(dataset)
        if self.cache:
            # check if we have the data cached
            cache_x = f'data/X_{dataset}.pkl'
            cache_y = f'data/y_{dataset}.pkl'
            cache_feature_dict = f'data/feature_dict_{dataset}.pkl'
            if os.path.exists(cache_x) and os.path.exists(cache_y):
                self.X = pd.read_pickle(cache_x)
                with open(cache_y, 'rb') as f:
                    self.y = pickle.load(f)
                with open(cache_feature_dict, 'rb') as f:
                    self.feature_dict = pickle.load(f)
            else:
                self.X, self.y, self.feature_dict = fetch_func()
                if not os.path.exists('data'):
                    os.makedirs('data')
                self.X.to_pickle(cache_x)
                with open(cache_y, 'wb') as f:
                    pickle.dump(self.y, f)
                with open(cache_feature_dict, 'wb') as f:
                    pickle.dump(self.feature_dict, f)
        else:
            self.X, self.y, self.feature_dict = fetch_func()
        self.n_features = self.X.shape[1]
        self.n_classes = len(np.unique(self.y))
        self.n_points = self.X.shape[0]
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seeds = seeds
        self.n_folds = len(self.random_seeds)

    def get_data(self, fold_idx=0):
        assert fold_idx < len(self.random_seeds), f'fold_idx must be less than {len(self.random_seeds)}'
        X_train, X_test_val, y_train, y_test_val = train_test_split(self.X, self.y, train_size=self.train_ratio, random_state=self.random_seeds[fold_idx])
        X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, train_size=self.val_ratio, random_state=self.random_seeds[fold_idx])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)        

        return X_train, y_train, X_val, y_val, X_test, y_test

    def fetch_function(self, dataset):
        function_dict = {
            'adult': self.fetch_adult, # use this
            'covtype': self.fetch_covtype, # Use this
            'mushroom': self.fetch_mushroom, # use this
            'pendigits': self.fetch_pendigits, # use this
            'bean': self.fetch_bean, # use this
            'mini-boone': self.fetch_mini_boone, 
            'electricity': self.fetch_electricity,
            'eye-movements': self.fetch_eye_movements, # REALLY GOOD
            'gas-drift': self.fetch_gasdrift, 
            'higgs': self.fetch_higgs,
            'eye-state': self.fetch_eye_state,
            'breast-cancer': self.fetch_breast_cancer,
            'credit': self.fetch_credit,
            'eucalyptus': self.fetch_eucalyptus,
            'spambase': self.fetch_spambase,
            'htru': self.fetch_htru,
            'avila': self.fetch_avila,
            'magic': self.fetch_magic,
            'skin': self.fetch_skin,
            'fault': self.fetch_fault,
            'page': self.fetch_page,
            'segment': self.fetch_segment,
            'wilt': self.fetch_wilt,
            'bidding': self.fetch_bidding,
            'raisin': self.fetch_raisin,
            'rice': self.fetch_rice,
            'occupancy': self.fetch_occupancy,
            'jannis': self.fetch_jannis,
            'landsat': self.fetch_landsat,
            'bank': self.fetch_bank,
            'room': self.fetch_room,
            'higgs_down': self.fetch_higgs_down,
        }
        if dataset in function_dict:
            return function_dict[dataset]
        else:
            raise ValueError(f"Dataset {dataset} not found")

    def _fetch_openml_dataset(self, data_id):
        X, y = fetch_openml(data_id=data_id, as_frame=True, return_X_y=True)
        y = y.values.ravel()
        le = LabelEncoder()
        y = le.fit_transform(y)
        X, mapping = process_dataframe(X)
        return X, y, mapping

    def fetch_higgs(self):
        X, y, mapping =  self._fetch_openml_dataset(data_id=45570)
        # downsample the dataset to 1,000,000 points in a stratified manner
        X_down, _, y_down, _ = train_test_split(
            X, y, 
            train_size=10000, 
            stratify=y, 
            random_state=42
        )
        return X_down, y_down, mapping
    
    def fetch_higgs_down(self):
        X, y, mapping =  self._fetch_openml_dataset(data_id=45570)
        # downsample the dataset to 1,000,000 points in a stratified manner
        X_down, _, y_down, _ = train_test_split(
            X, y, 
            train_size=10000, 
            stratify=y, 
            random_state=42
        )
        return X_down, y_down, mapping
    def fetch_credit(self):
        return self._fetch_openml_dataset(data_id=31)
    
    def fetch_spambase(self):
        return self._fetch_openml_dataset(data_id=44)
    
    def fetch_bank(self):
        return self._fetch_openml_dataset(data_id=1462)
    
    def fetch_cmc(self):
        return self._fetch_openml_dataset(data_id=23)
    
    def fetch_landsat(self):
        return self._fetch_openml_dataset(data_id=182)

    def fetch_breast_cancer(self):
        return self._fetch_openml_dataset(data_id=1510)
    
    def fetch_phoneme(self):
        return self._fetch_openml_dataset(data_id=1489)
    
    def fetch_eucalyptus(self):
        return self._fetch_openml_dataset(data_id=43924)

    def fetch_eye_state(self):
        return self._fetch_openml_dataset(data_id=1471)

    def fetch_mini_boone(self):
        return self._fetch_openml_dataset(data_id=41150)
    
    def fetch_electricity(self):
        return self._fetch_openml_dataset(data_id=151)
    
    def fetch_eye_movements(self):
        return self._fetch_openml_dataset(data_id=1044)

    def fetch_gasdrift(self):
        return self._fetch_openml_dataset(data_id=1476)

    def fetch_pendigits(self):
        return self._fetch_openml_dataset(data_id=32)

    def fetch_jannis(self):
        return self._fetch_openml_dataset(data_id=41168)

    def fetch_htru(self):
        return self._fetch_openml_dataset(data_id=45558)
    
    def fetch_magic(self):
        return self._fetch_openml_dataset(data_id=43971)
    def fetch_skin(self):
        return self._fetch_openml_dataset(data_id=1502)
    def fetch_fault(self):
        return self._fetch_openml_dataset(data_id=40982)
    def fetch_page(self):
        return self._fetch_openml_dataset(data_id=30)
    def fetch_segment(self):
        return self._fetch_openml_dataset(data_id=40984)
    def fetch_wilt(self):
        return self._fetch_openml_dataset(data_id=40983)
    
    def fetch_avila(self):
        X,_ = fetch_openml(data_id=42932, as_frame=True, return_X_y=True)
        y = X[['10']]
        X = X.drop(columns = ['10', 'train', 'test'])
        y = y.values.ravel()
        le = LabelEncoder()
        y = le.fit_transform(y)
        X, mapping = process_dataframe(X)
        return X, y, mapping
    
    def fetch_adult(self):

        X, y = fetch_openml(data_id=179, as_frame=True, return_X_y=True)
        y = y.astype(str)       
        X.drop(columns=['fnlwgt'], inplace=True)
        # X.dropna(inplace=True)
        #
        nan_mask = X.isna().any(axis=1)
        X = X[~nan_mask]
        y = y[~nan_mask]
        y = y.values.ravel()
        le = LabelEncoder()
        y = le.fit_transform(y)
        X, mapping = process_dataframe(X)
        return X, y, mapping
    
    def fetch_covtype(self):
        X,y = fetch_openml(data_id=150, as_frame=True, return_X_y=True)
        y = y.values.ravel()
        le = LabelEncoder()
        y = le.fit_transform(y)
        X_numeric = X.iloc[:, list(range(10))]
        X_wilderness = X.iloc[:, list(range(10, 14))]
        X_soil = X.iloc[:, list(range(14, 54))]

        X_wilderness = X_wilderness.astype(int)
        X_soil = X_soil.astype(int)


        X_numeric['Wilderness_Area'] = X_wilderness.idxmax(axis=1)
        X_numeric['Soil_Type'] = X_soil.idxmax(axis=1)
        X, mapping = process_dataframe(X_numeric)
        return X, y, mapping

    def fetch_mushroom(self):
        mushroom = fetch_ucirepo(id=73) 
        
        # data (as pandas dataframes) 
        X = mushroom.data.features 
        X = X.drop(columns=['veil-type'])
        y = mushroom.data.targets
        y = y.values.ravel()

        le = LabelEncoder()
        y = le.fit_transform(y)
        X, mapping = process_dataframe(X)
        return X, y, mapping
    
    def fetch_bidding(self):
        X, y = fetch_openml(data_id=42889, as_frame=True, return_X_y=True)
        y = y.values.ravel()
        le = LabelEncoder()
        y = le.fit_transform(y)
        X = X.drop(columns=['Record_ID', 'Auction_ID', 'Bidder_ID'])
        X, mapping = process_dataframe(X)
        return X, y, mapping


    def fetch_raisin(self):
        
        # fetch dataset 
        raisin = fetch_ucirepo(id=850) 
        
        # data (as pandas dataframes) 
        X = raisin.data.features 
        y = raisin.data.targets 
        
        le = LabelEncoder()
        y = le.fit_transform(y)
        X, mapping = process_dataframe(X)
        return X, y, mapping

    def fetch_rice(self):
        
        # fetch dataset 
        rice = fetch_ucirepo(id=545) 
        
        # data (as pandas dataframes) 
        X = rice.data.features 
        y = rice.data.targets 
        
        le = LabelEncoder()
        y = le.fit_transform(y)
        X, mapping = process_dataframe(X)
        return X, y, mapping

    def fetch_bean(self):
        
        # fetch dataset 
        dry_bean = fetch_ucirepo(id=602) 
        
        # data (as pandas dataframes) 
        X = dry_bean.data.features 
        # y = dry_bean.data.targets 
        
        y = dry_bean.data.targets
        y = y.values.ravel()

        le = LabelEncoder()
        y = le.fit_transform(y)
        X, mapping = process_dataframe(X)

        return X, y, mapping

    def fetch_room(self):
        
        # fetch dataset 
        room = fetch_ucirepo(id=864) 
        
        # data (as pandas dataframes) 
        X = room.data.features 
        print(X.columns)
        X = X.drop(columns=['Date', 'Time'])
        y = room.data.targets
        y = y.values.ravel()
        le = LabelEncoder()
        y = le.fit_transform(y)
        X, mapping = process_dataframe(X)

        return X, y, mapping

    def fetch_occupancy(self):
        
        # fetch dataset 
        occupancy = fetch_ucirepo(id=357) # occupancy detection dataset
        
        # data (as pandas dataframes) 
        X = occupancy.data.features
        X = X.drop(columns=['date'])
        mask = X['Temperature'] != 'Humidity'
        X = X[mask]
        X = X.astype(float)
        y = occupancy.data.targets
        y = y[mask]
        y = y.values.ravel()

        le = LabelEncoder()
        y = le.fit_transform(y)
        X, mapping = process_dataframe(X)

        return X, y, mapping

class DataFactory_rgr:
    def __init__(self, 
                 dataset,
                 train_ratio = 0.7,
                 val_ratio = 1.0/3.0, 
                 seeds = [42, 63, 84],
                 cache = True):
        self.dataset = dataset
        self.cache = cache
        fetch_func = self.fetch_function(dataset)
        if self.cache:
            # check if we have the data cached
            cache_x = f'data/X_{dataset}.pkl'
            cache_y = f'data/y_{dataset}.pkl'
            cache_feature_dict = f'data/feature_dict_{dataset}.pkl'
            if os.path.exists(cache_x) and os.path.exists(cache_y) and os.path.exists(cache_feature_dict):
                self.X = pd.read_pickle(cache_x)
                with open(cache_y, 'rb') as f:
                    self.y = pickle.load(f)
                with open(cache_feature_dict, 'rb') as f:
                    self.feature_dict = pickle.load(f)
            else:
                self.X, self.y, self.feature_dict = fetch_func()
                if not os.path.exists('data'):
                    os.makedirs('data')
                self.X.to_pickle(cache_x)
                with open(cache_y, 'wb') as f:
                    pickle.dump(self.y, f)
                with open(cache_feature_dict, 'wb') as f:
                    pickle.dump(self.feature_dict, f)
        else:
            self.X, self.y, self.feature_dict = fetch_func()

        self.n_features = self.X.shape[1]
        self.n_classes = len(np.unique(self.y))
        self.n_points = self.X.shape[0]
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seeds = seeds
        self.n_folds = len(self.random_seeds)

    def _fetch_openml_dataset(self, data_id):
        X, y = fetch_openml(data_id=data_id, as_frame=True, return_X_y=True)
        y = y.values.ravel()
        le = StandardScaler()
        y = le.fit_transform(y.reshape(-1,1))
        X, mapping = process_dataframe(X)
        return X, y, mapping

    def get_data(self, fold_idx=0):
        assert fold_idx < len(self.random_seeds), f'fold_idx must be less than {len(self.random_seeds)}'
        X_train, X_test_val, y_train, y_test_val = train_test_split(self.X, self.y, train_size=self.train_ratio, random_state=self.random_seeds[fold_idx])
        X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, train_size=self.val_ratio, random_state=self.random_seeds[fold_idx])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)        

        return X_train, y_train, X_val, y_val, X_test, y_test

    def fetch_function(self, dataset):
        function_dict = {
            'california': self.fetch_california,
            'black-friday': self.fetch_black_friday,
            'diamonds': self.fetch_diamonds,
            'ailerons': self.fetch_ailerons,
            'house16H': self.fetch_house_16H,
            'bikeshare': self.fetch_bikeshare,
            'sulfur': self.fetch_sulfur,
            'kin8nm': self.fetch_kin8nm,
            'space-ga': self.fetch_space_ga,
            'quake': self.fetch_quake,
            'cpu-act': self.fetch_cpu_act,
            'munich': self.fetch_munich,
            'delta-elevators': self.fetch_delta_elevators,
        }
        if dataset in function_dict:
            return function_dict[dataset]
        else:
            raise ValueError(f"Dataset {dataset} not found")

    def fetch_ailerons(self):
        return self._fetch_openml_dataset(data_id=44137)
    def fetch_house_16H(self):
        return self._fetch_openml_dataset(data_id=44139)
    def fetch_bikeshare(self):
        return self._fetch_openml_dataset(data_id=44063)
    def fetch_black_friday(self):
        return self._fetch_openml_dataset(data_id=41540)
    def fetch_diamonds(self):
        return self._fetch_openml_dataset(data_id=42225)
    def fetch_california(self):
        return self._fetch_openml_dataset(data_id=44025)
    def fetch_sulfur(self):
        return self._fetch_openml_dataset(data_id=23515)
    def fetch_kin8nm(self):
        return self._fetch_openml_dataset(data_id=189)
    def fetch_wine_quality(self):
        return self._fetch_openml_dataset(data_id=287)
    def fetch_space_ga(self):
        return self._fetch_openml_dataset(data_id=507)
    def fetch_brazil_house_prices(self):
        return self._fetch_openml_dataset(data_id=42688)
    def fetch_moneyball(self):
        return self._fetch_openml_dataset(data_id=41021)
    def fetch_quake(self):
        return self._fetch_openml_dataset(data_id=209)
    def fetch_cpu_act(self):
        return self._fetch_openml_dataset(data_id=197)
    def fetch_munich(self):
        return self._fetch_openml_dataset(data_id=46772)
    def fetch_delta_elevators(self):
        return self._fetch_openml_dataset(data_id=198)