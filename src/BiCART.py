import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
import itertools
from sklearn.preprocessing import StandardScaler


# implementation of BiCART from the paper "Bivariate Decision Trees: Smaller, Interpretable, More Accurate" https://dl.acm.org/doi/abs/10.1145/3637528.3671903 

def generate_line_orientations(H):
    """
    Generate a matrix W of line orientations sampled uniformly
    by rotating in two dimensions within 0 to 180 degrees.

    Parameters:
        H (int): Number of orientations to sample.

    Returns:
        numpy.ndarray: A matrix of shape (2, H) with orientations.
    """
    # Generate H angles uniformly between 0 and 180 degrees
    angles = np.linspace(0, 180, H, endpoint=True)  
    angles = np.deg2rad(angles)  # Convert degrees to radians

    
    # Create the 2xH matrix where each column represents an orientation vector
    W = np.array([[np.cos(angle), np.sin(angle)] for angle in angles]).T
    W = W.round(3)
    # drop columns where one of the values is 0
    W_mask = np.logical_or(W[0] == 0, W[1] == 0)
    W = W[:, ~W_mask]
    return W



class BiCART_:
    def __init__(self,
                 predictor = DecisionTreeClassifier,
                 max_depth: int = 5, 
                 min_samples_split: int = 2,
                 min_impurity_decrease: float = 0.0,
                 min_samples_leaf: int = 1,
                 max_leaf_nodes: int = None,
                 criterion: str = 'gini',
                 H: int = 8,
                 eps: float = 1e-7,
                 verbose: bool = False,
                 random_state: int = 42,
                 ccp_alpha: float = 0.0,
                 splitter: str = 'best'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.criterion = criterion
        self.H = H
        self.eps = eps
        self.verbose = verbose
        self.ccp_alpha = ccp_alpha
        self.splitter = splitter
        self.W = generate_line_orientations(H)
        self.random_state = random_state
        self.tree = predictor(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          min_impurity_decrease=self.min_impurity_decrease,
                                          criterion=self.criterion,
                                          min_samples_leaf=self.min_samples_leaf,
                                          max_leaf_nodes=self.max_leaf_nodes,
                                          ccp_alpha=self.ccp_alpha,
                                          splitter=self.splitter,
                                          random_state=self.random_state)
        self.scaler = StandardScaler()
    def data_assert(self, X, y):
        assert len(X) == len(y), 'X and y must have the same length'
        assert len(X.shape) == 2, 'X must be a 2D array'
        assert len(y) > 1, 'X and y must have at least 2 samples'

    def fit(self, X, y, sample_weight=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.data_assert(X, y)
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        
        # create augmented data matrix
        pairwise_combinations = list(itertools.combinations(range(X.shape[1]), 2))
        X_aug = [X]
        if self.H > 3:
            for i, j in pairwise_combinations:
                S = np.zeros((X.shape[1], 2))
                S[i, 0] = 1
                S[j, 1] = 1
                XS = X @ S
                X_i_biv = XS @ self.W
                X_aug.append(X_i_biv)
        X_aug = np.hstack(X_aug)
        self.tree.fit(X_aug, y, sample_weight=sample_weight)
        self.tree_ = self.tree.tree_
        return self
    
    def apply(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = self.scaler.transform(X)
        X_aug = [X]
        if self.H > 3:
            pairwise_combinations = list(itertools.combinations(range(X.shape[1]), 2))
            X_aug = [X]
            for i, j in pairwise_combinations:
                S = np.zeros((X.shape[1], 2))
                S[i, 0] = 1
                S[j, 1] = 1
                XS = X @ S
                X_i_biv = XS @ self.W
                X_aug.append(X_i_biv)
        X_aug = np.hstack(X_aug)
        return self.tree.apply(X_aug)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = self.scaler.transform(X)
        X_aug = [X]
        if self.H > 3:
            pairwise_combinations = list(itertools.combinations(range(X.shape[1]), 2))
            for i, j in pairwise_combinations:
                S = np.zeros((X.shape[1], 2))
                S[i, 0] = 1
                S[j, 1] = 1
                XS = X @ S
                X_i_biv = XS @ self.W
                X_aug.append(X_i_biv)
        X_aug = np.hstack(X_aug)
        X_aug = X_aug
        return self.tree.predict(X_aug)

    def score(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_aug = [X]
        if self.H > 3:
            pairwise_combinations = list(itertools.combinations(range(X.shape[1]), 2))
            for i, j in pairwise_combinations:
                S = np.zeros((X.shape[1], 2))
                S[i, 0] = 1
                S[j, 1] = 1
                XS = X @ S
                X_i_biv = XS @ self.W
                X_aug.append(X_i_biv)

        X_aug = np.hstack(X_aug)
        X_aug = X_aug
        return self.tree.score(X_aug, y)

class BiCARTRegressor(BiCART_):
    def __init__(self,
                 max_depth: int = 5, 
                 min_samples_split: int = 2,
                 min_impurity_decrease: float = 0.0,
                 min_samples_leaf: int = 1,
                 max_leaf_nodes: int = None,
                 criterion: str = 'squared_error',
                 H: int = 8,
                 eps: float = 1e-7,
                 verbose: bool = False,
                 random_state: int = 42,
                 ccp_alpha: float = 0.0,
                 splitter: str = 'best'):
        super().__init__(
                        predictor = DecisionTreeRegressor,
                        max_depth= max_depth, 
                        min_samples_split= min_samples_split,
                        min_impurity_decrease= min_impurity_decrease,
                        min_samples_leaf= min_samples_leaf,
                        max_leaf_nodes= max_leaf_nodes,
                        criterion= criterion,
                        H= H,
                        eps= eps,
                        verbose= verbose,
                        random_state= random_state,
                        ccp_alpha= ccp_alpha,
                        splitter= splitter
        )

class BiCARTClassifier(BiCART_):
    def __init__(self,
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 min_impurity_decrease: float = 0.0,
                 min_samples_leaf: int = 1,
                 max_leaf_nodes: int = None,
                 criterion: str = 'gini',
                 H: int = 8,
                 eps: float = 1e-7,
                 verbose: bool = False,
                 random_state: int = 42,
                 ccp_alpha: float = 0.0,
                 splitter: str = 'best'):
        super().__init__(
            predictor=DecisionTreeClassifier,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            criterion=criterion,
            H=H,
            eps=eps,
            verbose=verbose,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
            splitter=splitter
        )

