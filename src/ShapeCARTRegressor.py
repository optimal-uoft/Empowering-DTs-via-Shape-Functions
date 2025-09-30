import heapq
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.BranchingTreeRegressor import BranchingTreeRegressor
from src.BiTAO import BiCARTClassifier, BiTAOClassifier
from sklearn.metrics import accuracy_score
import contextlib
import io
from line_profiler import profile
import gc
import random
import time
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
class ShapeCARTRegressor:
    def __init__(self,
                 # Outer Tree Params
                 max_depth: int = None, 
                 min_samples_split: int = None,
                 min_impurity_decrease: float = 0.0,
                 min_samples_leaf: int = 1,
                 max_leaf_nodes: int = None,
                 criterion: str = 'squared_error',
                 use_kmeans: bool = False,
                 # Inner Tree Params
                 inner_max_depth: int = 6,
                 inner_min_samples_split: int = 2,
                 inner_min_impurity_decrease: float = 0.0,
                 inner_min_samples_leaf: int = 1,
                 inner_max_leaf_nodes: int = 32,
                 smart_init: bool = True,
                 max_iter: int = 20,
                 inner_splitter: str = 'best',
                 # tao params
                 use_tao: bool = False,
                 n_runs: int = 20,
                 tao_reg:int = 0,
                 tao_pair_scale: float = 1.1,
                 # pairwise control
                 pairwise_candidates: int = 0,
                 pairwise_penalty: float = 0.0,
                 random_pairs: bool = False,
                 H: int = 5,
                 # multiway Params
                 k: int = 2,
                 branching_penalty: float = 0.0,
                 random_state: int = 42,
                 verbose: bool = False,
                 max_features = None):
        np.random.seed(random_state)
        random.seed(random_state)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.inner_max_depth = inner_max_depth
        self.inner_min_samples_split = inner_min_samples_split
        self.inner_min_impurity_decrease = inner_min_impurity_decrease
        self.inner_min_samples_leaf = inner_min_samples_leaf
        self.inner_max_leaf_nodes = inner_max_leaf_nodes
        self.pairwise_candidates = pairwise_candidates
        self.pairwise_penalty = pairwise_penalty
        self.k = k
        self.use_kmeans = use_kmeans
        self.random_state = random_state
        self.verbose = verbose
        self.max_iter = max_iter
        self.smart_init = smart_init
        self.tao_reg = tao_reg
        self.tao_pair_scale = tao_pair_scale
        self.use_tao = use_tao
        self.n_runs = n_runs
        self.H = H
        self.branching_penalty = branching_penalty
        self.criterion = criterion
        self.inner_splitter = inner_splitter
        self.random_pairs = random_pairs
        if self.min_samples_split is None:
            self.min_samples_split = 2
        if self.min_samples_leaf is None:
            self.min_samples_leaf = 1
        self.max_features = max_features
        self.nodes = [] # list of nodes
        self.values = [] # distribution of classes at each node
        self.impurity = [] # impurity of each node
        self.point_idxs = [] # indices of points at each node
        self.is_leaf = [] # is the node a leaf
        self.children = [] # children of each node
        self.parents = [] # parent of each node
        self.depths = [] # depth of each node
        self.n_samples = []
        self.impurity_reduction = []
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def verbose_print(self, msg):
        if self.verbose:
            print(msg)

    def configure_feature_dict(self, X, feature_dict):
        """
        feature_dict: {key: list of indices]}
        """
        if feature_dict is None:
            self.verbose_print('No feature dict provided, assuming all features are continous')
            index_dict = {i:[i] for i in range(X.shape[1])}
            cat_dict = {i: False for i in range(X.shape[1])}
            return index_dict, cat_dict
        else:
            # index_dict = {key: feature_dict[key][1] for key in feature_dict.keys()}
            # cat_dict = {key: feature_dict[key][0] for key in feature_dict.keys()}
            # check the make sure there are no repeated indices
            index_dict = feature_dict.copy()
            all_cat_idxs = []
            for val in index_dict.values():
                all_cat_idxs.extend(val)

            assert len(all_cat_idxs) == len(set(all_cat_idxs)), 'Feature indices must be unique'
            for i in range(X.shape[1]): # fill in the gaps, assume non-categorical
                if i not in all_cat_idxs: # if the index isnt in the feature dict, add it as a non-categorical
                    index_dict[i] = [i]
                    self.verbose_print(f'Adding index {i} as continous')
            cat_dict = {}
            for k, v in feature_dict.items():
                if len(v) > 1:
                    cat_dict[k] = True
                else:
                    cat_dict[k] = False
            cat_list = [k for k, v in cat_dict.items() if v]
            cont_list = [k for k, v in cat_dict.items() if not v]
            self.verbose_print(f"Categorical features: {cat_list}, Continuous features: {cont_list}")
            return index_dict, cat_dict
    
    def data_assert(self, X, y):
        assert len(X) == len(y), 'X and y must have the same length'
        assert len(y.shape) == 2, 'y must be a 1D array'
        assert len(X.shape) == 2, 'X must be a 2D array'
        assert len(y) > 1, 'X and y must have at least 2 samples'  
    @profile
    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, feature_dict: dict = None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        # print(feature_dict)
        self.data_assert(X, y)
        #check if min_samples_split is a float
        if isinstance(self.min_samples_split, float):
            self.min_samples_split = int( np.ceil(self.min_samples_split * len(X)))
        # feature_dict format: {key: (is_categorical, [list of indices])}
        self.feature_dict, self.categorical_dict = self.configure_feature_dict(X, feature_dict)

        # init_distribution = 
        # dist_sum = np.sum(init_distribution)
        init_distribution = np.mean(y)
        self.verbose_print(f'Initial distribution: {init_distribution}')
        self.nodes = [None]
        self.values = [init_distribution]
        self.impurity = [None]
        self.point_idxs = [np.arange(len(y))]
        self.is_leaf = [True]
        self.children = [None]
        self.parents = [None]
        self.depths = [0]
        self.n_samples = [len(y)]
        self.impurity_reduction = [0.0]
        self.n_leaves = 1
        k = self.k
        tree_heap = []
        self.verbose_print('Fitting initial tree -----------------------------------------------------')
        init_tree = BranchingTreeRegressor(
            feature_dict = self.feature_dict,
            cat_dict = self.categorical_dict,
            max_depth = self.inner_max_depth,
            min_samples_split = self.inner_min_samples_split,
            min_impurity_decrease = self.inner_min_impurity_decrease,
            min_samples_leaf = self.inner_min_samples_leaf,
            max_leaf_nodes = self.inner_max_leaf_nodes,
            criterion = self.criterion,
            k = self.k,
            use_kmeans= self.use_kmeans,
            outer_min_samples_leaf = self.min_samples_leaf,
            outer_min_impurity_decrease = self.min_impurity_decrease,
            pairwise_candidates = self.pairwise_candidates,
            pairwise_penalty= self.pairwise_penalty,
            random_state = self.random_state,
            verbose = self.verbose,
            idx = 0,
            smart_init=self.smart_init,
            max_iter=self.max_iter,
            H=self.H,
            random_pairs=self.random_pairs,
            splitter=self.inner_splitter,
            branching_penalty=self.branching_penalty,
            max_features=self.max_features,
        )
        
        init_tree.fit(X,y)
        if init_tree.failed:
            self.verbose_print('Initial tree failed')
            self.impurity_reduction[0] = 0.0
            return None
        else:
            self.impurity_reduction[0] = init_tree.best_impurity_decrease
    
        heapq.heappush(tree_heap, (-init_tree.best_impurity_decrease, init_tree))
        
        n_iters = 0
        while True:
            if len(tree_heap) == 0:
                self.verbose_print('No more trees to branch')
                break
            else:
                self.verbose_print(f'Heap size: {len(tree_heap)}')
            # check number of leaves we can create and adjust if needed
            if self.max_leaf_nodes is not None:
                allowance = self.max_leaf_nodes - self.n_leaves
                if allowance == 0:
                    break # we can't create any more leaves
                if allowance + 1< k:
                    break 
            
            _, branching_tree = heapq.heappop(tree_heap)
            node_idx = branching_tree.idx
            self.nodes[node_idx] = branching_tree
            self.children[node_idx] = []
            self.is_leaf[node_idx] = False

            branching = branching_tree.predict(X[self.point_idxs[node_idx]])
            curr_node_solution = branching_tree.final_solution

            for i in range(branching_tree.best_k):
                new_node_idx = len(self.nodes)
                value = curr_node_solution['value'][i]
                error = curr_node_solution['error'][i]
                weight = curr_node_solution['partition_weights'][i]
                self.verbose_print(f"\t Testing leaf {i} of node {node_idx} with {weight} samples-----------------------------------------------------")

                self.nodes.append(None)
                self.values.append(value)
                self.impurity.append(error)
                self.n_samples.append(weight)

                mask = np.where(branching == i)[0]
                rel_point_idxs = self.point_idxs[node_idx][mask]

                self.point_idxs.append(self.point_idxs[node_idx][mask])
                self.is_leaf.append(True)
                self.children.append([])
                self.parents.append(node_idx)
                self.depths.append(self.depths[node_idx] + 1)
                self.children[node_idx].append(new_node_idx)
                self.impurity_reduction.append(0)
                # now we need to check if this new node can be split
                if len(self.point_idxs[-1]) < self.min_samples_split:
                    self.verbose_print(f"\t\t Node {new_node_idx} has too few samples ({len(self.point_idxs[-1])})")
                    continue
                if error < self.min_impurity_decrease:
                    self.verbose_print(f"\t\t Node {new_node_idx} has too low error ({error}) to be improved")
                    continue
                if self.max_depth is not None and self.depths[-1] >= self.max_depth:
                    self.verbose_print(f"\t\t Node {new_node_idx} has reached max depth")
                    continue

                new_tree = BranchingTreeRegressor(
                    feature_dict = self.feature_dict,
                    cat_dict= self.categorical_dict,
                    max_depth = self.inner_max_depth,
                    criterion= self.criterion,
                    min_samples_split = self.inner_min_samples_split,
                    min_impurity_decrease = self.inner_min_impurity_decrease,
                    min_samples_leaf = self.inner_min_samples_leaf,
                    max_leaf_nodes = self.inner_max_leaf_nodes,
                    k = self.k,
                    use_kmeans= self.use_kmeans,
                    outer_min_samples_leaf = self.min_samples_leaf,
                    outer_min_impurity_decrease = self.min_impurity_decrease,
                    pairwise_candidates = self.pairwise_candidates,
                    random_state = self.random_state,
                    verbose = self.verbose,
                    idx = new_node_idx,
                    pairwise_penalty=self.pairwise_penalty,
                    max_iter=self.max_iter,
                    smart_init=self.smart_init,
                )
                new_tree.fit(X[rel_point_idxs], y[rel_point_idxs])                
                if new_tree.failed:
                    continue
                self.verbose_print(f"\t\t Node {new_node_idx} has been improved")
                old_heap = tree_heap.copy()
                try:
                    heapq.heappush(tree_heap, (-new_tree.best_impurity_decrease, new_tree))
                except TypeError:
                    for item in old_heap:
                        print(item)
                        if item[0] == -new_tree.best_impurity_decrease:
                            print(f"\t found duplicate: {item[1].idx}")
                            print(f"\t new tree: {new_tree.idx}")
                    raise Exception('Error')
                self.impurity_reduction[-1] = new_tree.best_impurity_decrease

            self.n_leaves = np.sum(self.is_leaf)
            n_iters += 1     
        
        for i in range(len(self.nodes)): # cleanup
            y_points = y[self.point_idxs[i]]
            if len(y_points) == 0:
                self.values[i] = 0
                self.n_samples[i] = 0
            else:
                dist = np.mean(y_points)
                sum = len(y_points)
                self.values[i] = dist
                self.n_samples[i] = sum
        gc.collect()
        if self.use_tao:
            self.run_tao(X, y)

    def run_tao(self, X: np.ndarray, y: np.ndarray):
        for run in range(self.n_runs):
            pred = self.predict(X)
            error = mean_squared_error(y, pred)
            self.verbose_print(f'Running TAO run {run} with error {error} -----------------------------------------------------')
            change = False
            # self.verbose_print(f'Running TAO iter {run}')
            for depths in reversed(range(self.max_depth)):
                rel_nodes = np.where(np.array(self.depths) == depths)[0]
                if len(rel_nodes) == 0:
                    continue
                # self.verbose_print(f'Running TAO at depth {depths} with {len(rel_nodes)} nodes')
                for node_idx in rel_nodes:
                    # self.verbose_print(f'\t Running TAO at node {node_idx} with {len(self.point_idxs[node_idx])} points')
                    node = self.nodes[node_idx]
                    node_idxs = self.point_idxs[node_idx]
                    if len(node_idxs) == 0:
                        # self.verbose_print(f'\t \t Node {node_idx} has no points, skipping')
                        continue
                    if node is None:
                        # self.verbose_print(f'\t \t Node {node_idx} is a leaf, skipping')
                        continue
                    
                    X_node = X[node_idxs]
                    y_node = y[node_idxs]
                    node_children = self.children[node_idx]
                    pseudolabel_matrix = []
                    pred_matrix = []
                    for _, child in enumerate(node_children):
                        child_pred = self.recurse_predict(X_node, child)
                        error = (y_node.ravel() - child_pred) **2
                        pseudolabel_matrix.append(error)
                        pred_matrix.append(child_pred)
                    pseudolabel_matrix = np.array(pseudolabel_matrix).T
                    pred_matrix = np.array(pred_matrix).T
                    # now we have a matrix of shape (n_samples, n_children) with the errors
                    # pseudolabels = np.argmin(pseudolabel_matrix, axis=1)
                    # weights = np.max(pseudolabel_matrix, axis=1) - np.min(pseudolabel_matrix, axis=1)

                    X_up, pseudolabels, weights = upsample(X_node, pseudolabel_matrix)

                    # dummy branching
                    dummy_clf = DummyClassifier(strategy='most_frequent')
                    dummy_clf.fit(X_up, pseudolabels, sample_weight=weights) # train on upsampled data
                    dummy_branching = dummy_clf.predict(X_node) # predict on original data
                    dummy_pred = pred_matrix[np.arange(len(pred_matrix)), dummy_branching]
                    dummy_error = mean_squared_error(y_node, dummy_pred)

                    curr_branching = node.predict(X_node) #
                    curr_pred = pred_matrix[np.arange(len(pred_matrix)), curr_branching]
                    curr_error = mean_squared_error(y_node, curr_pred)
                    # curr_accuracy = accuracy_score(pseudolabels, curr_branching, sample_weight=weights)
                    if node.final_key is not None and isinstance(node.final_key, tuple):
                        curr_error += self.tao_pair_scale * self.tao_reg
                    elif node.final_key is not None:
                        curr_error += self.tao_reg

                    feature_dict = node.feature_dict
                    cat_dict = node.cat_dict
                    single_features = {
                        k: v for k, v in feature_dict.items() if not isinstance(k, tuple)
                    }
                    best_single_feature = None
                    best_single_feature_error = np.inf
                    best_single_feature_tree = None 
                    for key, idxs in single_features.items():
                        clf_ = DecisionTreeClassifier(
                            max_depth=self.inner_max_depth,
                            min_samples_split=self.inner_min_samples_split,
                            min_impurity_decrease=self.inner_min_impurity_decrease,
                            min_samples_leaf=self.inner_min_samples_leaf,
                            max_leaf_nodes=self.inner_max_leaf_nodes,
                            random_state=self.random_state,
                        )
                        clf_.fit(X_up[:, idxs], pseudolabels, sample_weight=weights) # train on upsampled data
                        new_branching = clf_.predict(X_node[:, idxs]) # predict on original data
                        new_pred = pred_matrix[np.arange(len(pred_matrix)), new_branching]
                        new_error = mean_squared_error(y_node, new_pred)
                        new_error += self.tao_reg
                        if new_error < best_single_feature_error:
                            best_single_feature_error = new_error
                            best_single_feature = key
                            best_single_feature_tree = clf_
                    
                    pairwise_features = {
                        k: v for k, v in feature_dict.items() if isinstance(k, tuple)
                    }
                    best_pairwise_feature = None
                    best_pairwise_feature_error = np.inf
                    best_pairwise_feature_tree = None
                    for key, idxs in pairwise_features.items():
                        cat_dict_1 = cat_dict[key[0]]
                        cat_dict_2 = cat_dict[key[1]]
                        if cat_dict_1 or cat_dict_2:
                            clf_ = DecisionTreeClassifier(
                                max_depth=self.inner_max_depth,
                                min_samples_split=self.inner_min_samples_split,
                                min_samples_leaf=self.inner_min_samples_leaf,
                                max_leaf_nodes=self.inner_max_leaf_nodes,
                                random_state=self.random_state,
                            )
                        else:
                            clf_ = BiCARTClassifier(
                                max_depth=self.inner_max_depth,
                                min_samples_split=self.inner_min_samples_split,
                                min_samples_leaf=self.inner_min_samples_leaf,
                                max_leaf_nodes=self.inner_max_leaf_nodes,
                                random_state=self.random_state,
                                H = self.H,
                            )
                        clf_.fit(X_up[:, idxs], pseudolabels, sample_weight=weights) 
                        new_branching = clf_.predict(X_node[:, idxs]).astype(int) # predict on original data
                        new_pred = pred_matrix[np.arange(len(pred_matrix)), new_branching]
                        new_error = mean_squared_error(y_node, new_pred)
                        new_error += self.tao_pair_scale * self.tao_reg
                        if new_error < best_pairwise_feature_error:
                            best_pairwise_feature_error = new_error
                            best_pairwise_feature = key
                            best_pairwise_feature_tree = clf_
                    best_key = None
                    best_tree = None
                    node_change = False
                    if dummy_error < curr_error and \
                        dummy_error < best_single_feature_error and \
                        dummy_error < best_pairwise_feature_error:
                        
                        self.verbose_print(f'\t\t\t Node {node_idx} is better with dummy classifier: {dummy_error}. curr_accuracy: {curr_error}')
                        best_key  = None
                        best_tree = dummy_clf
                        node_change = True
                    elif best_single_feature_error < curr_error and \
                        best_single_feature_error < dummy_error and \
                        best_single_feature_error < best_pairwise_feature_error:
                        self.verbose_print(f'\t\t\t Node {node_idx} is better with single feature: {best_single_feature_error}. curr_accuracy: {curr_error}')
                        best_key = best_single_feature
                        best_tree = best_single_feature_tree
                        node_change = True
                    elif best_pairwise_feature_error < curr_error and \
                        best_pairwise_feature_error < dummy_error and \
                        best_pairwise_feature_error < best_pairwise_feature_error:
                        self.verbose_print(f'\t\t\t Node {node_idx} is better with pairwise feature: {best_pairwise_feature_error}. curr_accuracy: {curr_error}')
                        best_key = best_pairwise_feature
                        best_tree = best_pairwise_feature_tree
                        node_change = True
                    else:
                        self.verbose_print(f'\t\t\t Node {node_idx} is not better than current: {curr_error}, dummy: {dummy_error}, single: {best_single_feature_error}, pairwise: {best_pairwise_feature_error}')
                    
                    if node_change:
                        node.update_tree(
                            best_key, 
                            best_tree
                        )
                        change = True
                        self.recurse_predict_and_recalc(X, y, node_idx, self.point_idxs[node_idx])
            if not change:
                self.verbose_print(f'No changes in TAO run {run}, stopping')
                break

    def _subtree_at(self, node_idx):
        """
        Returns the subtree at the given node index.
        """
        if self.is_leaf[node_idx]:
            return [node_idx]
        else:
            subtree = [node_idx]
            for child in self.children[node_idx]:
                subtree.extend(self._subtree_at(child))
            return subtree
        
    def recurse_predict_and_recalc(self, X_, y_, node_idx, point_idxs):
        rel_points = X_[point_idxs]
        y_points = y_[point_idxs]
        self.point_idxs[node_idx] = point_idxs
        if len(y_points) == 0:
            self.values[node_idx] = 0
            self.n_samples[node_idx] = 0
            self.impurity[node_idx] = 0
            self.is_leaf[node_idx] = True
            return
        else:
            dist = np.mean(y_points)
            sum = len(y_points)
            self.values[node_idx] = dist
            self.n_samples[node_idx] = sum
            self.impurity[node_idx] = np.var(y_points)
        if len(rel_points) == 0:
            return 
        node = self.nodes[node_idx]
        if node is None:
            return # this is a leaf
        node_pred = node.predict(rel_points)
        unique_pred = np.unique(node_pred)
        for child in unique_pred:
            rel_child_idx = self.children[node_idx][child]
            child_mask = node_pred == child
            new_point_idxs = point_idxs[child_mask]
            self.recurse_predict_and_recalc(X_, y_, rel_child_idx, new_point_idxs)

    def recurse_predict(self, X, node):
        try: 
            self.is_leaf[node]
        except IndexError:
            print(node)
            raise Exception('Error')

        if self.is_leaf[node]:
            return np.ones(X.shape[0]).astype(int) * self.values[node]
        else:
            node_children = self.children[node]
            node_pred = self.nodes[node].predict(X)
            result = np.zeros(X.shape[0]).astype(np.float32)
            for i, child in enumerate(node_children):
                mask = node_pred == i
                mask = mask.astype(np.float32)
                result += mask * self.recurse_predict(X, child)
            return result
    
    def recurse_predict_limit(self, X, node, depth_limit):
        if self.depths[node] > depth_limit:
            raise ValueError('Depth limit must be greater than or equal to the depth of the node')
        if self.is_leaf[node]:
            return np.ones(X.shape[0]).astype(np.float32) * self.values[node]
        elif self.depths[node] == depth_limit:
            return np.ones(X.shape[0]).astype(np.float32) * self.values[node]
        else:
            node_children = self.children[node]
            node_pred = self.nodes[node].predict(X)
            result = np.zeros(X.shape[0]).astype(np.float32)
            for i, child in enumerate(node_children):
                mask = node_pred == i
                result += mask * self.recurse_predict_limit(X, child, depth_limit=depth_limit)
            return result
    
    def predict(self, X, max_depth=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if max_depth is not None:
            predictions =  self.recurse_predict_limit(X, 0, max_depth)
        else:
            predictions = self.recurse_predict(X, 0)
        return predictions.reshape(-1, 1)



def upsample(X, pseudolabel_matrix):
    X_list = []
    y_list = []
    sample_weights = []
    rank_matrix = pseudolabel_matrix.argsort(axis=1).argsort(axis=1) 
    max_error = np.max(pseudolabel_matrix, axis=1) # get worst branch error
    for i in range(X.shape[0]):
        for label in range(pseudolabel_matrix.shape[1]):
            if rank_matrix[i, label]+1 == pseudolabel_matrix.shape[1]:
                continue
            X_list.append(X[i])
            y_list.append(label)
            sample_weights.append(max_error[i] - pseudolabel_matrix[i, label])  # Weight for each label
            # calculated as the difference from the maximum error 
    X_new = np.vstack(X_list)
    y_new = np.array(y_list)
    sample_weights = np.array(sample_weights)
    return X_new, y_new, sample_weights