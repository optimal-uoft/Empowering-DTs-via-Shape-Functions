from sklearn.tree import DecisionTreeClassifier
import heapq
import numpy as np
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from src.BranchingTree import BranchingTree
from src.BiCART import BiCARTClassifier
from sklearn.metrics import accuracy_score
import gc
import random
from sklearn.dummy import DummyClassifier
class ShapeCARTClassifier:
    def __init__(self,
                 # Outer Tree Params
                 max_depth: int = None, 
                 min_samples_split: int = None,
                 min_impurity_decrease: float = 0.0,
                 min_samples_leaf: int = 1,
                 max_leaf_nodes: int = None,
                 criterion: str = 'gini',
                 # Inner Tree Params
                 inner_max_depth: int = 6,
                 inner_min_samples_split: int = 2,
                 inner_min_impurity_decrease: float = 0.0,
                 inner_min_samples_leaf: int = 1,
                 inner_max_leaf_nodes: int = 32,
                 max_iter: int = 20,
                 inner_splitter: str = 'best',
                 # tao params
                 use_tao: bool = False,
                 n_runs: int = 10,
                 tao_reg:int = 0,
                 tao_pair_scale: float = 1.1,
                 # pairwise control
                 pairwise_candidates: int | float = 0,
                 pairwise_penalty: float = 0.0,
                 H: int = 5,
                 # multiway Params
                 k: int = 2,
                 branching_penalty: float = 0.0,
                 random_state: int = 42,
                 verbose: bool = False,):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.criterion = criterion
        self.branching_penalty = branching_penalty
        self.inner_max_depth = inner_max_depth
        self.inner_min_samples_split = inner_min_samples_split
        self.inner_min_impurity_decrease = inner_min_impurity_decrease
        self.inner_min_samples_leaf = inner_min_samples_leaf
        self.inner_max_leaf_nodes = inner_max_leaf_nodes
        self.pairwise_candidates = pairwise_candidates
        self.pairwise_penalty = pairwise_penalty
        self.use_dpdt = False
        self.k = k
        self.random_state = random_state
        self.verbose = verbose
        self.max_iter = max_iter
        self.smart_init = True
        self.tao_reg = tao_reg
        self.use_tao = use_tao
        self.n_runs = n_runs
        self.H = H
        self.random_pairs = False
        self.internal_splitter = inner_splitter
        self.tao_pair_scale = tao_pair_scale
        
        if self.min_samples_split is None:
            self.min_samples_split = 2
        if self.min_samples_leaf is None:
            self.min_samples_leaf = 1

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
        self.n_leaves = 0
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
        assert len(y.shape) == 1, 'y must be a 1D array'
        assert len(X.shape) == 2, 'X must be a 2D array'
        assert len(y) > 1, 'X and y must have at least 2 samples'

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
        if isinstance(self.min_samples_leaf, float):
            self.min_samples_leaf = int( np.ceil(self.min_samples_leaf * len(X)))
        # feature_dict format: {key: (is_categorical, [list of indices])}
        self.feature_dict, self.categorical_dict = self.configure_feature_dict(X, feature_dict)
        if isinstance(self.pairwise_candidates, float):
            self.pairwise_candidates = int(np.ceil(self.pairwise_candidates * len(self.feature_dict)))
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        self.n_classes = len(unique_labels(y))
        
        init_distribution = np.bincount(y, minlength = self.n_classes)
        dist_sum = np.sum(init_distribution)
        init_distribution = init_distribution / dist_sum
        self.verbose_print(f'Initial distribution: {init_distribution}')

        tree_heap = []
        self.verbose_print('Fitting initial tree -----------------------------------------------------')
        self.values = [init_distribution]
        self.is_leaf = [True]
        self.children = [None]
        self.parents = [None]
        self.depths = [0]
        self.n_samples = [len(y)]
        init_tree = BranchingTree(
            feature_dict = self.feature_dict,
            cat_dict = self.categorical_dict,
            max_depth = self.inner_max_depth,
            min_samples_split = self.inner_min_samples_split,
            min_impurity_decrease = self.inner_min_impurity_decrease,
            min_samples_leaf = self.inner_min_samples_leaf,
            max_leaf_nodes = self.inner_max_leaf_nodes,
            criterion = self.criterion,
            k = self.k,
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
            use_dpdt=self.use_dpdt,
            random_pairs=self.random_pairs,
            splitter=self.internal_splitter,
            branching_penalty=self.branching_penalty,
        )
        
        init_tree.fit(X,y)
        if init_tree.failed:
            self.verbose_print('Initial tree failed')
            return None
        heapq.heappush(tree_heap, (-init_tree.best_impurity_decrease, init_tree))
        self.nodes = [None]
        
        self.impurity = [init_tree.initial_impurity]
        self.point_idxs = [np.arange(len(y))]
        
        self.impurity_reduction = [init_tree.best_impurity_decrease]
        self.n_leaves = 1
        k = self.k
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
                if allowance + 1 < k: # if we can't create k-1 leaves, break
                    break
            _, branching_tree = heapq.heappop(tree_heap)
            node_idx = branching_tree.idx
            if self.max_depth is not None and self.depths[node_idx] >= self.max_depth:
                self.verbose_print(f'Node {node_idx} has reached max depth ({self.depths[node_idx]})')
                continue
            self.nodes[node_idx] = branching_tree
            self.children[node_idx] = []
            self.is_leaf[node_idx] = False

            branching = branching_tree.predict(X[self.point_idxs[node_idx]])
            curr_node_solution = branching_tree.final_solution

            for i in range(branching_tree.best_k):
                self.verbose_print(f"\t Testing leaf {i} of node {node_idx} -----------------------------------------------------")
                new_node_idx = len(self.nodes)
                weighted_dist = curr_node_solution['weighted_distributions'][i]
                impurity = curr_node_solution['impurities'][i]
                weighted_n_samples = np.sum(weighted_dist)
                dist = weighted_dist / weighted_n_samples
                
                self.nodes.append(None)
                self.values.append(dist)
                self.impurity.append(impurity)
                self.n_samples.append(weighted_n_samples)
                mask = np.where(branching == i)[0]
                rel_point_idxs = self.point_idxs[node_idx][mask]

                self.point_idxs.append(self.point_idxs[node_idx][mask])
                self.is_leaf.append(True)
                self.children.append([])
                self.parents.append(node_idx)
                self.depths.append(self.depths[node_idx] + 1)
                self.children[node_idx].append(new_node_idx)
                self.impurity_reduction.append(0)
                if len(self.point_idxs[-1]) < self.min_samples_split:
                    self.verbose_print(f"\t\t Node {new_node_idx} has too few samples ({len(self.point_idxs[-1])})")
                    continue
                if impurity < self.min_impurity_decrease:
                    self.verbose_print(f"\t\t Node {new_node_idx} has too low impurity ({impurity}) to be improved")
                    continue
                if self.max_depth is not None and self.depths[node_idx] + 1 >= self.max_depth:
                    self.verbose_print(f"\t\t Node {new_node_idx} has reached max depth {self.depths[node_idx] + 1}")
                    continue

                new_tree = BranchingTree(
                    feature_dict = self.feature_dict,
                    cat_dict= self.categorical_dict,
                    max_depth = self.inner_max_depth,
                    min_samples_split = self.inner_min_samples_split,
                    min_impurity_decrease = self.inner_min_impurity_decrease,
                    min_samples_leaf = self.inner_min_samples_leaf,
                    max_leaf_nodes = self.inner_max_leaf_nodes,
                    criterion = self.criterion,
                    k = self.k,
                    outer_min_samples_leaf = self.min_samples_leaf,
                    outer_min_impurity_decrease = self.min_impurity_decrease,
                    pairwise_candidates = self.pairwise_candidates,
                    random_state = self.random_state,
                    verbose = self.verbose,
                    idx = new_node_idx,
                    pairwise_penalty=self.pairwise_penalty,
                    max_iter=self.max_iter,
                    smart_init=self.smart_init,
                    random_pairs=self.random_pairs,
                    splitter=self.internal_splitter,
                    use_dpdt=self.use_dpdt,
                    branching_penalty=self.branching_penalty,
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
                dist = np.zeros(self.n_classes)
                self.values[i] = dist
                self.n_samples[i] = 0
                self.impurity[i] = 0.0
            else:
                dist = np.bincount(y_points, minlength = self.n_classes)
                sum = np.sum(dist)
                dist = dist / sum
                self.values[i] = dist
                self.n_samples[i] = sum
                if self.criterion == 'gini':
                    self.impurity[i] = 1.0 - np.sum(dist ** 2)
                elif self.criterion == 'entropy':
                    self.impurity[i] = -np.sum(dist * np.log2(dist + 1e-10))
        del tree_heap
        gc.collect()

        if self.use_tao:
            self.verbose_print('Running TAO -----------------------------------------------------')
            self.run_tao(X, y)


    def run_tao(self, X: np.ndarray, y: np.ndarray):
        for run in range(self.n_runs):
            change = False
            self.verbose_print(f'Running TAO iter {run} -----------------------------------------------------')
            for depths in reversed(range(self.max_depth)):
                rel_nodes = np.where(np.array(self.depths) == depths)[0]
                if len(rel_nodes) == 0:
                    continue
                self.verbose_print(f'\t Running TAO on depth {depths} with {len(rel_nodes)} nodes')
                for node_idx in rel_nodes:
                    self.verbose_print(f'\t\t Running TAO on node {node_idx}, run: {run}')
                    node_idxs = self.point_idxs[node_idx]
                    if len(node_idxs) == 0:
                        continue
                    node = self.nodes[node_idx]
                    if node is None:
                        continue # this means leaf

                    # get points that are in this node
                    X_node = X[node_idxs]
                    y_node = y[node_idxs]

                    # curr_node_branching = node.predict(X_node)
                    node_children = self.children[node_idx]
                    pseudolabel_matrix = []
                    for _,child in enumerate(node_children):
                        child_pred = self.recurse_predict(X_node, child) # (n_samples,)
                        correct_mask = child_pred == y_node
                        del child_pred
                        gc.collect()
                        pseudolabel_matrix.append(correct_mask)
                    pseudolabel_matrix = np.array(pseudolabel_matrix).T * 1.0 # (n_samples, n_children)
                    pl_sum = np.sum(pseudolabel_matrix, axis=1) # (n_samples,)

                    care_mask = (pl_sum == 0) | (pl_sum == pseudolabel_matrix.shape[1]) # (n_samples,)

                    care_mask = ~care_mask # we only care about samples that are not all correct or all incorrect
                    if np.sum(care_mask) == 0:
                        self.verbose_print(f'\t\t\t Node {node_idx} has no samples to improve') 
                        continue
                    X_care = X_node[care_mask]
                    plm_care = pseudolabel_matrix[care_mask] # (n_samples, n_children)
                    X_new_care, pseudolabels, sample_weights = multi_to_single(X_care, plm_care)
                    dummy_clf = DummyClassifier(strategy='most_frequent')
                    dummy_clf.fit(X_new_care, pseudolabels)
                    dummy_branching = dummy_clf.predict(X_care)
                    dummy_accuracy = any_label_accuracy(plm_care, dummy_branching)
                    
                    curr_branching = node.predict(X_care)
                    curr_accuracy = any_label_accuracy(plm_care, curr_branching)
                    if node.final_key is not None and isinstance(node.final_key, tuple):
                        curr_accuracy -= self.tao_pair_scale * self.tao_reg
                    elif node.final_key is not None:
                        curr_accuracy -= self.tao_reg

                    feature_dict = node.feature_dict
                    cat_dict = node.cat_dict
                    single_features = {
                        k: v for k, v in feature_dict.items() if not isinstance(k, tuple)
                    }
                    best_single_feature = None
                    best_single_feature_accuracy = 0.0
                    best_single_feature_tree = None
                    for key, idxs in single_features.items():
                        clf_ = DecisionTreeClassifier(
                            max_depth=self.inner_max_depth,
                            min_samples_split=self.inner_min_samples_split,
                            # min_impurity_decrease=self.inner_min_impurity_decrease,
                            min_samples_leaf=self.inner_min_samples_leaf,
                            max_leaf_nodes=self.inner_max_leaf_nodes,
                            criterion=self.criterion,
                            random_state=self.random_state,
                        )
                        clf_.fit(X_new_care[:, idxs], pseudolabels)
                        new_branching = clf_.predict(X_care[:, idxs])
                        new_accuracy = any_label_accuracy(plm_care, new_branching)
                        new_accuracy -= self.tao_reg
                        if new_accuracy > best_single_feature_accuracy:
                            best_single_feature_accuracy = new_accuracy
                            best_single_feature = key
                            best_single_feature_tree = clf_

                    pairwise_features = {
                        k: v for k, v in feature_dict.items() if isinstance(k, tuple)
                    }
                    best_pairwise_feature = None
                    best_pairwise_feature_accuracy = 0.0
                    best_pairwise_feature_tree = None
                    for key, idxs in pairwise_features.items():
                        cat_dict_1 = cat_dict[key[0]]
                        cat_dict_2 = cat_dict[key[1]]
                        if cat_dict_1 or cat_dict_2:
                            clf_ = DecisionTreeClassifier(
                                max_depth=self.inner_max_depth,
                                min_samples_split=self.inner_min_samples_split,
                                # min_impurity_decrease=self.inner_min_impurity_decrease,
                                min_samples_leaf=self.inner_min_samples_leaf,
                                max_leaf_nodes=self.inner_max_leaf_nodes,
                                criterion=self.criterion,
                                random_state=self.random_state,
                            )
                        else:
                            clf_ = BiCARTClassifier(
                                max_depth=self.inner_max_depth,
                                min_samples_split=self.inner_min_samples_split,
                                # min_impurity_decrease=self.inner_min_impurity_decrease,
                                min_samples_leaf=self.inner_min_samples_leaf,
                                max_leaf_nodes=self.inner_max_leaf_nodes,
                                criterion=self.criterion,
                                random_state=self.random_state,
                                H = self.H,
                            )
                        # X_care_new_subset = 
                        clf_.fit(X_new_care[:, idxs], pseudolabels)
                        new_branching = clf_.predict(X_care[:, idxs])                        
                        new_accuracy = any_label_accuracy(plm_care, new_branching)
                        new_accuracy -= self.tao_pair_scale * self.tao_reg
                        if new_accuracy > best_pairwise_feature_accuracy:
                            best_pairwise_feature_accuracy = new_accuracy
                            best_pairwise_feature = key
                            best_pairwise_feature_tree = clf_
                    best_key = None
                    best_tree = None
                    node_change = False
                    if dummy_accuracy >= curr_accuracy and \
                        dummy_accuracy >= best_single_feature_accuracy and \
                        dummy_accuracy >= best_pairwise_feature_accuracy:
                        self.verbose_print(f'\t\t\t Node {node_idx} is better with dummy classifier: {dummy_accuracy}')
                        best_key  = None
                        best_tree = dummy_clf
                        node_change = True
                    elif best_single_feature_accuracy > curr_accuracy and \
                        best_single_feature_accuracy > dummy_accuracy and \
                        best_single_feature_accuracy > best_pairwise_feature_accuracy:
                        self.verbose_print(f'\t\t\t Node {node_idx} is better with single feature: {best_single_feature_accuracy}')
                        best_key = best_single_feature
                        best_tree = best_single_feature_tree
                        node_change = True
                    elif best_pairwise_feature_accuracy > curr_accuracy and \
                        best_pairwise_feature_accuracy > dummy_accuracy and \
                        best_pairwise_feature_accuracy > best_single_feature_accuracy:
                        self.verbose_print(f'\t\t\t Node {node_idx} is better with pairwise feature: {best_pairwise_feature_accuracy}')
                        best_key = best_pairwise_feature
                        best_tree = best_pairwise_feature_tree
                        node_change = True
                    
                    if node_change:
                        node.update_tree(
                            best_key, 
                            best_tree
                        )
                        change = True
                        self.recurse_predict_and_recalc(X, y, node_idx, self.point_idxs[node_idx])

            if not change:
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
            dist = np.zeros(self.n_classes)
            self.values[node_idx] = dist
            self.n_samples[node_idx] = 0
            self.impurity[node_idx] = 0.0
            return 
        else:
            dist = np.bincount(y_points, minlength = self.n_classes)
            sum = np.sum(dist)
            dist = dist / sum
            self.values[node_idx] = dist
            self.n_samples[node_idx] = sum
            if self.criterion == 'gini':
                self.impurity[node_idx] = 1.0 - np.sum(dist ** 2)
            elif self.criterion == 'entropy':
                self.impurity[node_idx] = -np.sum(dist * np.log2(dist + 1e-10))
        if len(rel_points) == 0:
            return
        node = self.nodes[node_idx]
        if node is None:
            return # this is a leaf
        node_pred = node.predict(rel_points).astype(int)
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
            return np.ones(X.shape[0]).astype(int) * np.argmax(self.values[node])
        else:
            node_children = self.children[node]
            node_pred = self.nodes[node].predict(X)
            result = np.zeros(X.shape[0]).astype(int)
            for i, child in enumerate(node_children):
                mask = node_pred == i
                result += mask * self.recurse_predict(X, child)
            return result
    
    def recurse_predict_limit(self, X, node, depth_limit):
        if self.depths[node] > depth_limit:
            raise ValueError('Depth limit must be greater than or equal to the depth of the node')
        
        if self.is_leaf[node]:
            return np.ones(X.shape[0]).astype(int) * np.argmax(self.values[node])
        elif self.depths[node] == depth_limit:
            return np.ones(X.shape[0]).astype(int) * np.argmax(self.values[node])
        else:
            node_children = self.children[node]
            node_pred = self.nodes[node].predict(X)
            result = np.zeros(X.shape[0]).astype(int)
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
        return self.label_encoder.inverse_transform(predictions)


def any_label_accuracy(y_true, y_pred):
    """
    Accuracy for multi-label classification with single-label predictions.
    A prediction is considered correct if it matches any of the true labels.

    Parameters:
    - y_true: np.ndarray of shape (n_samples, n_classes), binary indicator matrix
    - y_pred: np.ndarray of shape (n_samples,), predicted class indices

    Returns:
    - float: accuracy score
    """
    # Check for each sample whether the predicted label is among the true labels
    y_pred = y_pred.astype(int)
    correct = y_true[np.arange(len(y_true)), y_pred] == 1
    return np.mean(correct)


def multi_to_single(X, label_matrix):
    """
    Transforms a multi-label dataset into a single-label one by duplicating each
    sample for each of its labels.

    Parameters:
    - X: np.ndarray of shape (n_samples, n_features)
         Feature matrix.
    - label_matrix: np.ndarray of shape (n_samples, n_labels)
         Binary indicator matrix of labels.

    Returns:
    - X_new: np.ndarray of shape (n_new_samples, n_features)
         Duplicated feature matrix with one sample per label.
    - y_new: np.ndarray of shape (n_new_samples,)
         Corresponding single-label targets.
    """
    X_list = []
    y_list = []
    sample_weights = []
    for i in range(X.shape[0]):
        labels = np.where(label_matrix[i] == 1)[0]
        for label in labels:
            X_list.append(X[i])
            y_list.append(label)
            sample_weights.append(1.0 / len(labels))  # Weight for each label
    X_new = np.vstack(X_list)
    y_new = np.array(y_list).astype(int)
    sample_weights = np.array(sample_weights)
    return X_new, y_new, sample_weights

