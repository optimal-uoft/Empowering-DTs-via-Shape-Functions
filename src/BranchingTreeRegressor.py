from sklearn.tree import DecisionTreeRegressor
import numpy as np
import itertools
import queue
import warnings
from sklearn.cluster import KMeans
from src.BiTAO import BiCARTClassifier, BiCARTRegressor
from line_profiler import profile
from argparse import Namespace
class KMeansBranch:
    def __init__(self, n_clusters=8, random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.tree_ = Namespace()
    def fit(self, X, y):
        """
        Fit the KMeans model to the data.
        """
        X = np.asarray(X)
        self.kmeans.fit(X)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.labels_ = self.kmeans.labels_
        
        """
        Store the target values for each cluster.
        """
        self.tree_.children_left = np.array([1] + [-1] * self.n_clusters)
        impurity = np.zeros(self.n_clusters + 1)
        impurity[0] = np.var(y)
        value = np.zeros((self.n_clusters + 1, 1))
        value[0, 0] = np.mean(y)
        for i in range(self.n_clusters):
            mask = self.labels_ == i
            if np.any(mask):
                impurity[i + 1] = np.var(y[mask])
                value[i + 1, 0] = np.mean(y[mask])
            else:
                impurity[i + 1] = 0.0
        self.tree_.impurity = impurity
        self.tree_.n_node_samples = np.array([len(y)] + [np.sum(self.labels_ == i) for i in range(self.n_clusters)])
        self.tree_.value = value
        return self
    def apply(self, X):
        """
        Apply the KMeans clustering to the data and return the cluster labels.
        """
        if not hasattr(self, 'cluster_centers_'):
            raise ValueError("KMeans model has not been fitted yet.")
        X = np.asarray(X)
        return self.kmeans.predict(X) + 1
    


class BranchingTreeRegressor: 
    def __init__(self,
                 feature_dict,
                 cat_dict,
                 max_depth,
                 min_samples_split,
                 min_impurity_decrease,
                 min_samples_leaf,
                 max_leaf_nodes,
                 criterion,
                 k,
                 outer_min_samples_leaf,
                 outer_min_impurity_decrease,
                 pairwise_candidates,
                 pairwise_penalty,
                 idx,
                 random_state = 42,
                 verbose = False,
                 use_kmeans = False,
                 H = 7,
                 max_iter = 10,
                 smart_init = True,
                 random_pairs = False,
                 splitter = 'best',
                 branching_penalty = 0.0,
                 max_features = None
                 ):
        self.feature_dict = feature_dict.copy()
        self.cat_dict = cat_dict.copy()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.use_kmeans = use_kmeans
        if self.use_kmeans and self.max_leaf_nodes is None and self.max_depth is not None:
            self.max_leaf_nodes = 2 ** self.max_depth
        elif self.use_kmeans and self.max_leaf_nodes is None:
            self.max_leaf_nodes = 32

        self.k = k
        self.outer_min_samples_leaf = outer_min_samples_leaf
        self.outer_min_impurity_decrease = outer_min_impurity_decrease
        self.pairwise_candidates = pairwise_candidates
        self.pairwise_penalty = pairwise_penalty
        self.idx = idx
        self.random_state = random_state
        self.verbose = verbose
        self.initial_impurity = None
        self.failed = 0
        self.solutions = {k_: {} for k_ in range(2, self.k+1)}
        self.trees = {}
        self.best_impurity_decrease = {k_: (-np.inf, None) for k_ in range(2, self.k+1)}
        self.best_score_dict = {k_: (-np.inf, None) for k_ in range(2, self.k+1)}
        self.best_score = -np.inf
        self.best_impurity = -np.inf
        self.best_k = None
        self.H = H
        self.max_iter = max_iter
        self.smart_init = smart_init
        self.final_tree = None
        self.mapping = None
        self.final_key = None
        self.viz_store = {}
        self.random_pairs = random_pairs
        self.criterion = criterion
        self.splitter = splitter
        self.branching_penalty = branching_penalty
        self.tao_pred = False
        self.max_features = max_features

        # self.feature_dict = feature_dict.copy()
        if self.max_features is None:
            self.max_features = len(self.feature_dict)
        else:
            feature_dict_keys = list(self.feature_dict.keys())
            if isinstance(self.max_features, int):
                if self.max_features <= 0:
                    raise ValueError("max_features must be greater than 0 if specified as an integer.")
                if self.max_features > len(self.feature_dict):
                    raise ValueError("max_features cannot be greater than the number of features.")
                features_to_select = np.random.choice(feature_dict_keys, self.max_features, replace=False)
            elif isinstance(self.max_features, float):
                if not (0 < self.max_features <= 1):
                    raise ValueError("max_features must be between 0 and 1 if specified as a float.")
                num_features = int(np.ceil(len(self.feature_dict) * self.max_features))
                features_to_select = np.random.choice(feature_dict_keys, num_features, replace=False)
            elif isinstance(self.max_features, str):
                if self.max_features == 'sqrt':
                    num_features = int(np.ceil(np.sqrt(len(self.feature_dict))))
                    features_to_select = np.random.choice(feature_dict_keys, num_features, replace=False)
                elif self.max_features == 'log2':
                    num_features = int(np.ceil(np.log2(len(self.feature_dict))))
                    features_to_select = np.random.choice(feature_dict_keys, num_features, replace=False)
                else:
                    raise ValueError(f"Unknown string value for max_features: {self.max_features}. Use 'sqrt', 'log2', or a float/int.")
            else:
                raise ValueError(f"max_features must be an int, float, or str, got {type(self.max_features)}.")
            self.feature_dict = {key: self.feature_dict[key] for key in features_to_select}
            
    def __lt__(self, other):

        return self.idx < other.idx
    def verbose_print(self, msg):
        if self.verbose:
            print(msg)
    
    def get_chosen_feature(self):
        return self.final_key
    
    @profile
    def fit(self, X, y):
        self.verbose_print(f'Fitting BranchingTreeRegressor with {len(self.feature_dict)} features')
        total_points = X.shape[0]
        if total_points < self.min_samples_split:
            self.failed = True
            self.verbose_print(f'\t \t Not enough samples to split: {total_points} < {self.min_samples_split}')
            return None
        var_y = np.var(y)
        
        if var_y < self.outer_min_impurity_decrease or var_y < 1e-8:
            self.failed = True
            self.verbose_print(f'\t \t Not enough variance in y: {var_y}')
            return None

        self.failed=  True
        solutions = {}
        trees = {}
        for key, values in self.feature_dict.items():
            self.verbose_print(f'\t Fitting tree for feature {key}')
            # check if the feature is categorical
            if len(values) == 1:
                rel_x = X[:, values[0]].reshape(-1, 1)
            else:
                rel_x = X[:, values]
            
            k_solutions, tree = self.construct_mapping(rel_x, y)
            if k_solutions is None:
                self.verbose_print(f'\t \t No solutions found for feature {key}')
                continue
            else:
                self.verbose_print(f'\t \t Found {len(k_solutions)} solutions for feature {key}')
            self.failed = False
            trees[key] = tree # save the tree
            best_k_key = np.inf
            best_score_key = np.inf
            for k_, value in k_solutions.items(): # save the solutions for each k-way branching
                # update the best impurity decrease for each k-way branching
                self.verbose_print(f'\t \t Checking k={k_} for feature {key} with score {value["score"]}')
                if value['score'] < best_score_key:
                    best_score_key = value['score']
                    best_k_key = k_
                    self.verbose_print(f'\t \t Found a better solution for feature {key} with k={k_} and score {value["score"]}')
                best_sol = k_solutions[best_k_key]
                solutions[key] = best_sol
        if self.failed:
            return None
        
        if self.pairwise_candidates > 0:
            keys = list(solutions.keys())
            pairwise_combos = list(itertools.combinations(keys, 2))
            best_features = queue.PriorityQueue()
            for key1, key2 in pairwise_combos:
                if self.random_pairs:
                    rng = np.random.default_rng(self.random_state)
                    score = rng.random()
                else:
                    branching1 = solutions[key1]['branching']
                    branching2 = solutions[key2]['branching']
                    impurity1 = solutions[key1]['impurity']
                    impurity2 = solutions[key2]['impurity']
                    score = self.score_pairwise(y, branching1, branching2, min(impurity1, impurity2))
                best_features.put((score, (key1, key2)))
            for keys in solutions.keys():
                del solutions[keys]['branching']
            
            for _ in range (self.pairwise_candidates):
                # check if the queue is empty
                if best_features.empty():
                    break
                key1, key2 = best_features.get()[1]
                idx_1 = self.feature_dict[key1]
                idx_2 = self.feature_dict[key2]
                new_key = (key1, key2)
                new_idxs = idx_1 + idx_2
                self.feature_dict[new_key] = new_idxs
                self.verbose_print(f'\t Fitting tree for feature {new_key}')
                cat_1 = self.cat_dict[key1]
                cat_2 = self.cat_dict[key2]
                # if either is categorical, don't use bicart
                if cat_1 or cat_2:
                    use_bicart = False
                else:
                    use_bicart = True
                k_solutions, tree = self.construct_mapping(X[:,new_idxs], y, bicart=use_bicart)
                if k_solutions is None:
                    self.verbose_print(f'\t \t No solutions found for feature {new_key}')
                    continue
                trees[new_key] = tree
                best_k_key = np.inf
                best_score_key = np.inf
                for k_, value in k_solutions.items():
                    # update the best impurity decrease for each k-way branching
                    if value['score'] < best_score_key:
                        best_score_key = value['score']
                        best_k_key = k_
                best_sol = k_solutions[best_k_key]
                solutions[new_key] = best_sol
        
        sorted_keys = sorted(solutions, key=lambda key_: solutions[key_]["score"])
        failed = True
        for key in sorted_keys:
            solution = solutions[key]
            partition_weights = solution['partition_weights']
            impurity_decrease = solution['impurity_decrease']
            # print(partition_weights)
            if np.min(partition_weights) < self.outer_min_samples_leaf:
                self.verbose_print(f'\t \t Skipping feature {key} due to partition weights')
                continue
            if impurity_decrease < self.outer_min_impurity_decrease:
                self.verbose_print(f'\t \t Skipping feature {key} due to impurity decrease')
                continue
            # passed the conditions, so we can commit and break
            failed = False
            self.verbose_print(f'\t \t Committing feature {key} with score {solution["score"]}')
            self.best_k = solution['k_value']
            self.best_impurity_decrease = solution['impurity_decrease']
            self.final_key = key
            self.final_tree = trees[key]
            self.mapping = solution['mapping']
            self.final_solution = solution
            break
        # clear the fit trees
        # del solutions
        
        self.failed = failed
        self.trees = trees # store the trees for later use

    def score_pairwise(self, y, branching1, branching2, min_impurity):

        # branching1: n_samples, This is a numpy array of shape (n_points,)
        # branching2: n_samples, This is a numpy array (n_points,)
        # imp1: scalar of the impurity of branching 1
        # imp2: scalar of the impurity of branching 2
        # y: n_samples, This is a numpy array of the true labels
        # one_hot = np.eye(len(np.unique(y)))[y]
        # cartesion product of branching1 and branching2
        n_samples = branching1.shape[0]
        k_2 = len(np.unique(branching2))
        joint_branching = branching1 * k_2 + branching2
        # raise ValueError('joint_branching is not a 1D array')
        partitions = np.unique(joint_branching)

        total_impurity = 0
        for partition in partitions:
            mask = joint_branching == partition
            rel_points = y[mask]
            avg = np.mean(rel_points)
            rel_points = rel_points - avg
            rel_points = rel_points**2

            total_impurity += np.sum(rel_points)
        total_imp = total_impurity / n_samples
        impurity_reduction = -1 * (min_impurity - total_imp)
        return impurity_reduction

    @profile
    def predict(self, X):
        if self.tao_pred:
            if self.final_key is None:
                return self.final_tree.predict(X)
            return self.final_tree.predict(X[:, self.feature_dict[self.final_key]])
        leaf_nodes = self.final_tree.apply(X[:, self.feature_dict[self.final_key]])
        predictions = self.mapping[leaf_nodes]
        return predictions.astype(int)
    def update_tree(self, new_key, new_tree):
        self.tao_pred = True
        self.final_key = new_key
        self.final_tree = new_tree
    
    def construct_mapping(self, X, y, bicart=False):
        if self.use_kmeans: 
            # Compute number of unique points
            unique_X = np.unique(X, axis=0)
            n_unique = len(unique_X)

            # Set max_leaf_nodes conservatively based on both dataset size and number of unique rows
            if self.max_leaf_nodes is None:
                max_leaf_nodes = min(len(X), n_unique, 16)
            else:
                max_leaf_nodes = min(len(X), n_unique, self.max_leaf_nodes)

            tree = KMeansBranch(
                n_clusters=int(max_leaf_nodes)
            )
        elif bicart:
            tree = BiCARTRegressor(
                max_depth = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_impurity_decrease = self.min_impurity_decrease,
                min_samples_leaf = self.min_samples_leaf,
                max_leaf_nodes = self.max_leaf_nodes,
                random_state = self.random_state,
                H=self.H
            )
        else:
            tree = DecisionTreeRegressor(
                max_depth = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_impurity_decrease = self.min_impurity_decrease,
                min_samples_leaf = self.min_samples_leaf,
                max_leaf_nodes = self.max_leaf_nodes,
                random_state = self.random_state,
            )
        # 

        tree.fit(X,y)
        self.initial_impurity = tree.tree_.impurity[0]

        leaf_nodes = np.where(tree.tree_.children_left == -1)[0]
        if len(leaf_nodes) == 1:
            return None, None
        leaf_values = tree.tree_.value.squeeze(axis=1)[leaf_nodes]
        leaf_samples = tree.tree_.n_node_samples[leaf_nodes]
        leaf_error = tree.tree_.impurity[leaf_nodes]
        solutions = {}

        if self.pairwise_candidates > 0:
            apply_ = tree.apply(X)

        for k_value in reversed(range(2, self.k + 1)):
            if k_value > len(leaf_nodes):
                # if k_value is greater than the number of leaf nodes, we cannot split further
                continue
            if k_value == len(leaf_nodes):
                # weighted_distributions = leaf_values * leaf_samples[:, np.newaxis]
                total_impurity = np.sum(leaf_error * leaf_samples) / np.sum(leaf_samples)
                impurity_decrease = self.initial_impurity - total_impurity
                if impurity_decrease <= self.outer_min_impurity_decrease:
                    return None, None
                if np.min(leaf_samples) < self.outer_min_samples_leaf:
                    return None, None
                max_leaf = max(leaf_nodes)
                mapping = np.zeros(max_leaf+1, dtype=np.int32)
                for i, leaf in enumerate(leaf_nodes):
                    mapping[leaf] = i
                # mapping = {leaf_nodes[i]: i for i in range(self.k)}
                if len(leaf_values.shape) == 2:
                    leaf_values = leaf_values.squeeze(axis=1)
                return_dict= {
                    'value': {i: v for i, v in enumerate(leaf_values)},
                    'error': {i: v for i, v in enumerate(leaf_error)},
                    'partition_weights': leaf_samples,
                    'impurity_decrease': impurity_decrease,
                    'impurity': total_impurity,
                    'mapping': mapping,
                    'k_value': k_value,
                }
                return_dict['score'] = total_impurity + self.branching_penalty * (k_value - 1)
                if self.pairwise_candidates > 0:
                    apply_ = tree.apply(X)
                    return_dict['branching'] = mapping[apply_]
                solutions[k_value] = return_dict
                continue
            curr_solution = self.coordinate_descent(
                k_ = k_value,
                leaf_values = leaf_values,
                leaf_samples = leaf_samples,
                leaf_nodes = leaf_nodes,
                leaf_error = leaf_error,
            )
            if curr_solution is None:
                continue
            # check for conditions
            mapping = curr_solution['mapping']
            if self.pairwise_candidates > 0:
                branching = mapping[apply_]
                # branching = np.array([mapping[idx] for idx in apply_])
                curr_solution['branching'] = branching
            curr_solution['score'] = curr_solution['impurity'] + self.branching_penalty * (k_value - 1)
            curr_solution['k_value'] = k_value
            solutions[k_value] = curr_solution
        if len(solutions) == 0:
            return None, None
        else:
            return solutions, tree
    
    @profile
    def coordinate_descent(self, k_, 
                           leaf_values, 
                           leaf_samples, 
                           leaf_nodes, 
                           leaf_error):
                # k_ = k_value,
                # leaf_values = leaf_values,
                # leaf_samples = leaf_samples,
                # leaf_nodes = leaf_nodes,
                # leaf_error = leaf_error,
        # run the kmeans algorithm but catch convergence warnings and raise an exception
        if self.smart_init:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    if len(leaf_values.shape) == 1:
                        leaf_values = leaf_values.reshape(-1, 1)
                    kmeans = KMeans(n_clusters=k_, n_init=1, copy_x=False)
                    kmeans.fit(leaf_values, sample_weight=leaf_samples)
                except Warning:
                    # this means that the # of distinct points is less than k_
                    return None
            assignments = kmeans.labels_
        else:
            assignments = np.random.randint(0, k_, size=len(leaf_nodes))

        if len(leaf_values.shape) == 2:
            leaf_values = leaf_values.squeeze(axis=1)
        
        # leaf_values: (n_leaves,) - mean value of each leaf
        # leaf_samples: (n_leaves,) - number of samples in each leaf
        # leaf_nodes: (n_leaves,) - index of each leaf
        # leaf_error: (n_leaves,) - impurity/squared error of each leaf

        partition_samples, partition_mean, partition_error = compute_partition_stats(
            assignments = assignments,
            leaf_values = leaf_values,
            leaf_error = leaf_error,
            leaf_samples = leaf_samples,
            k_ = k_,
        )

        rng = np.random.default_rng(self.random_state)
        n_leaf_nodes = len(leaf_nodes)
        best_partition_samples = partition_samples
        best_partition_mean = partition_mean
        best_partition_error = partition_error
        best_partition_weighted_error = np.sum(partition_error * partition_samples) / np.sum(partition_samples)
        best_assignment = assignments.copy()
        counter = 0
        for _ in range(self.max_iter):
            leaves = rng.permutation(n_leaf_nodes)
            switched = False
            for leaf in leaves:
                old_assignments = assignments.copy()
                curr_assignment = assignments[leaf]
                for k_val in range(k_):
                    if k_val == curr_assignment:
                        continue
                    assignments[leaf] = k_val
                    partition_samples, partition_mean, partition_error = compute_partition_stats(
                        assignments = assignments,
                        leaf_values = leaf_values,
                        leaf_error = leaf_error,
                        leaf_samples = leaf_samples,
                        k_ = k_,
                    )
                    partition_weighted_error = np.sum(partition_error * partition_samples) / np.sum(partition_samples)
                    if partition_weighted_error < best_partition_weighted_error:
                        best_partition_samples = partition_samples
                        best_partition_mean = partition_mean
                        best_partition_error = partition_error
                        best_partition_weighted_error = partition_weighted_error
                        best_assignment = assignments.copy()
                        switched = True
                    else:
                        assignments = old_assignments.copy() #if no improvement, revert the assignment
            if not switched:
                counter += 1
                if counter > 5:
                    break
        mapping = np.zeros(np.max(leaf_nodes) + 1, dtype=np.int32)
        for i, leaf in enumerate(leaf_nodes):
            mapping[leaf] = best_assignment[i]
        solution = {
            'value': best_partition_mean,
            'error': best_partition_error,
            'partition_weights': best_partition_samples,
            'impurity_decrease': self.initial_impurity - best_partition_weighted_error,
            'impurity': best_partition_weighted_error,
            'mapping': mapping,
        }
        return solution



def compute_partition_stats(assignments, leaf_values, leaf_error, leaf_samples, k_):
    """
    Compute, for each of k_ clusters:
      - partition_samples[i]: total number of samples in cluster i
      - partition_mean[i]: weighted mean (centroid) of cluster i
      - partition_error[i]: 1/N_i * sum_s [ (var_s + ||c_s - mu_i||^2) * n_s ]
    
    Parameters
    ----------
    assignments : array_like of shape (L,)
        Cluster index (0..k_-1) for each leaf s=0..L-1.
    leaf_values : array_like of shape (L,)
        Centroid (mean) of each leaf (c_s).
    leaf_error : array_like of shape (L, )
        Error of each leaf (variance-like term var_s).
    leaf_samples : array_like of shape (L,)
        Number of samples in each leaf (n_s).
    k_ : int
        Number of clusters.
    
    Returns
    -------
    partition_samples : numpy.ndarray of shape (k_,)
        Total samples per cluster (N_i).
    partition_mean : numpy.ndarray of shape (k_,)
        Weighted mean per cluster (mu_i).
    partition_error : numpy.ndarray of shape (k_,)
        Average shifted variance per cluster.
    """
    
    # Ensure inputs are numpy arrays for consistent behavior and vectorized operations
    assignments = np.asarray(assignments)
    leaf_values = np.asarray(leaf_values)
    leaf_error = np.asarray(leaf_error)
    leaf_samples = np.asarray(leaf_samples)

    L = assignments.shape[0]

    # Handle L=0 (no leaves) case explicitly
    if L == 0:
        partition_samples = np.zeros(k_, dtype=float)
        partition_mean = np.full(k_, np.nan, dtype=float)
        partition_error = np.full(k_, np.nan, dtype=float)
        return partition_samples, partition_mean, partition_error

    # Validate that assignments are within the expected range [0, k_-1].
    # This assumption is critical for np.bincount with minlength=k_ 
    # to produce arrays of exactly length k_.
    if k_ > 0 and (np.any(assignments < 0) or np.any(assignments >= k_)):
        raise ValueError(
            f"Elements in 'assignments' must be between 0 and k_-1 (k_={k_}). "
            f"Found min assignment: {np.min(assignments)}, max assignment: {np.max(assignments)}."
        )
    elif k_ == 0 and L > 0 : # No clusters but leaves exist
         raise ValueError("k_ is 0, but there are leaves to assign.")
    elif k_ == 0 and L == 0: # Already handled by L==0 check, but good to be thorough
        return np.array([]), np.array([]), np.array([])


    # 1. Compute partition_samples[i]: total number of samples in cluster i (N_i)
    # N_i = sum_{s in cluster i} n_s
    # .astype(float) ensures results are float, e.g. if leaf_samples are integers.
    partition_samples = np.bincount(assignments, weights=leaf_samples, minlength=k_).astype(float)

    # 2. Compute partition_mean[i]: weighted mean (centroid) of cluster i (mu_i)
    # mu_i = (sum_{s in cluster i} c_s * n_s) / N_i
    # Numerator: sum_{s in cluster i} (leaf_values_s * leaf_samples_s)
    weighted_value_sum_per_cluster = np.bincount(assignments, 
                                                  weights=leaf_values * leaf_samples, 
                                                  minlength=k_).astype(float)
    
    # Initialize partition_mean with NaNs for potentially empty clusters
    partition_mean = np.full(k_, np.nan, dtype=float)
    
    # Mask for non-empty clusters (where partition_samples[i] > 0)
    non_empty_clusters_mask = partition_samples > 0
    
    # Calculate mean only for non-empty clusters to avoid division by zero
    # np.divide can also be used with a 'where' clause.
    if np.any(non_empty_clusters_mask): # Proceed only if there's at least one non-empty cluster
        partition_mean[non_empty_clusters_mask] = \
            weighted_value_sum_per_cluster[non_empty_clusters_mask] / partition_samples[non_empty_clusters_mask]

    # 3. Compute partition_error[i]: 1/N_i * sum_s [ (var_s + ||c_s - mu_i||^2) * n_s ]
    #    where s are leaves in cluster i.
    
    # Retrieve the mean (mu_i) for each leaf's assigned cluster.
    # For leaves assigned to cluster i, this is partition_mean[i].
    # If cluster i is empty, partition_mean[i] is NaN.
    # So, mu_for_each_leaf[s] will be NaN if leaf s is in an empty cluster.
    mu_for_each_leaf = partition_mean[assignments] 
    
    # Calculate squared difference: ||c_s - mu_i||^2. For scalar c_s, this is (c_s - mu_i)^2.
    # If mu_for_each_leaf[s] is NaN, then squared_diff_from_cluster_mean[s] will also be NaN.
    squared_diff_from_cluster_mean = (leaf_values - mu_for_each_leaf)**2
    
    # Term inside the sum for each leaf s: (var_s + ||c_s - mu_i||^2) * n_s
    # var_s is leaf_error[s].
    # If squared_diff_from_cluster_mean[s] is NaN, this entire term for leaf s becomes NaN.
    term_to_sum_for_error_per_leaf = (leaf_error + squared_diff_from_cluster_mean) * leaf_samples
    
    # Sum these terms per cluster using bincount.
    # If term_to_sum_for_error_per_leaf[s] is NaN (because leaf s is in an empty cluster),
    # and leaf s is assigned to that empty cluster, np.bincount for that cluster might result in NaN.
    # This is acceptable, as the error for an empty cluster should be NaN.
    # A NaN term from a leaf in an *empty* cluster will not corrupt the sum for a *non-empty* cluster,
    # as bincount sums contributions by `assignments` index.
    # If `assignments[s]` corresponds to a non-empty cluster, then `mu_for_each_leaf[s]` is not NaN,
    # thus `term_to_sum_for_error_per_leaf[s]` will not be NaN.
    sum_of_error_terms_per_cluster = np.bincount(assignments, 
                                                  weights=term_to_sum_for_error_per_leaf, 
                                                  minlength=k_).astype(float)

    # Initialize partition_error with NaNs
    partition_error = np.full(k_, np.nan, dtype=float)
    
    # Calculate error only for non-empty clusters
    if np.any(non_empty_clusters_mask):
        partition_error[non_empty_clusters_mask] = \
            sum_of_error_terms_per_cluster[non_empty_clusters_mask] / partition_samples[non_empty_clusters_mask]
            
    return partition_samples, partition_mean, partition_error

