from sklearn.tree import DecisionTreeClassifier
import numpy as np
import itertools
import queue
import warnings
from sklearn.cluster import KMeans
from src.BiTAO import BiCARTClassifier
from argparse import Namespace
from src.dpdt_clf import DPDTreeClassifierApply
from sklearn.preprocessing import LabelEncoder

def get_leaf_subtree_sides(clf: DecisionTreeClassifier):
    tree = clf.tree_
    children_left = tree.children_left
    children_right = tree.children_right

    # Step 1: Find leaf nodes (nodes with no children)
    is_leaf = (children_left == -1) & (children_right == -1)
    leaf_indices = np.where(is_leaf)[0]

    # Step 2: Traverse the tree and record side of root for each node
    node_side = {}  # node_id -> 0 (left of root) or 1 (right of root)

    def assign_side(node_id, side):
        node_side[node_id] = side
        if children_left[node_id] != -1:
            assign_side(children_left[node_id], side)
        if children_right[node_id] != -1:
            assign_side(children_right[node_id], side)

    # Start from root's left and right children
    assign_side(children_left[0], 0)
    assign_side(children_right[0], 1)

    # Step 3: Create output vector
    leaf_sides = np.array([node_side[leaf_id] for leaf_id in leaf_indices])

    return leaf_sides

class DPDTBranch:
    def __init__(self, max_depth=3, 
                 min_samples_leaf=1, 
                 random_state=None, 
                 min_impurity_decrease=0.0,
                 min_samples_split=2,
                 criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.criterion = criterion
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = min_samples_split
        self.dpdt = DPDTreeClassifierApply(max_depth=max_depth, 
                                            min_samples_leaf=min_samples_leaf, 
                                            min_samples_split=min_samples_split,
                                            min_impurity_decrease=min_impurity_decrease,
                                            random_state=random_state)
        self.tree_ = Namespace()
        self.label_encoder = None

    def fit(self, X, y):
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        X = np.asarray(X)
        self.dpdt.fit(X, y)
        self.labels_ = self.dpdt.apply(X)

        self.n_leaves = np.max(self.labels_) + 1
        self.tree_.children_left = np.array([1] + [-1] * self.n_leaves)

        y_onehot = np.eye(self.n_classes)[y]
        overall_distribution = np.sum(y_onehot, axis=0)
        overall_criterion = self.gini(overall_distribution) if self.criterion == 'gini' else self.entropy(overall_distribution)

        impurity = np.zeros(self.n_leaves + 1)
        value = np.zeros((self.n_leaves + 1, 1, self.n_classes))
        n_node_samples = np.zeros(self.n_leaves + 1, dtype=np.int32)

        impurity[0] = overall_criterion
        value[0, 0, :] = overall_distribution / np.sum(overall_distribution)
        n_node_samples[0] = len(y)

        for i in range(self.n_leaves):
            mask = self.labels_ == i
            rel_y = y_onehot[mask]
            if rel_y.shape[0] > 0:
                distribution = np.sum(rel_y, axis=0)
                normalized = distribution / np.sum(distribution)
            else:
                normalized = np.zeros(self.n_classes)
            impurity[i + 1] = self.gini(normalized) if self.criterion == 'gini' else self.entropy(normalized)
            value[i + 1, 0, :] = normalized
            n_node_samples[i + 1] = np.sum(mask)

        self.tree_.n_node_samples = n_node_samples
        self.tree_.value = value
        self.tree_.impurity = impurity
        return self

    @staticmethod
    def gini(p):
        p = p / np.sum(p)
        return 1.0 - np.sum(p ** 2)

    @staticmethod
    def entropy(p):
        p = p / np.sum(p)
        return -np.sum(p * np.log2(p + 1e-9))
    
    def apply(self, X):
        return self.dpdt.apply(X) + 1 # to shift the labels to start from 1 instead of 0


class BranchingTree: 
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
                 H = 7,
                 max_iter = 10,
                 smart_init = True,
                 random_pairs = False,
                 splitter = 'best',
                 branching_penalty = 0.0,
                 use_dpdt = False
                 ):
        self.feature_dict = feature_dict.copy()
        self.cat_dict = cat_dict.copy()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.criterion = criterion
        self.k = k
        self.outer_min_samples_leaf = outer_min_samples_leaf
        self.outer_min_impurity_decrease = outer_min_impurity_decrease
        self.pairwise_candidates = pairwise_candidates
        self.pairwise_penalty = pairwise_penalty
        self.branching_penalty = branching_penalty
        self.idx = idx
        self.random_state = random_state
        self.verbose = verbose
        self.initial_impurity = None
        self.failed = 0
        self.best_score = -np.inf
        self.best_impurity = -np.inf
        self.best_k = None
        self.H = H
        self.max_iter = max_iter
        self.smart_init = smart_init
        self.final_tree = None
        self.mapping = None
        self.tao_pred =  False
        self.final_key = None
        self.viz_store = {}
        self.random_pairs = random_pairs
        self.splitter = splitter
        self.use_dpdt = use_dpdt
        self.criterion_flag = 0 if self.criterion == 'gini' else 1
    def __lt__(self, other):

        return self.idx < other.idx
    def verbose_print(self, msg):
        if self.verbose:
            print(msg)
    
    def get_chosen_feature(self):
        return self.final_key
    
    def fit(self, X, y):
        total_points = X.shape[0]
        if total_points < self.min_samples_split:
            self.failed = True
            return None
        n_classes = len(np.unique(y))
        if n_classes == 1:
            self.failed = True
            return None
        self.failed=  True
        solutions = {}
        trees = {}
        for key, values in self.feature_dict.items():
            # check if the feature is categorical
            if len(values) == 1:
                rel_x = X[:, values[0]].reshape(-1, 1)
            else:
                rel_x = X[:, values]
            k_solutions, tree = self.construct_mapping(rel_x, y)
            if k_solutions is None:
                self.verbose_print(f'\t \t No solutions found for feature {key}')
                continue
            self.failed = False
            trees[key] = tree # save the tree
            best_k_key = np.inf
            best_score_key = np.inf
            for k_, value in k_solutions.items():
                if value['score'] < best_score_key:
                    best_k_key = k_
                    best_score_key = value['score']

            best_sol = k_solutions[best_k_key]
            solutions[key] = best_sol # save the solutions for each feature
       
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
                del solutions[keys]['branching']  # remove branching from the solutions
                
            for _ in range(self.pairwise_candidates):
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
                    self.verbose_print(f'\t \t No solutions found for feature {key}')
                    continue
                trees[new_key] = tree  # save the tree
                best_k_key = np.inf
                best_score_key = np.inf
                for k_, value in k_solutions.items():
                    if value['score'] < best_score_key:
                        best_k_key = k_
                        best_score_key = value['score']

                best_sol = k_solutions[best_k_key]
                best_sol['score'] += self.pairwise_penalty
                del best_sol['branching']  # remove branching from the solutions
                solutions[new_key] = best_sol  # save the solutions for each feature
        # now we have all the solutions, we need to find the one with the lowest score that satisfies the conditions

        sorted_keys = sorted(solutions, key=lambda key_: solutions[key_]["score"])
        failed = True
        for key in sorted_keys:
            solution = solutions[key]
            partition_weights = solution['partition_weights']
            impurity_decrease = solution['impurity_decrease']
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
        del solutions
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
        partitions = np.unique(joint_branching)
        total_impurity = 0
        for partition in partitions:
            mask = joint_branching == partition
            rel_points = y[mask]
            bin_count = np.bincount(rel_points).astype(np.float64)
            if len(bin_count) == 0:
                continue
            imp = self.calculate_impurity(bin_count, weighted = True)
            total_impurity += imp * np.sum(mask)
        total_imp = total_impurity / n_samples # this is better if low
        impurity_reduction =-1 * (min_impurity - total_imp) # this is better if high, so we negate
    
        return impurity_reduction 

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
        if bicart:
            tree = BiCARTClassifier(
                max_depth = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_impurity_decrease = self.min_impurity_decrease,
                min_samples_leaf = self.min_samples_leaf,
                max_leaf_nodes = self.max_leaf_nodes,
                criterion = self.criterion,
                random_state = self.random_state,
                H=self.H,
                splitter = self.splitter,
            )
        else:
            if self.use_dpdt:
                tree = DPDTBranch(
                    max_depth = self.max_depth,
                    min_samples_leaf= self.min_samples_leaf,
                    criterion= self.criterion,
                    min_impurity_decrease = self.min_impurity_decrease,
                    min_samples_split = self.min_samples_split,
                )
            else:
                tree = DecisionTreeClassifier(
                    max_depth = self.max_depth,
                    min_samples_split = self.min_samples_split,
                    min_impurity_decrease = self.min_impurity_decrease,
                    min_samples_leaf = self.min_samples_leaf,
                    max_leaf_nodes = self.max_leaf_nodes,
                    criterion = self.criterion,
                    random_state = self.random_state,
                    splitter = self.splitter,
                )
            
        tree.fit(X,y)
        self.initial_impurity = tree.tree_.impurity[0]

        leaf_nodes = np.where(tree.tree_.children_left == -1)[0]

        if len(leaf_nodes) == 1:
            self.verbose_print(f'\t \t Only one leaf node found')
            return None, None
        if len(leaf_nodes) < self.k:
            k_ = len(leaf_nodes)
        else:
            k_ = self.k
        all_node_distributions = tree.tree_.value.squeeze(axis=1)
        all_node_samples = tree.tree_.n_node_samples
        leaf_distributions = all_node_distributions[leaf_nodes]
        leaf_samples = all_node_samples[leaf_nodes]
        solutions = {}

        if self.pairwise_candidates > 0:
            apply_ = tree.apply(X)

        for k_value in reversed(range(2, k_ + 1)):
            if k_value > len(leaf_nodes):
                # if k_value is greater than the number of leaf nodes, we cannot split further
                self.verbose_print(f'\t \t Skipping k_value {k_value} as it is greater than the number of leaf nodes {len(leaf_nodes)}')
                continue
            if k_value == len(leaf_nodes): # if the number of leaf nodes is equal to k, we can just return the solution
                weighted_distributions = leaf_distributions * leaf_samples[:, np.newaxis]
                leaf_impurity = tree.tree_.impurity[leaf_nodes]
                partition_weights = [leaf_samples[i] for i in range(k_value)]
                total_impurity = np.sum(leaf_impurity * leaf_samples) / np.sum(leaf_samples)
                impurity_decrease = self.initial_impurity - total_impurity

                max_leaf = max(leaf_nodes)
                mapping = np.zeros(max_leaf+1, dtype=np.int32)
                for i, leaf in enumerate(leaf_nodes):
                    mapping[leaf] = i
                return_dict= {
                    'weighted_distributions': {i: v for i, v in enumerate(weighted_distributions)},
                    'impurities': {i: v for i, v in enumerate(leaf_impurity)},
                    'partition_weights': partition_weights,
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
            if k_value == 2 and not self.use_dpdt: 
                # get init solution from tree
                leaf_sides = get_leaf_subtree_sides(tree)
            else:
                leaf_sides = None
            curr_solution = self.coordinate_descent(
                k_ = k_value,
                leaf_distributions = leaf_distributions,
                leaf_samples = leaf_samples,
                leaf_nodes = leaf_nodes,
                leaf_sides = leaf_sides,
            )
            if curr_solution is None:
                continue
            # check for conditions
            mapping_ = curr_solution['mapping']
            if self.pairwise_candidates > 0:
                branching = mapping_[apply_]
                # branching = np.array([mapping_[idx] for idx in apply_])
                curr_solution['branching'] = branching
            curr_solution['score'] = curr_solution['impurity'] + self.branching_penalty * (k_value - 1)
            curr_solution['k_value'] = k_value
            solutions[k_value] = curr_solution
        if len(solutions) == 0:
            return None, None
        else:
            return solutions, tree
    
    def coordinate_descent(self, k_, leaf_distributions, leaf_samples, leaf_nodes, leaf_sides=None):

        # run the kmeans algorithm but catch convergence warnings and raise an exception
        if self.smart_init:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    kmeans = KMeans(n_clusters=k_, copy_x=False)
                    kmeans.fit(leaf_distributions, sample_weight=leaf_samples)
                    assignments = kmeans.labels_
                except Warning:
                    # this means that the # of distinct points is less than k_, we can fallback to random init
                    assignments = np.random.randint(0, k_, size=len(leaf_nodes))
        else:
            assignments = np.random.randint(0, k_, size=len(leaf_nodes))
        leaf_weighted_dists = leaf_distributions * leaf_samples[:, np.newaxis]

        partition_weighted_distributions = np.zeros((k_, leaf_distributions.shape[1]), dtype=np.float64)
        for i in range(k_):
            mask = assignments == i
            weighted_distributions = leaf_weighted_dists[mask]
            partition_weighted_distribution = np.sum(weighted_distributions, axis = 0)
            partition_weighted_distributions[i] = partition_weighted_distribution

        leaf_side_partition_weighted_distributions = np.zeros((k_, leaf_distributions.shape[1]), dtype=np.float64)
        if leaf_sides is not None:
            for i in range(k_):
                mask = leaf_sides == i
                weighted_distributions = leaf_weighted_dists[mask]
                partition_weighted_distribution = np.sum(weighted_distributions, axis = 0)
                leaf_side_partition_weighted_distributions[i] += partition_weighted_distribution

        assignments, partition_weighted_distributions, total_impurity = run_descent(
            len(leaf_nodes),
            assignments,
            leaf_weighted_dists,
            partition_weighted_distributions,
            k_,
            self.criterion_flag,
            max_iter=self.max_iter,
            seed = self.random_state,
            leaf_sides=leaf_sides,
            leaf_side_partition_weighted_distributions=leaf_side_partition_weighted_distributions
        )
        mapping = np.zeros(leaf_nodes.max() + 1, dtype=np.int32)
        for i in range(len(leaf_nodes)):
            mapping[leaf_nodes[i]] = assignments[i]
        
        impurities = {}
        impurities = {k_val: self.calculate_impurity(partition_weighted_distributions[k_val], weighted = True) for k_val in range(partition_weighted_distributions.shape[0])}
        partition_weights = [np.sum(partition_weighted_distributions[part_]) for part_ in range(k_)]
        impurity_decrease = self.initial_impurity - total_impurity
        return {
            'weighted_distributions': partition_weighted_distributions,
            'impurities': impurities,
            'impurity': total_impurity,
            'partition_weights': partition_weights,
            'impurity_decrease': impurity_decrease,
            'mapping': mapping,
        }

    def calculate_impurity(self, distribution, weighted = False):
        if weighted:
            if np.sum(distribution) == 0:
                return 0
            distribution = distribution / np.sum(distribution)
        
        if self.criterion == 'gini':
            return 1 - np.sum(distribution**2)
        elif self.criterion == 'entropy':
            impurity = 0
            for p in distribution:
                if p > 0:
                    impurity -= p * np.log2(p)
            return impurity
        else:
            raise ValueError('Criterion must be either gini or entropy')


def calculate_total_impurity(weighted_dist: np.ndarray, criterion_flag: int) -> float:
    """
    weighted_dist: 2D array of shape (n_partitions, n_classes)
                   each row i is the count-vector for partition i.
    criterion_flag: 0 for Gini, 1 for entropy
    """
    imp = 0.0
    total = 0.0
    n_parts, n_classes = weighted_dist.shape

    for i in range(n_parts):
        # 1) compute sum of counts for this partition
        s = 0.0
        for j in range(n_classes):
            s += weighted_dist[i, j]
        if s == 0.0:
            continue

        if criterion_flag == 0:
            # Gini: sum * (1 - sum_k (p_k^2))
            dot = 0.0
            for j in range(n_classes):
                p = weighted_dist[i, j] / s
                dot += p * p
            imp += s * (1.0 - dot)
        else:
            # Entropy: - sum * sum_k (p_k * log2 p_k)
            e = 0.0
            for j in range(n_classes):
                p = weighted_dist[i, j] / s
                if p > 0.0:
                    e += p * np.log2(p)
            imp -= s * e

        total += s

    if total == 0.0:
        return 0.0
    return imp / total

def run_descent(n_leaf_nodes: int,
                assignments: np.ndarray, # assignments from kmeans
                leaf_weighted_dists: np.ndarray, # leaf_weights
                partition_weighted_distributions: np.ndarray, # left/right weighted dists from kmeans
                k_: int,
                criterion_flag: int,
                max_iter: int = 10,
                seed: int = 42,
                leaf_sides: np.ndarray = None, # assignments from root split
                leaf_side_partition_weighted_distributions: np.ndarray = None # left/right weighted dists from root 
                ):
    counter = 0
    total_impurity = calculate_total_impurity(
        partition_weighted_distributions, criterion_flag=criterion_flag
    )
    if leaf_sides is not None: # if we have leaf sides, check if they are better. if yes, use them as init
        leaf_sides_total_impurity = calculate_total_impurity(
            leaf_side_partition_weighted_distributions, criterion_flag=criterion_flag
        )
        if leaf_sides_total_impurity < total_impurity:
            total_impurity = leaf_sides_total_impurity
            assignments = leaf_sides.copy()
            partition_weighted_distributions = leaf_side_partition_weighted_distributions.copy()
        else:
            leaf_sides_total_impurity = np.inf

    old_total_impurity = total_impurity
    best_impurity = total_impurity
    if max_iter == 0:
        return assignments, partition_weighted_distributions, total_impurity
    rng = np.random.default_rng(seed)
    for _ in range(max_iter):
        leaves = rng.permutation(n_leaf_nodes)
        switched = False
        for leaf in leaves:
            curr_assignment = assignments[leaf]
            contr = leaf_weighted_dists[leaf]
            
            # Temporarily remove contribution of current leaf
            partition_weighted_distributions[curr_assignment] -= contr
            
            best_assignment = curr_assignment
            
            for k_val in range(k_):
                if k_val == curr_assignment: # skip the current assignment as this is the best impurity
                    continue
                # Temporarily add leaf's contribution to new candidate partition
                partition_weighted_distributions[k_val] += contr
                
                # Compute impurity only if candidate partition is changed
                new_impurity = calculate_total_impurity(partition_weighted_distributions, criterion_flag=criterion_flag)
                
                if new_impurity < best_impurity:
                    best_impurity = new_impurity
                    best_assignment = k_val
                    switched = True
                
                # Restore partition distribution
                partition_weighted_distributions[k_val] -= contr
            
            # Permanently assign the leaf to the best partition
            assignments[leaf] = best_assignment
            partition_weighted_distributions[best_assignment] += contr
            total_impurity = best_impurity  # Update total_impurity directly
        assert total_impurity <= old_total_impurity, f'Total impurity increased: {total_impurity} > {old_total_impurity}'
        if not switched:
            counter += 1
        else:
            counter = 0
        
        if counter > 5:
            break

    return assignments, partition_weighted_distributions, total_impurity