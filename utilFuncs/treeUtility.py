from .utilTemplate import utilTemplate
import numpy as np
from collections import defaultdict

class treeUtility(utilTemplate):
    def __init__(self, model, x, class_index=None):
        super().__init__()
        # if class_index = None, model is treated as regression trees.
        self.class_index = class_index
        self.x = x
        self.n_players = len(x)
            
        if hasattr(model, 'estimators_'):
            # for models trained using sklearn.ensemble.GradientBoostingClassifier/GradientBoostingRegressor
            self.tree = model.estimators_
            self.learning_rate = model.learning_rate
            self.init_logit = model._raw_predict_init(x[None,:])[0]
        else:
            # for models trained using sklearn.tree.DecisionTreeClassifier/DecisionTreeRegressor
            self.tree = model.tree_
            
    
    def evaluate(self, subset):        
        if isinstance(self.tree, np.ndarray):
            shape = np.shape(self.tree)
            result = np.empty(shape[0], dtype=np.float64)
            for i, stage in enumerate(self.tree):
                if shape[1] == 1:
                    result[i] = self._evaluate(stage[0].tree_, subset, 0)
                else:
                    assert self.class_index is not None
                    result[i] = self._evaluate(stage[self.class_index].tree_, subset, 0)
            outcome = self.learning_rate * result.sum()
            if shape[1] == 1 and self.class_index == 0:
                outcome = -outcome
        else:
            outcome = self._evaluate(self.tree, subset, self.class_index or 0) 
        return outcome
            
             
        
    def _evaluate(self, tree, subset, value_index):
        value = tree.value[:, 0, value_index]
        
        def traverse(node, n_sample_parent=None):
            left = tree.children_left[node]
            right = tree.children_right[node]
            if left == right:
                collect = value[node].copy()
            else:            
                feature = tree.feature[node]
                if subset[feature]:
                    if self.x[feature] <= tree.threshold[node]:
                        node_next = left
                    else:
                        node_next = right
                    collect = traverse(node_next)
                else:
                    n_sample_cur = tree.n_node_samples[node]
                    collect = traverse(left, n_sample_cur)
                    collect += traverse(right, n_sample_cur)  
                    
            if n_sample_parent is not None:
                collect *= tree.n_node_samples[node] / n_sample_parent
                
            return collect
        
        return traverse(0)
    
    
    def treeprob(self, semivalue):
        if isinstance(self.tree, np.ndarray):
            shape = np.shape(self.tree)
            result = np.empty((shape[0], self.n_players), dtype=np.float64)
            for i, stage in enumerate(self.tree):
                if shape[1] == 1:
                    result[i] = self._treeprob(stage[0].tree_, semivalue, 0)
                else:
                    assert self.class_index is not None
                    result[i] = self._treeprob(stage[self.class_index].tree_, semivalue, 0)
                    
            outcome = self.learning_rate * result.sum(axis=0)
            if shape[1] == 1 and self.class_index == 0:
                outcome = -outcome           
        else:
            outcome = self._treeprob(self.tree, semivalue, self.class_index or 0)
            
        return outcome
    
    
    def _treeprob(self, tree, semivalue, value_index):
        D = min(tree.max_depth, len(self.x))
        weights = self.compute_weights(semivalue, D)

        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        n_node_samples = tree.n_node_samples
        value = tree.value[:, 0, value_index]
        quotient = n_node_samples / n_node_samples[0]
        
        nodes = np.exp(1j * 2 * np.pi / D * np.arange(D), dtype=np.complex128) 
        weights = np.vander(nodes, increasing=True).conjugate().dot(weights) / D 
        M_scaling = np.vander(nodes+1, increasing=True)
         
        features_seen = defaultdict(list)
        gammas = np.full(len(value), -1, dtype=np.float64)
        polynomials = dict()
        
        def traverse(node, n_samples_parent, feature_parent, activation, p=None, degree=None):
            if p is None:
                p = np.ones(D, dtype=np.complex128)
                degree = 1
            
            n_samples_current = n_node_samples[node]
            gamma = n_samples_parent / n_samples_current * activation
            if len(features_seen[feature_parent]):
                gamma_ancestor = gammas[features_seen[feature_parent][-1]]
                p /= 1 + gamma_ancestor * nodes
                gamma *= gamma_ancestor
                p *= 1 + gamma * nodes
            else:
                p *= 1 + gamma * nodes
                degree += 1
            
            left, right = children_left[node], children_right[node]
            if left == right:                
                p *= value[node] * quotient[node]
                p *= M_scaling[:, D-degree+1]
            else:
                gammas[node] = gamma
                features_seen[feature_parent].append(node)
                polynomials[node] = np.zeros(D, dtype=np.complex128)
                
                feature_current = feature[node]
                if self.x[feature_current] <= threshold[node]:
                    activation_left, activation_right = 1, 0
                else:
                    activation_left, activation_right = 0, 1         
                p_a = traverse(left, n_samples_current, feature_current, activation_left, p.copy(), degree)
                p_b = traverse(right, n_samples_current, feature_current, activation_right, p.copy(), degree)
                p = p_a + p_b
                
                features_seen[feature_parent].pop()
                
            if len(features_seen[feature_parent]):
                node_ancestor = features_seen[feature_parent][-1]
                polynomials[node_ancestor] -= p        
            
            if left == right:
                p_current = p.copy()
            else:
                p_current = polynomials.pop(node)
                p_current += p
            p_current /= 1 + gamma * nodes
            phi[feature_parent] += (gamma - 1) * np.dot(p_current, weights).real
            
            return p
            
        phi = np.zeros(self.n_players, dtype=np.float64)
        left, right = children_left[0], children_right[0]
        feature_root = feature[0]
        n_samples_root = n_node_samples[0]
        if self.x[feature_root] <= threshold[0]:
            activation_left, activation_right = 1, 0
        else:
            activation_left, activation_right = 0, 1
        traverse(left, n_samples_root, feature_root, activation_left)
        traverse(right, n_samples_root, feature_root, activation_right)
        
        return phi