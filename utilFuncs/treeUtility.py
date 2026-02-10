from .utilTemplate import utilTemplate
import numpy as np
from collections import defaultdict
import numbers

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
                    result[i] = self._evaluate(stage[self.class_index].tree_, subset, 0)
            
            if self.init_logit.size > 1:                       
                outcome = self.learning_rate * result.sum() + self.init_logit[self.class_index]
            else:
                outcome = self.learning_rate * result.sum() + self.init_logit[0]
                
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
    
    
    def treestab(self, semivalue):
        if isinstance(self.tree, np.ndarray):
            # for models trained using sklearn.ensemble.GradientBoostingClassifier/GradientBoostingRegressor
            shape = np.shape(self.tree)
            result = np.empty((shape[0], self.n_players), dtype=np.float64)
            for i, stage in enumerate(self.tree):
                if shape[1] == 1:
                    result[i] = self.treestab_(stage[0].tree_, semivalue, 0)
                else:
                    result[i] = self.treestab_(stage[self.class_index].tree_, semivalue, 0)
                
            outcome = self.learning_rate * result.sum(axis=0)
            if shape[1] == 1 and self.class_index == 0:
                outcome = -outcome           
        else:
            # for models trained using sklearn.tree.DecisionTreeClassifier/DecisionTreeRegressor
            outcome = self.treestab_(self.tree, semivalue, self.class_index or 0)
            
        return outcome
    
    
    def treestab_(self, tree, semivalue, value_index):
        if isinstance(semivalue, tuple):
            assert len(semivalue) == 2
            alpha, beta = semivalue
            assert isinstance(alpha, numbers.Integral) and alpha > 0
            assert isinstance(beta, numbers.Integral) and beta > 0
            return self.treegrad_shap_(tree, semivalue, value_index)
        else:
            assert 0 < semivalue and semivalue < 1
            return self.treegrad_(tree, semivalue, value_index)
        
    
    def treegrad_shap_(self, tree, semivalue, value_index):
        alpha, beta = semivalue
        D = min(tree.max_depth, len(self.x)) + alpha + beta - 2
        n_points = -(-D // 2)
        points, weights = np.polynomial.legendre.leggauss(n_points)
        points += 1
        points /= 2
        weights /= 2
        
        tmp_alpha = np.arange(1, alpha, dtype=np.float64)
        tmp_beta = np.arange(1, beta, dtype=np.float64)
        tmp = np.arange(1, alpha+beta, dtype=np.float64)[::-1]
        init = ((tmp[:beta-1] * tmp[-1] / tmp_beta)[:,None] * points[None,:]).prod(axis=0)
        init *= ((tmp[beta-1:-1] / tmp_alpha)[:,None] * (1-points)[None,:]).prod(axis=0)
        
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        n_node_samples = tree.n_node_samples
        value = tree.value[:, 0, value_index]
        quotient = n_node_samples / n_node_samples[0]
        
        features_seen = defaultdict(list)
        gammas = np.full(len(value), -1, dtype=np.float64)
        ss = dict()  
     
        
        def traverse(node, n_samples_parent, feature_parent, activation, s=None):
            if s is None:
                s = init.copy()
            
            n_samples_current = n_node_samples[node]
            gamma = n_samples_parent / n_samples_current * activation

            if len(features_seen[feature_parent]):
                gamma_ancestor = gammas[features_seen[feature_parent][-1]]
                s /= 1 - points + points * gamma_ancestor
                gamma *= gamma_ancestor
                s *= 1 - points + points * gamma
            else:
                s *= 1 - points + points * gamma
            
            left, right = children_left[node], children_right[node]
            if left == right:
                s *= value[node] * quotient[node]
            else:
                gammas[node] = gamma
                features_seen[feature_parent].append(node)
                ss[node] = np.zeros(n_points, dtype=np.float64)
                
                feature_current = feature[node]
                if self.x[feature_current] <= threshold[node]:
                    activation_left, activation_right = 1, 0
                else:
                    activation_left, activation_right = 0, 1         
                s_a = traverse(left, n_samples_current, feature_current, activation_left, s.copy())
                s_b = traverse(right, n_samples_current, feature_current, activation_right, s.copy())
                s = s_a + s_b
                
                features_seen[feature_parent].pop()
                
            if len(features_seen[feature_parent]):
                node_ancestor = features_seen[feature_parent][-1]
                ss[node_ancestor] -= s      
            
            if left == right:
                s_current = s.copy()
            else:
                s_current = ss.pop(node)
                s_current += s
            s_current /= 1 - points + points * gamma
            phi[feature_parent] += np.dot((gamma - 1) * s_current, weights)
            
            return s
            
        phi = np.zeros(len(self.x), dtype=np.float64)
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
    
    
    def treegrad_(self, tree, semivalue, value_index):
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        n_node_samples = tree.n_node_samples
        value = tree.value[:, 0, value_index]
        quotient = n_node_samples / n_node_samples[0]
        
        features_seen = defaultdict(list)
        gammas = np.full(len(value), -1, dtype=np.float64)
        ss = dict()
        
        def traverse(node, n_samples_parent, feature_parent, activation, s=None):
            if s is None:
                s = np.float64(1)
            
            n_samples_current = n_node_samples[node]
            gamma = n_samples_parent / n_samples_current * activation
            
            if len(features_seen[feature_parent]):
                gamma_ancestor = gammas[features_seen[feature_parent][-1]]
                s /= 1 - semivalue + semivalue * gamma_ancestor
                gamma *= gamma_ancestor
                s *= 1 - semivalue + semivalue * gamma
            else:
                s *= 1 - semivalue + semivalue * gamma
            
            left, right = children_left[node], children_right[node]
            if left == right:     
                s *= value[node] * quotient[node]
            else:
                gammas[node] = gamma
                features_seen[feature_parent].append(node)
                ss[node] = np.float64(0)
                
                feature_current = feature[node]
                if self.x[feature_current] <= threshold[node]:
                    activation_left, activation_right = 1, 0
                else:
                    activation_left, activation_right = 0, 1         
                s_a = traverse(left, n_samples_current, feature_current, activation_left, s)
                s_b = traverse(right, n_samples_current, feature_current, activation_right, s)
                s = s_a + s_b
                
                features_seen[feature_parent].pop()
                
            if len(features_seen[feature_parent]):
                node_ancestor = features_seen[feature_parent][-1]
                ss[node_ancestor] -= s      
            
            if left == right:
                s_current = s
            else:
                s_current = ss.pop(node)
                s_current += s
            s_current /= 1 - semivalue + semivalue * gamma
            gradient[feature_parent] += (gamma - 1) * s_current
            
            return s        
            
        gradient = np.zeros(len(self.x), dtype=np.float64)
        left, right = children_left[0], children_right[0]
        feature_root = feature[0]
        n_samples_root = n_node_samples[0]
        if self.x[feature_root] <= threshold[0]:
            activation_left, activation_right = 1, 0
        else:
            activation_left, activation_right = 0, 1
        traverse(left, n_samples_root, feature_root, activation_left)
        traverse(right, n_samples_root, feature_root, activation_right)
        
        return gradient
