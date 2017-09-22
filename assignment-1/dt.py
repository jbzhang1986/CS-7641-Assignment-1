'''dt.py

Decision Trees
'''
import logging 
import numpy as np
import pandas as pd
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
logger = logging.getLogger(__name__)

class DT(Experiment):

  def __init__(self, attributes, classifications, dataset, **kwargs):
      ''' Construct the object
      '''
      pipeline = Pipeline([('scale', StandardScaler()), ('predict', PrunedDecisionTreeClassifier())])
      params = {
        'predict__criterion':['gini','entropy'],
        'predict__alpha': np.arange(0, 2.1, 0.1),
        'predict__class_weight': ['balanced']
      }
      learning_curve_train_sizes = np.arange(0.01, 1.0, 0.025)
      super().__init__(attributes, classifications, dataset, 'dt', pipeline, params, 
        learning_curve_train_sizes, True, verbose=1)

  def DTpruningVSnodes(clf,alphas,trgX,trgY,dataset):
      '''Dump table of pruning alpha vs. # of internal nodes'''
      out = {}
      for a in alphas:
          clf.set_params(**{'DT__alpha':a})
          clf.fit(trgX,trgY)
          out[a]=clf.steps[-1][-1].numNodes()
          print(dataset,a)
      out = pd.Series(out)
      out.index.name='alpha'
      out.name = 'Number of Internal Nodes'
      out.to_csv('./output/DT_{}_nodecounts.csv'.format(dataset))

class PrunedDecisionTreeClassifier(DecisionTreeClassifier):        
    def __init__(self,
               criterion='gini',
               splitter='best',
               max_depth=None,
               min_samples_split=2,
               min_samples_leaf=1,
               min_weight_fraction_leaf=0.,
               max_features=None,
               random_state=None,
               max_leaf_nodes=None,
               min_impurity_split=1e-7,
               class_weight=None,
               presort=False,
               alpha = 0):
      super().__init__(
          criterion=criterion,
          splitter=splitter,
          max_depth=max_depth,
          min_samples_split=min_samples_split,
          min_samples_leaf=min_samples_leaf,
          min_weight_fraction_leaf=min_weight_fraction_leaf,
          max_features=max_features,
          max_leaf_nodes=max_leaf_nodes,
          class_weight=class_weight,
          random_state=random_state,
          min_impurity_split=min_impurity_split,
          presort=presort)
      self.alpha = alpha

    def remove_subtree(self, root):
        '''Clean up'''
        tree = self.tree_
        visited, stack = set(), [root]
        while stack:
            v = stack.pop()
            visited.add(v)
            left = tree.children_left[v]
            right = tree.children_right[v]
            if left >= 0:
                stack.append(left)
            if right >= 0:
                stack.append(right)
        for node in visited:
            tree.children_left[node] = -1
            tree.children_right[node] = -1
        
    def prune(self):      
        if self.alpha == 0: # Early exit
            return 
        tree = self.tree_        
        bestScore = self.score(self.valX, self.valY)        
        # children_left and right will both be non zero at the same indexes to be a leaf
        candidates = np.flatnonzero(tree.children_left >= 0)
        # candidate is an index
        for candidate in reversed(candidates): # Go backwards/leaves up
            if tree.children_left[candidate] == tree.children_right[candidate]: # leaf node. Ignore
                continue
            left = tree.children_left[candidate]
            right = tree.children_right[candidate]
            # what happens if i remove?
            tree.children_left[candidate] = tree.children_right[candidate] = -1            
            score = self.score(self.valX, self.valY)
            if score >= self.alpha * bestScore:
                bestScore = score                
                self.remove_subtree(candidate)
            else:
                tree.children_left[candidate] = left
                tree.children_right[candidate] = right
        assert (tree.children_left >= 0).sum() == (tree.children_right>=0).sum() 
        
    def fit(self, X, Y, sample_weight=None, check_input=True, X_idx_sorted=None):        
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0]) 
        self.trgX = X.copy()
        self.trgY = Y.copy()
        self.trgWts = sample_weight.copy()        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
        # leave some out to use as a score candidate otherwise overfits
        for train_index, test_index in sss.split(self.trgX, self.trgY):
            self.valX = self.trgX[test_index]
            self.valY = self.trgY[test_index]
            self.trgX = self.trgX[train_index]
            self.trgY = self.trgY[train_index]
            self.valWts = sample_weight[test_index]
            self.trgWts = sample_weight[train_index]
        super().fit(self.trgX, self.trgY, self.trgWts, check_input, X_idx_sorted)
        self.prune()
        return self
            
    def num_nodes(self):
        assert (self.tree_.children_left >= 0).sum() == (self.tree_.children_right>=0).sum() 
        return  (self.tree_.children_left >= 0).sum() 
