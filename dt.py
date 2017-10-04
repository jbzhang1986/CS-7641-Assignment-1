'''dt.py

Decision Trees
'''
import matplotlib.pyplot as plt
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
      self._alphas = np.append(np.arange(0.99, 1.03, 0.001), 0)
      params = {
        'predict__criterion': ['gini','entropy'],
        'predict__alpha': self._alphas,
        'predict__class_weight': ['balanced']
      }
      learning_curve_train_sizes = np.arange(0.01, 1.0, 0.025)
      super().__init__(attributes, classifications, dataset, 'dt', pipeline, params, 
        learning_curve_train_sizes, True, verbose=1)

  def run(self):
      '''Run the expierment, but we need to check the prine size
      '''
      cv_best = super().run()
      logger.info('Writing out nodes')
      x_train, _, y_train, _ = self._split_train_test()
      clf = cv_best.best_estimator_
      rows = []
      nodes = []
      # alpha vs internal nodes
      for alpha in self._alphas[:-1]:
          clf.set_params(**{'predict__alpha': alpha })
          clf.fit(x_train, y_train)
          node_count = clf.steps[-1][-1].num_nodes()
          rows.append([alpha, node_count])
          nodes.append(node_count)
      out = pd.DataFrame(columns=['alpha', 'nodes'], data=rows)
      csv_str = '{}/{}'.format(self._dataset, self._algorithm)
      out.to_csv('./results/{}/node-count.csv'.format(csv_str), index=False)
      plt.figure(4)
      plt.plot(self._alphas[:-1], nodes, marker='o', color='blue')
      plt.grid(linestyle='dotted')
      plt.xlabel('Weighted Accuracy of Pruned vs Non-Prune Tree')
      plt.ylabel('Nodes')
      plt.savefig('./results/{}/node-count.png'.format(csv_str))
      
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
            if self.alpha * score >= bestScore:
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
