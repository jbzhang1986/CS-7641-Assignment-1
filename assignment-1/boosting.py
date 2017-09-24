'''boosting.py


Boosting with decision trees
'''
import logging 
import numpy as np
from dt import PrunedDecisionTreeClassifier
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
logger = logging.getLogger(__name__)

class Boosting(Experiment):

  def __init__(self, attributes, classifications, dataset, **kwargs):
      ''' Construct the object
      '''
      if dataset == 'wine':
          basic_tree = PrunedDecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=10)
      else:
          basic_tree = PrunedDecisionTreeClassifier(criterion='gini', class_weight='balanced', random_state=10)

      pipeline = Pipeline([('scale', StandardScaler()), 
        ('predict', AdaBoostClassifier(algorithm='SAMME', base_estimator=basic_tree, random_state=10))])
      params = {
        'predict__n_estimators': np.power(2, np.arange(1, 8)),
        'predict__base_estimator_alpha': np.arange(0, 2.1, 0.1)
      }
      super().__init__(attributes, classifications, dataset, 'boosting', pipeline, params, 
        learning_curve_train_sizes, True, verbose=1, iteration_curve=True)
