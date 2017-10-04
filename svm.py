'''svm.py

Support Vector Machines
'''
import logging 
import numpy as np
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
logger = logging.getLogger(__name__)

class SVM(Experiment):

  def __init__(self, attributes, classifications, dataset, **kwargs):
      ''' Construct the object
      '''
      pipeline = Pipeline([('scale', StandardScaler()), ('predict', SVC())])
      params = {
        'predict__kernel': ['linear', 'poly', 'rbf'],
        'predict__C': 10.0 ** np.arange(-3, 8), 
        # penalize distance, low = use all, high = use close b/c distance to decision boundry to penalized
        'predict__gamma': 10. ** np.arange(-5, 4),
        'predict__cache_size': [200],
        'predict__max_iter': [3000],
        'predict__degree': [2, 3],
        'predict__coef0': [0, 1]
      }
      learning_curve_train_sizes = np.arange(0.05, 1.0, 0.05)
      super().__init__(attributes, classifications, dataset, 'svm', pipeline, params, 
        learning_curve_train_sizes, True, verbose=0, iteration_curve=True)
