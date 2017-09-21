'''knn.py

K-nearest neighbors
'''
import logging 
import numpy as np
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
logger = logging.getLogger(__name__)

class KNN(Experiment):

  def __init__(self, attributes, classifications, dataset, **kwargs):
      ''' Construct the object
      '''
      pipeline = Pipeline([('scale', StandardScaler()), ('predict', KNeighborsClassifier())])
      params = {
        'predict__metric':['manhattan','euclidean','chebyshev'],
        'predict__n_neighbors': np.arange(1, 30, 3),
        'predict__weights': ['uniform','distance']
      }
      learning_curve_train_sizes = np.arange(0.01, 1.0, 0.025)
      super().__init__(attributes, classifications, dataset, 'knn', pipeline, params, 
        learning_curve_train_sizes, True)
