'''ann.py

Artifical Neural Networks
'''
import logging 
import numpy as np
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
logger = logging.getLogger(__name__)

class ANN(Experiment):

  def __init__(self, attributes, classifications, dataset, **kwargs):
      ''' Construct the object
      '''
      pipeline = Pipeline([('scale', StandardScaler()), ('predict', MLPClassifier(random_state=10, max_iter=2000, early_stopping=True))])
      params = {
        'predict__activation':['logistic', 'relu'],
        'predict__alpha': np.arange(0.05, 3, 0.1),
        'predict__hidden_layer_sizes': [(32), (64), (128), (32, 64, 32), (64, 128, 64)]
      }
      learning_curve_train_sizes = np.arange(0.01, 1.0, 0.025)
      super().__init__(attributes, classifications, dataset, 'ann', pipeline, params, 
        learning_curve_train_sizes, True, verbose=1, iteration_curve=True)
