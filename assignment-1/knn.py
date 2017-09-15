'''knn.py

K-nearest neighbors
'''
import pandas as pd
import logging 
from experiment import Experiment
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
logger = logging.getLogger(__name__)

class KNN(Experiment):

  def __init__(self, attributes, classifications, **kwargs):
      ''' Construct the object
      '''
      super().__init__(attributes, classifications)
  
  def run(self):
      ''' Run the experiment
      '''
      logger.info('Running experiment')
      x_train, x_test, y_train, y_test = self._split_train_test()
      logger.info('Got data split')
      experiment_pipe = Pipeline([('scale', StandardScaler()), ('predict', KNeighborsClassifier())])
      print(experiment_pipe) 
      experiment_pipe.fit(x_train, y_train)
      print(experiment_pipe.score(x_test, y_test))
