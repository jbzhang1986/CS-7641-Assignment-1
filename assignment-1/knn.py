'''knn.py

K-nearest neighbors
'''
import pandas as pd
import logging 
from experiment import Experiment
logger = logging.getLogger(__name__)

class KNN(Experiment):

  def __init__(self, dataset=None, **kwargs):
      ''' Construct the object
      '''
      super().__init__(dataset)
  
  def run(self):
      ''' Run the experiment
      '''
      logger.info('Running experiment')
    
