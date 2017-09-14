'''knn.py

K-nearest neighbors
'''
import pandas as pd
import logging 
logger = logging.getLogger(__name__)

class KNN():

  def __init__(self):
      pass
  
  def run(self):
      logger.info(pd.read_csv('./data/wine-red-white-final.csv'))
    
