'''experiment.py

Share all of the experiment items here
'''
import numpy as np
import logging 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight
logger = logging.getLogger(__name__)

def balanced_accuracy(truth,pred):
    wts = compute_sample_weight('balanced',truth)
    return accuracy_score(truth,pred,sample_weight=wts)
scorer = make_scorer(balanced_accuracy) 


class Experiment:

    def __init__(self, attributes, classifications):
        ''' Constructor
        '''
        # what data are we looking at
        self._atttributes = attributes
        self._classifications = classifications
        
        # force a seed for the experiment
        np.random.seed(10)

    def _pipeline(self):
        return None

    def _params(self):
        return {}

    def run(self):
        ''' Run the expierment
        '''
        logger.info('Running the experiment')
        x_train, x_test, y_train, y_test = self._split_train_test()
        logger.info('Got data split')
        experiment_pipe = self._pipeline()
        model_params = self._params()
        # cross validation -> best estimator
        # refit is true ...
        cv = GridSearchCV(experiment_pipe, \
          n_jobs=4, param_grid=model_params, cv=10, scoring=scorer, \
          refit=True, verbose=5)
        cv.fit(x_train, y_train)
        print(cv.best_estimator_)
        print(cv.best_score_)
        print(cv.best_params_)
        print(cv.best_index_)
    
    def _split_train_test(self, test_size=0.3):
        '''Split up the data correctly according to a ratio

        Returns:
            The split data
        '''
        return train_test_split(self._atttributes, self._classifications, \
          test_size=test_size, random_state=0, stratify=self._classifications)
