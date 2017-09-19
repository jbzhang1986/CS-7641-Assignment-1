'''experiment.py

Share all of the experiment items here
'''
import numpy as np
import logging 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight
logger = logging.getLogger(__name__)

def balanced_accuracy(truth,pred):
    wts = compute_sample_weight('balanced',truth)
    return accuracy_score(truth,pred,sample_weight=wts)
scorer = make_scorer(balanced_accuracy) 


class Experiment:

    def __init__(self, attributes, classifications,
      dataset, algorithm, pipeline, params, learning_curve_train_sizes):
        ''' Constructor
        '''
        # what data are we looking at
        self._atttributes = attributes
        self._classifications = classifications
        self._dataset = dataset
        self._algorithm = algorithm
        self._cv = 10
        self._pipeline = pipeline
        self._params = params
        self._learning_curve_train_sizes = learning_curve_train_sizes
        self._verbose = 0
        self._random = 10
        # force a seed for the experiment
        np.random.seed(self._random)

    def run(self):
        ''' Run the expierment
        '''
        logger.info('Running the experiment')
        x_train, x_test, y_train, y_test = self._split_train_test()
        logger.info('Got data split')
        experiment_pipe = self._pipeline
        model_params = self._params
        # cross validation -> best estimator
        # refit is true ...
        cv = GridSearchCV(experiment_pipe,
          n_jobs=4, param_grid=model_params, cv=self._cv, scoring=scorer,
          refit=True, verbose=self._verbose)
        logger.info('Searching params')
        cv.fit(x_train, y_train)
        cv_all = pd.DataFrame(cv.cv_results_)
        csv_str = '{}-{}'.format(self._dataset, self._algorithm)
        cv_all.to_csv('./results/{}-cv.csv'.format(csv_str), index=False)
        self._basic_accuracy(cv, x_test, y_test, csv_str)
        self._learning_curve(cv, x_train, y_train, csv_str)

    def _basic_accuracy(self, cv, x_test, y_test, csv_str):
        # simple best fit against test data
        logger.info('Writing out basic result')
        results_df = pd.DataFrame(columns=['best_estimator', 'best_score', 'best_params', 'test_score'],
          data=[[cv.best_estimator_, cv.best_score_, cv.best_params_, cv.score(x_test, y_test)]])
        results_df.to_csv('./results/{}-basic.csv'.format(csv_str), index=False)
        
    def _learning_curve(self, cv, x_train, y_train, csv_str):
        # learning curve
        logger.info('Creating learning curve')
        accuracy_learning_curve = learning_curve(cv.best_estimator_, x_train, y_train,
          cv=self._cv, train_sizes = self._learning_curve_train_sizes, verbose=self._verbose, 
          scoring=scorer, n_jobs=4)
        train_scores = pd.DataFrame(index = accuracy_learning_curve[0], data = accuracy_learning_curve[1])
        train_scores.to_csv('./results/{}-lc-train.csv'.format(csv_str), index=False)
        test_scores = pd.DataFrame(index = accuracy_learning_curve[0], data = accuracy_learning_curve[2])
        test_scores.to_csv('./results/{}-lc-test.csv'.format(csv_str), index=False)
        logger.info('Saving learning curves')
        plt.figure(1)
        plt.plot(self._learning_curve_train_sizes, train_scores, marker='o', color='blue',
          label='Training Score')
        plt.plot(self._learning_curve_train_sizes, test_scores, marker='o', color='green',
          label='Cross-Validation Score')
        plt.legend()
        plt.grid(linestyle='dotted')
        plt.xlabel('Training Examples Percentage')
        plt.ylabel('Accuracy')
        plt.savefig('./results/{}-learning-curve.png'.format(csv_str))
    
    def _split_train_test(self, test_size=0.3):
        '''Split up the data correctly according to a ratio

        Returns:
            The split data
        '''
        return train_test_split(self._atttributes, self._classifications,
          test_size=test_size, random_state=self._random, stratify=self._classifications)
