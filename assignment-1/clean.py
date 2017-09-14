'''clean.py

Make the datasets correct for the experiment
'''
import pandas as pd
import logging 
logger = logging.getLogger(__name__)

def create_final_datasets():
    ''' Munge the datasets
    '''
    logger.info('Loading datasets')
    create_wine_dataset()
    
def create_wine_dataset():
    wine_red = pd.read_csv('./data/winequality-red-original.csv', sep=';')
    wine_red['red'] = 1
    wine_white = pd.read_csv('./data/winequality-white-original.csv', sep=';')
    wine_white['red'] = 0
    logger.info('Red wine dataframe shape %s', wine_red.shape)
    logger.info('White wine dataframe shape %s', wine_white.shape)
    logger.info('Randomly selecting wines to match red')
    wine_white_final = wine_white.sample(wine_red.shape[0], random_state=0)
    wine_final = wine_red.append(wine_white_final)
    logger.info('Final wine set information \n %s', wine_final.describe())

def create_credit_card_dataset():
    pass
