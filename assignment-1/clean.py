'''clean.py

Make the datasets correct for the experiment
'''
import pandas as pd
import logging 
logger = logging.getLogger(__name__)

def create_final_datasets():
    ''' Munge the datasets
    '''
    create_wine_dataset()
    create_credit_card_dataset()
    
def create_wine_dataset():
    '''Create the final wine dataset
    '''
    logger.info('Cleaning wine datasets')
    wine_red = pd.read_csv('./data/winequality-red-original.csv', sep=';')
    wine_red['red'] = 1
    wine_white = pd.read_csv('./data/winequality-white-original.csv', sep=';')
    wine_white['red'] = 0
    logger.info('Red wine dataframe shape %s', wine_red.shape)
    logger.info('White wine dataframe shape %s', wine_white.shape)
    logger.info('Randomly selecting wines to match red')
    wine_white_final = wine_white.sample(wine_red.shape[0], random_state=0)
    wine_final = wine_red.append(wine_white_final)
    wine_final.columns = ['_'.join(col.split(' ')) for col in wine_final.columns]
    logger.info('Final wine set information \n %s', wine_final.describe(include='all'))
    logger.info('Writing final wine csv to ./data/wine-red-white-final.csv')
    wine_final.to_csv('./data/wine-red-white-final.csv', index=False)

def create_credit_card_dataset():
    '''Create the final credit card dataset
    '''
    logger.info('Cleaning credit card dataset')
    credit_card = pd.read_csv('./data/credit-card-original.csv', skiprows=1)
    # drop dummy header
    credit_card = credit_card.drop('ID', axis=1)
    logger.info('Dropped unnecessary information')
    credit_card.columns = map(str.lower, credit_card.columns)
    credit_card.columns = ['_'.join(col.split(' ')) for col in credit_card.columns]
    logger.info('Initial credit card information \n %s', credit_card.describe())
    logger.info('Sampling down to 10%')
    credit_card = credit_card.groupby('default_payment_next_month')
    credit_card = credit_card.apply(pd.DataFrame.sample, frac=0.1, random_state=0).reset_index(drop=True)
    logger.info('Final credit card information \n %s', credit_card.describe(include='all'))
    logger.info('Writing final credit card csv to ./data/credit-card-final.csv')
    credit_card.to_csv('./data/credit-card-final.csv', index=False)
