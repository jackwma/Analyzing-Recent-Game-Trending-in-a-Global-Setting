"""
Hanzhi Cao
CSE 163 AD

This is the first part of our project. 
It takes in datasets that contain information about the video games
with many other factors in them. 
It will return graphs of analyses that will provide the reader much
information about the consumer market in gaming.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA


def merge_data(game_sales_2016, game_sales_2017):
    """
    Returns a merged dataset that will be used in the first two
    research questions.
    Games releasing years will be limited to 1996-2015.
    """
    merged_data = pd.concat([game_sales_2017, game_sales_2016])
    merged_data = merged_data.drop_duplicates(keep=False)
    merged_data = merged_data[(merged_data['Year_of_Release'] > 1995) &
                              (merged_data['Year_of_Release'] < 2016)]
    return merged_data


def globalSalesTrend(merged_data, result_path):
    """
    Saves a most sale genre and its global sales in millions as a .png file;
    And another 12 line graphes (3 by 4) in one file displaying the global
    sales trend among game genres.
    It will take in a merged dataset, merged_data, and a string as a output
    file path, result_path.
    """
    global_sales = merged_data[['Genre', 'Year_of_Release', 'Global_Sales']]
    sum_global_sales = global_sales.groupby(['Genre', 'Year_of_Release']
                                            ).sum()
    ave_global_sales = sum_global_sales.groupby(['Genre']).mean()
    most_selling_genre = ave_global_sales[ave_global_sales.Global_Sales ==
                                          ave_global_sales.Global_Sales.max()]
    fig, ax = plt.subplots(figsize=(4, 1))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    most_selling_genre = table(ax, most_selling_genre, loc='center right',
                               colWidths=[0.75] *
                               len(most_selling_genre.columns))
    most_selling_genre.auto_set_font_size(False)
    most_selling_genre.set_fontsize(10)
    plt.savefig(result_path + 'most_selling_genre.png', transparent=True)

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(25, 10))
    sum_global_sales.unstack('Genre').plot(subplots=True, ax=ax,
                                           xlim=[1995, 2016], ylim=[0, 300])
    plt.suptitle('Global Sales trend (in millions) from 1996 to 2015)',
                 fontsize=20)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(result_path + '/global_sales_trend.png', bbox_inches='tight')


def userRatingTrend(merged_data, result_path):
    """
    Saves a higest rating genre and its rate in a 0-10 scale as a .png file;
    And another 12 line graphes (3 by 4) in one file displaying the user
    ratings trend among game genres.
    It will take in a merged dataset, merged_data, and a string as a output
    file path, result_path.
    """
    user_ratings = merged_data[['Genre', 'Year_of_Release', 'User_Score']]
    ave_user_ratings = user_ratings.groupby(['Genre', 'Year_of_Release']
                                            ).mean()
    user_ratings = ave_user_ratings.groupby(['Genre']).mean()
    highest_rating_genre = user_ratings[user_ratings.User_Score
                                        == user_ratings.User_Score.max()]
    fig, ax = plt.subplots(figsize=(4, 1))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    highest_rating_genre = table(ax, highest_rating_genre,
                                 loc='center right', colWidths=[0.6] *
                                 len(highest_rating_genre.columns))
    highest_rating_genre.auto_set_font_size(False)
    highest_rating_genre.set_fontsize(10)
    plt.savefig(result_path + 'highest_rating_genre.png', transparent=True)

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(25, 10))
    ave_user_ratings.unstack('Genre').plot(subplots=True, ax=ax,
                                           xlim=[1995, 2016], ylim=[1, 10])
    plt.suptitle('User Ratings (1-10) trend from 1996 to 2015', fontsize=20)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(result_path + 'user_rating_trend.png', bbox_inches='tight')


def predict_gs(merged_data, result_path): 
    """
    Saves 12 line graphs of rolling mean of sales among game genres.
    Predicts the most selling game genre based on the rolling mean and
    ARIMA model of machine learning.
    It will take in a merged dataset, merged_data, and a string as a output
    file path, result_path.
    """
    global_sales = merged_data[['Genre', 'Year_of_Release', 'Global_Sales']]
    sum_global_sales = global_sales.groupby(['Genre', 'Year_of_Release']
                                            ).sum()
    gs_sum = sum_global_sales.unstack(['Genre'])
    gs_series_value = gs_sum.values
    gs_genre = list(gs_sum.columns.get_level_values('Genre'))
    gs_genre[7] = 'Role_playing'

    gs_value = pd.DataFrame(gs_series_value)
    gs_series = pd.concat([gs_value, gs_value.shift(1)], axis=1)
    gs_series.columns = gs_genre + [w+'_forecast' for w in gs_genre]
    gs_series.index = range(1996, 2016)
    gs_series = gs_series[1:20]

    train = gs_series[0:14]
    test = gs_series[15:19]

    gs_mean = gs_sum.rolling(window=5).mean()
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(25,10))
    gs_mean.plot(subplots=True, ax=ax, ylim=[0,250])
    plt.savefig(result_path + 'gs_mean.png', bbox_inches='tight')

    action_predict_sales = action(train, test, gs_series)
    shooter_predict_sales = shooter(train, test, gs_series)

    result = max(action_predict_sales, shooter_predict_sales)
    if result == action_predict_sales:
        output = 'Action games'
    else:
        output = 'Shooting games'
    print('Most promising genre in selling: ' + output)


def action(train, test, gs_series):
    """
    Returns the predicted sale of action games in the future 5 years.
    It will take in three datasets, train test and gs_series, to train
    the ARIMA model.
    """
    action_model = ARIMA(train.Action, order=(0,0,1))
    action_model_fit = action_model.fit()
    action_forecast = action_model_fit.forecast(steps = 4)[0]
    action_mse = np.sqrt(mean_squared_error(test.Action_forecast,
                                            action_forecast))
    if action_mse < gs_series.Action.describe()['std']: 
        action_forecast = action_model_fit.forecast(steps = 9)[0]
        return action_forecast[4:9].mean()


def shooter(train, test, gs_series):
    """
    Returns the predicted sale of shooter games in the future 5 years.
    It will take in three datasets, train test and gs_series, to train
    the ARIMA model.
    """
    shooter_model = ARIMA(train.Shooter, order=(2,2,0))
    shooter_model_fit = shooter_model.fit()
    shooter_forecast = shooter_model_fit.forecast(steps = 4)[0]
    shooter_mse = np.sqrt(mean_squared_error(test.Shooter_forecast,
                                             shooter_forecast))
    if shooter_mse < gs_series.Shooter.describe()['std']: 
        shooter_forecast = shooter_model_fit.forecast(steps = 9)[0]
        return shooter_forecast[4:9].mean()


def predict_ur(merged_data, result_path): 
    """
    Saves 12 line graphs of rolling mean of rating among game genres.
    Predicts the higest rating game genre based on the rolling mean and
    ARIMA model of machine learning.
    It will take in a merged dataset, merged_data, and a string as a output
    file path, result_path.
    """
    user_ratings = merged_data[['Genre', 'Year_of_Release', 'User_Score']]
    ave_user_ratings = user_ratings.groupby(['Genre', 'Year_of_Release']
                                            ).mean()
    ur_sum = ave_user_ratings.unstack(['Genre'])
    ur_series_value = ur_sum.values
    ur_genre = list(ur_sum.columns.get_level_values('Genre'))
    ur_genre[7] = 'Role_playing'
    ur_value = pd.DataFrame(ur_series_value)
    ur_series = pd.concat([ur_value, ur_value.shift(1)], axis=1)
    ur_series.columns = ur_genre + [w+'_forecast' for w in ur_genre]
    ur_series.index = range(1996, 2016)
    ur_series = ur_series[1:20]
    train_ur = ur_series[0:14]
    test_ur = ur_series[15:19]

    ur_mean = ur_sum.rolling(window=5).mean()
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(25,10))
    ur_mean.plot(subplots=True, ax=ax, ylim=[5, 9])
    plt.savefig(result_path + 'ur_mean.png', bbox_inches='tight')

    train_ur_adv = train_ur.Adventure[1:19]
    train_ur_misc = train_ur.Misc[2:19]

    adv_predict_ratings = adv(train_ur_adv, test_ur, ur_series)
    misc_predict_ratings = misc(train_ur_misc, test_ur, ur_series)

    result = max(adv_predict_ratings, misc_predict_ratings)
    if result == misc_predict_ratings:
        output = 'Misc games'
    else:
        output = 'Adventure games'
    print('Most promising genre in user ratings: ' + output)


def adv(train_ur_adv, test_ur, ur_series): 
    """
    Returns the predicted rate of adventure games in the future 5 years.
    It will take in three datasets, train test and gs_series, to train
    the ARIMA model.
    """
    adv_model = ARIMA(train_ur_adv, order=(1,0,0))
    adv_model_fit = adv_model.fit()
    adv_forecast = adv_model_fit.forecast(steps = 4)[0]
    adv_mse = np.sqrt(mean_squared_error(test_ur.Adventure_forecast, adv_forecast))
    if adv_mse < ur_series.Adventure.describe()['std']: 
        adv_forecast = adv_model_fit.forecast(steps = 9)[0]
        return adv_forecast[4:9].mean()


def misc(train_ur_misc, test_ur, ur_series): 
    """
    Returns the predicted rate of Misc games in the future 5 years.
    It will take in three datasets, train test and gs_series, to train
    the ARIMA model.
    """
    misc_model = ARIMA(train_ur_misc, order=(0,0,0))
    misc_model_fit = misc_model.fit()
    misc_forecast = misc_model_fit.forecast(steps = 4)[0]
    misc_mse = np.sqrt(mean_squared_error(test_ur.Misc_forecast, misc_forecast))
    if misc_mse < ur_series.Adventure.describe()['std']: 
        misc_forecast = misc_model_fit.forecast(steps = 9)[0]
        return misc_forecast[4:9].mean()


def main():
    file_path = 'game_data/'
    result_path = 'result/'
    df1 = pd.read_csv(file_path + '/Video_Games_Sales_as_at_22_Dec_2016.csv')
    df2 = pd.read_csv(file_path + 'Video_Game_Sales_as_of_Jan_2017.csv')
    merged_data = merge_data(df1, df2)
    globalSalesTrend(merged_data, result_path)
    userRatingTrend(merged_data, result_path)
    predict_gs(merged_data, result_path)
    predict_ur(merged_data, result_path)


if __name__ == '__main__':
    main()
