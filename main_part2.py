"""
Jack W. Ma 

This program takes in datasets that contain information about the game sales
with many other factors in them. It will return graphs of analyses that will
provide the reader much information about the consumer market in gaming.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def salesByPlatform(df, result_path):
    """
    This method will return a graph of which it will tell the reader about
    the global sales amount of the top gaming platforms in the era past 2000.
    It will take in a dataset of which it will contain the data, df.
    """
    sns.set()
    data = df.loc[:, ["Global_Sales", "Genre", "Platform", "Year"]]
    min_year = df['Year'] >= 2000
    max_year = df['Year'] <= 2020
    min_sales = df['Global_Sales'] > 2
    data = data[min_sales & min_year & max_year]
    data.dropna()
    sns.catplot(x='Global_Sales', y='Platform', data=data, kind='bar', ci=None)
    plt.xlabel('Global_Sales')
    plt.ylabel('Platform')
    plt.title("Global Sales by Platform (2000-Present)")
    plt.savefig(result_path + 'global_sales_platform.png',
                bbox_inches='tight')
    plt.close()


def salesWithGenre(df, result_path):
    """
    This method will return a graph of which it will tell the reader about
    the global sales amount of each genre in the top 9 gaming platforms
    in the era past 2000.
    It will take in a dataset of which it will contain the data, df.
    """
    data = df.loc[:, ["Genre", "Platform", "Global_Sales", "Year"]]
    all_popular_platforms = ["Wii", "DS", "X360", "PS3", "PS2",
                             "GBA", "PS4", "XB", "PC", "PSP",
                             "XONE", "GC", "WiiU", "GB", "PS", "N64", "PSV"]
    min_year = df['Year'] >= 2000
    max_year = df['Year'] <= 2020
    for each_platform in all_popular_platforms:
        sns.set()
        file_name = "genere_sales_" + each_platform + ".png"
        is_platform = data['Platform'] == each_platform
        temp_df = data[is_platform & max_year & min_year]
        temp_df.dropna()
        if temp_df.empty is not True:
            sns.catplot(x='Global_Sales', y='Genre', data=temp_df,
                        kind='bar', ci=None)
            plt.xlabel('Global_Sales')
            plt.ylabel('Genre')
            plt.title("Sales on " + each_platform + " by Genre (2000-Present)")
            plt.savefig(result_path + file_name, bbox_inches='tight')
            plt.close()


def gameSalesWithRatingsPre2010(df, result_path):
    """
    This method will return a graph of which it will tell the user of the
    correlation between global sales and critic score in the era of before
    the year 2010.
    It will take in a dataset of which it will contain the data.
    """
    plt.clf()
    data = df.loc[:, ["Year_of_Release", "Global_Sales", "Critic_Score"]]
    cutoff_year = df['Year_of_Release'] <= 2010
    min_sales = df['Global_Sales'] >= 3
    data_one = data[cutoff_year & min_sales]
    data_one.dropna()
    cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
    sns.scatterplot(x="Critic_Score", y="Global_Sales", data=data_one,
                    hue="Global_Sales", sizes=(20, 200),
                    hue_norm=(0, 7), palette=cmap)
    plt.xlabel('Critic Score (Metacritic)')
    plt.ylabel('Global Sales (2000-2010)')
    plt.title("Critic score vs Global sales (2000-2010)")
    plt.savefig(result_path + 'global_sales_vs_rating_pre_2010.png',
                bbox_inches='tight')
    plt.close()


def gameSalesWithRatingsPast2010(df, result_path):
    """
    This method will return a graph of which it will tell the user of the
    correlation between global sales and critic score in the era of before
    the year 2010.
    It will take in a dataset of which it will contain the data.
    """
    plt.clf()
    data = df.loc[:, ["Year_of_Release", "Global_Sales", "Critic_Score"]]
    greater_year = df['Year_of_Release'] > 2010
    min_sales = df['Global_Sales'] >= 3
    data_two = data[greater_year & min_sales]
    data_two.dropna()
    cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
    sns.scatterplot(x="Critic_Score", y="Global_Sales", data=data_two,
                    hue="Critic_Score", sizes=(20, 200),
                    hue_norm=(0, 7), palette=cmap)
    plt.xlabel('Critic Score (Metacritic)')
    plt.ylabel('Global Sales (2010-Present)')
    plt.title("Critic score vs Global sales (2010-Present)")
    plt.savefig(result_path + 'global_sales_vs_rating_past_2010.png',
                bbox_inches='tight')
    plt.close()


def main():
    sns.set()
    file_path = 'game_data/'
    result_path = 'result/'
    df = pd.read_csv(file_path + 'vgsales.csv', na_values='---')
    df2 = pd.read_csv(file_path + 'Video_Games_Sales_as_at_22_Dec_2016.csv',
                      na_values='---')
    salesByPlatform(df, result_path)
    salesWithGenre(df, result_path)
    gameSalesWithRatingsPre2010(df2, result_path)
    gameSalesWithRatingsPast2010(df2, result_path)


if __name__ == '__main__':
    main()
