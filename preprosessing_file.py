import pandas as pd
import numpy as np


def preprossesing(data_list, dct):
    def season_ch(month):
        month = int(month.split('-')[1])
        if 3 <= month <= 5:
            return 'spring'
        elif 6 <= month <= 8:
            return 'summer'
        elif 9 <= month <= 11:
            return 'autumn'
        else:
            return 'winter'

    def daytime_ch(time):
        time = int(time.split(':')[0])
        if 0 <= time <= 5:
            return 'night'
        elif 6 <= time <= 11:
            return 'morning'
        elif 12 <= time <= 17:
            return 'day'
        else:
            return 'evening'

    def to_int(x):
        if int(x) >= 5:
            return 5
        else:
            return int(x)

    data_list[0], data_list[1], data_list[2] = season_ch(data_list[0]), daytime_ch(data_list[1]), to_int(data_list[2])
    dataframe = pd.DataFrame(
        columns=['visit_date', 'visit_time', 'visit_number', 'utm_source', 'utm_medium', 'utm_campaign',
                 'utm_adcontent', 'device_category', 'device_brand', 'device_screen_resolution', 'device_browser',
                 'geo_country', 'geo_city'],
        data=[data_list])
    for col in dataframe.columns:
        dataframe[col].iloc[0] = dataframe[col].iloc[0] if dataframe[col].iloc[0] in dct[col] else 'other'

    return dataframe


