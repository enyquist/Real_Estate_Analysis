import json
import os
import pandas as pd
import numpy as np
import requests
import math

RAPIDAPI_KEY = os.environ.get('RAPIDAPI_KEY')
RAPIDAPI_HOST_RE = os.environ.get('RAPIDAPI_HOST_RE')
RAPIDAPI_OFFSET = 200


class RealEstateData:
    """
    RealEstateData is designed to collect real estate listings for analysis from a given CITY, STATE,
    parsing data from the RapidAPI for Realtor.com.

    Use Guidelines:

    my_real_estate = RealEstateData('CITY', 'STATE_CODE')

    my_real_estate_results = my_real_estate.results

    To Do:
    - Check for null values in API return
    - Check for invalid input
    - Reduce number of API calls, currently 1 + N per instance, where N is CEIL(total_results / 200)
    """

    def __init__(self, city, state):
        self.city = city.upper()
        self.state = state.upper()
        self._jsonREData = self.fetch_housing_data()
        self.results = self.parse()

    def __repr__(self):
        return f"RealEstateData('{self.city, self.state}')"

    def __str__(self):
        return f'{self.city, self.state} real estate data'

    def fetch_housing_data(self):
        """
        Function to fetch all housing data from Realtor.com via RapidAPI
        :return: Dictionary of Dictionaries containing all the results from the the API call
        """
        url = "https://realtor-com-real-estate.p.rapidapi.com/for-sale"

        housing_total = self.fetch_housing_total()

        list_offset = self.define_chunks(housing_total)

        list_json_data = []

        for offset in list_offset:

            querystring = {"city": self.city,
                           "offset": offset,
                           "state_code": self.state,
                           "limit": "200",
                           "sort": "newest"}

            headers = {
                'x-rapidapi-key': RAPIDAPI_KEY,
                'x-rapidapi-host': RAPIDAPI_HOST_RE
            }

            response = requests.request("GET", url, headers=headers, params=querystring)

            if response.status_code == 200:
                json_content = json.loads(response.content)
                list_json_data.append(json_content)
            else:
                pass

        return list_json_data

    def fetch_housing_total(self):
        """
        Function to fetch the total number of listings in CITY, STATE from Realtor.com via Rapid API
        :return: Total number of listings in CITY, STATE as int
        """
        url = "https://realtor-com-real-estate.p.rapidapi.com/for-sale"

        querystring = {"city": self.city,
                       "offset": 0,
                       "state_code": self.state,
                       "limit": "1",
                       "sort": "newest"}

        headers = {
            'x-rapidapi-key': RAPIDAPI_KEY,
            'x-rapidapi-host': RAPIDAPI_HOST_RE
        }

        response = requests.request("GET", url, headers=headers, params=querystring)

        if response.status_code == 200:
            json_content = json.loads(response.content)
            housing_total = json_content['data']['total']
            return housing_total
        else:
            return None

    @staticmethod
    def define_chunks(total, chunk=RAPIDAPI_OFFSET):
        """
        Function to define the offset to collect total number of listings in CITY, STATE from Realtor.com via Rapid API
        :param total: Total number of listings
        :param chunk: Offset to collect on, RAPIDAPI can return up to RAPIDAPI_OFFSET each call
        :return: list of offsets needed to collect entire dataset
        """

        list_chunk_sizes = []
        for x in range(math.ceil(total / chunk)):
            list_chunk_sizes.append(chunk * x)

        return list_chunk_sizes

    def parse(self):
        """
        Function to format the entire dataset as a DataFrame
        :return: DataFrame built from total dataset
        """

        list_results_dfs = [pd.json_normalize(result['data']['results']) for result in self._jsonREData]

        df_results = pd.concat(list_results_dfs, ignore_index=True)

        return df_results


def probability(df):
    """

    :param df:
    :return:
    """
    s = np.sum(df, axis=0)
    m = len(df)
    mu = s / m
    vr = np.sum((df - mu) ** 2, axis=0)
    variance = vr / m
    var_dia = np.diag(variance)
    k = len(mu)
    X = df - mu
    p = 1 / ((2 * np.pi) ** (k / 2) * (np.linalg.det(var_dia) ** 0.5)) * np.exp(
        -0.5 * np.sum(X @ np.linalg.pinv(var_dia) * X, axis=1))
    return p


def feat_importance(m, df_train):
    """

    :param m:
    :param df_train:
    :return:
    """
    importance = m.feature_importances_
    importance = pd.DataFrame(importance, index=df_train.columns, columns=["Importance"])
    return importance.sort_values(by=['Importance'], ascending=False)
