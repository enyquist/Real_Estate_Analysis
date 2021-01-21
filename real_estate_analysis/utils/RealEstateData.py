import json
import pandas as pd
import requests
import math
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')


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

        housing_total, dict_first_call = self.first_call

        list_json_data = [dict_first_call]

        list_offset = self.define_chunks(housing_total)

        for offset in list_offset:

            querystring = {"city": self.city,
                           "offset": offset,
                           "state_code": self.state,
                           "limit": "200",
                           "sort": "newest"}

            headers = {
                'x-rapidapi-key': config['DEFAULT']['rapidapi_key'],
                'x-rapidapi-host': config['DEFAULT']['rapidapi_host_re']
            }

            response = requests.request("GET", url, headers=headers, params=querystring)

            if response.status_code == 200:
                json_content = json.loads(response.content)
                list_json_data.append(json_content)

        return list_json_data

    @property
    def first_call(self):
        """
        Function to fetch the total number of listings in CITY, STATE from Realtor.com via Rapid API
        :return: Total number of listings in CITY, STATE as int
        """
        url = "https://realtor-com-real-estate.p.rapidapi.com/for-sale"

        querystring = {"city": self.city,
                       "offset": 0,
                       "state_code": self.state,
                       "limit": "200",
                       "sort": "newest"}

        headers = {
            'x-rapidapi-key': config['DEFAULT']['rapidapi_key'],
            'x-rapidapi-host': config['DEFAULT']['rapidapi_host_re']
        }

        response = requests.request("GET", url, headers=headers, params=querystring)

        if response.status_code == 200:
            json_content = json.loads(response.content)
            housing_total = json_content['data']['total']
            return housing_total, json_content

        # try:  todo fix error catching
        #     response = requests.request("GET", url, headers=headers, params=querystring)
        #
        # except requests.exceptions.RequestException as e:
        #     raise SystemExit(e)
        #
        # if response.status_code == 200:
        #     json_content = json.loads(response.content)
        #
        #     if json_content['status'] == 500:
        #         pass
        #     else:
        #         housing_total = json_content['data']['total']
        #         return housing_total, json_content
        # else:
        #     pass

    @staticmethod
    def define_chunks(total, chunk=int(config['DEFAULT']['rapidapi_offset'])):
        """
        Function to define the offset to collect total number of listings in CITY, STATE from Realtor.com via Rapid API
        :param total: Total number of listings
        :param chunk: Offset to collect on, RAPIDAPI can return up to RAPIDAPI_OFFSET each call
        :return: list of offsets needed to collect entire dataset
        """

        list_chunk_sizes = []
        for x in range(1, math.ceil(total / chunk)):
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
