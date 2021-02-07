import os
import json
import pandas as pd
import requests
import math
import configparser
import logging
import pickle

config = configparser.ConfigParser()
config.read('../config.ini')

logger = logging.getLogger(__name__)


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

        list_missed_states = []
        list_missed_cities = []
        list_missed_offsets = []
        list_collected_data = []

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

            # Check for 200 response from requests module
            if response.status_code == 200:
                json_content = json.loads(response.content)

                # Check for 200 response from RapidAPI server
                if json_content['status'] == 200:
                    list_json_data.append(json_content)

                else:  # Try again if the server didn't return anything todo Error is usually 500: Error JSON parsing
                    response = requests.request("GET", url, headers=headers, params=querystring)

                    json_content = json.loads(response.content)

                    if json_content['status'] == 200:
                        list_json_data.append(json_content)

                    else:
                        logger.error(f'{self.state}-{self.city} failed on offset: {offset}')
                        list_missed_states.append(self.state)
                        list_missed_cities.append(self.city)
                        list_missed_offsets.append(offset)
                        list_collected_data.append(-1)

        dict_missed_data = {'state': list_missed_states, 'city': list_missed_cities, 'offset': list_missed_offsets,
                            'collected': list_collected_data}

        if os.path.exists('../../data/models/missed_data.pickle'):
            with open('../../data/models/missed_data.pickle', 'rb') as file:
                df = pickle.load(file)
                df = df.append(dict_missed_data, ignore_index=True)
        else:
            df = pd.DataFrame(dict_missed_data)

        with open('../../data/models/missed_data.pickle', 'wb') as file:
            pickle.dump(df, file)

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
