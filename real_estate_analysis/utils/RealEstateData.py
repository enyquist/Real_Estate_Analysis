import os
import json
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
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

    def __init__(self, city, state, api):
        self.city = city.upper()
        self.state = state.upper()
        self.api = api
        self.url = config.get(api, 'rapidapi_url')
        self._jsonREData = self.fetch_housing_data()
        self.results = self.parse()
        self.requests_remaining = 99999

    def __repr__(self):
        return f"RealEstateData('{self.city, self.state}')"

    def __str__(self):
        return f'{self.city, self.state} real estate data'

    def fetch_housing_data(self):
        """
        Function to fetch all housing data from Realtor.com via RapidAPI
        :return: Dictionary of Dictionaries containing all the results from the the API call
        """
        list_json_data = None
        list_missed_states = []
        list_missed_cities = []
        list_missed_offsets = []
        list_collected_data = []

        response = self.api_call()

        if self.validate_api_call(response):
            json_content = json.loads(response.content)
            list_json_data = [json_content]
            housing_total = self.get_housing_total(json_content=json_content)
            list_offsets = self.define_chunks(total=housing_total)

            for offset in list_offsets:

                response = self.api_call(offset=offset)

                if self.validate_api_call(response):
                    list_json_data.append(json_content)

                else:  # Try again todo Error is usually 500: Error JSON parsing

                    response = self.api_call(offset=offset)

                    if self.validate_api_call(response):
                        list_json_data.append(json_content)

                    else:
                        logger.error(f'{self.state}-{self.city} failed on offset: {offset}')
                        list_missed_states.append(self.state)
                        list_missed_cities.append(self.city)
                        list_missed_offsets.append(offset)
                        list_collected_data.append(-1)

                        dict_missed_data = {'state': list_missed_states, 'city': list_missed_cities,
                                            'offset': list_missed_offsets, 'collected': list_collected_data}

                        if os.path.exists('../../data/models/missed_data.pickle'):
                            with open('../../data/models/missed_data.pickle', 'rb') as file:
                                df = pickle.load(file)
                                df = df.append(dict_missed_data, ignore_index=True)
                        else:
                            df = pd.DataFrame(dict_missed_data)

                        with open('../../data/models/missed_data.pickle', 'wb') as file:
                            pickle.dump(df, file)

        return list_json_data

    def api_call(self, offset=0):
        """
        Function to conduct an API call and return the response
        :param offset:
        :return:
        """
        querystring = {"city": self.city,
                       "offset": offset,
                       "state_code": self.state,
                       "limit": "200",
                       "sort": config.get(self.api, 'rapidapi_sort_method')}

        headers = {
            'x-rapidapi-key': config.get(self.api, 'rapidapi_key'),
            'x-rapidapi-host': config.get(self.api, 'rapidapi_host')
        }

        s = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        s.mount('https://', HTTPAdapter(max_retries=retries))

        response = s.get(self.url, headers=headers, params=querystring)

        config.set(self.api, 'rapidapi_api_call_limit', response.headers['X-RateLimit-Requests-Remaining'])

        with open('../config.ini', 'w') as configfile:
            config.write(configfile)

        self.requests_remaining = int(response.headers['X-RateLimit-Requests-Remaining'])

        return response

    def validate_api_call(self, response):
        """
        Checks that the 'status' code in the JSON content is 200, indicating that RapidAPI returned data
        :param response: RapidAPI response object
        :return: True if response contained data, False otherwise
        """
        if self.api == 'RAPIDAPI_SALE':
            json_content = json.loads(response.content)

            # Check for 200 response from RapidAPI server
            if json_content['status'] == 200 and json_content['data']['total'] is not None:
                return True

        if self.api == 'RAPIDAPI_SOLD':
            json_content = json.loads(response.content)

            # Check for content
            if json_content['returned_rows'] > 0:
                return True

        return False

    def get_housing_total(self, json_content):
        """

        :param json_content:
        :return:
        """
        data = config.get(self.api, 'rapidapi_json_lvl1')
        total = config.get(self.api, 'rapidapi_json_lvl2')
        return json_content[data][total]

    def define_chunks(self, total):
        """
        Function to define the offset to collect total number of listings in CITY, STATE from Realtor.com via Rapid API
        :param total: Total number of listings
        :return: list of offsets needed to collect entire dataset
        """

        chunk = int(config.get(self.api, 'rapidapi_offset'))

        list_chunk_sizes = []
        for x in range(1, math.ceil(total / chunk)):
            list_chunk_sizes.append(chunk * x)

        return list_chunk_sizes

    def parse(self):
        """
        Function to format the entire dataset as a DataFrame
        :return: DataFrame built from total dataset
        """

        df_results = None
        list_results_dfs = None

        if self._jsonREData is not None:

            if self.api == 'RAPIDAPI_SALE':

                list_results_dfs = [pd.json_normalize(result['data']['results']) for result in self._jsonREData]

            if self.api == 'RAPIDAPI_SOLD':
                list_results_dfs = [pd.json_normalize(result['properties']) for result in self._jsonREData]

            df_results = pd.concat(list_results_dfs, ignore_index=True)

        return df_results
