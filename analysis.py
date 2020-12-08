import pandas as pd
import numpy as np

from utils.functions import RealEstateData

#######################################################################################################################
# Placeholder for user input
#######################################################################################################################

str_city = 'Durham'
str_state_code = 'NC'

#######################################################################################################################
# Initialize City Data
#######################################################################################################################

# def main(str_city, str_state_code):
# durham = RealEstateData(str_city, str_state_code)


if __name__ == '__main__':
    durham = RealEstateData(str_city, str_state_code)
