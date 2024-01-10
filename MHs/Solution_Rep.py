import copy
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels
import sklearn

from InputHandler import InputHandler
from Solution_Object import Solution_Object


class Solution_Rep:

    def __init__(self: 'Solution_Rep',
                 input_h: 'InputHandler') -> None:

        self.PUE_MODEL = InputHandler.get_PUE_MODEL()
        self.SENSOR_I_MODEL = InputHandler.get_SENSOR_I_MODEL()
        self.SENSOR_II_MODEL = InputHandler.get_SENSOR_II_MODEL()

        self.init_solutions = input_h.solutions
        self.no_timestamps = int(input_h.no_timestamps)

        self.N_List = [self.N1, self.N2, self.N3, self.N4, self.N5, self.N6, self.N7]

        # !!!!!!
        # make sure that name of the keys are same with the name of the attributes in Solution_Object
        # !!!!!!
        dict_keys = ['I_KOMP1_HIZ', 'II_KOMP1_HIZ', 'III_KOMP1_HIZ', 'IV_KOMP1_HIZ',
                     'CH1_CIKIS_SIC', 'CH1_GIRIS_SIC',
                     'CH2_CIKIS_SIC', 'CH2_GIRIS_SIC',
                     'CH3_CIKIS_SIC', 'CH3_GIRIS_SIC',
                     'SENSOR_I_TEMP', 'SENSOR_II_TEMP',
                     'OUTLET_TEMP', 'OUTLET_HUMIDITY', 'KS10_UDP_TUKETIM',
                     'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat',
                     'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4',
                     'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9',
                     'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14',
                     'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19',
                     'hour_20', 'hour_21', 'hour_22', 'hour_23',
                     'month_4', 'month_5', 'month_6', 'month_7',
                     'month_8', 'month_9', 'month_10',
                     'PUE_lag_one_day']

        self.PUE_DICT_OLS = dict.fromkeys(dict_keys, 0)

        # remove SENSOR_I_TEMP and SENSOR_II_TEMP from dict_keys
        sensor_dict_keys = dict_keys.copy()[:-38]
        sensor_dict_keys.remove('SENSOR_I_TEMP')
        sensor_dict_keys.remove('SENSOR_II_TEMP')
        self.SENSOR_I_DICT_OLS = dict.fromkeys(sensor_dict_keys, 0)
        self.SENSOR_II_DICT_OLS = dict.fromkeys(sensor_dict_keys, 0)

    @staticmethod
    def dict_to_predict(s_timestamp: Solution_Object,
                        dictionary: dict) -> dict:

        copy_dict = dictionary.copy()

        s_timestamp_dict = s_timestamp.__dict__

        for key in copy_dict.keys():
            if key in s_timestamp_dict.keys():
                copy_dict[key] = s_timestamp_dict[key]

        dummies_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat',
                        'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4',
                        'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9',
                        'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14',
                        'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19',
                        'hour_20', 'hour_21', 'hour_22', 'hour_23',
                        'month_4', 'month_5', 'month_6', 'month_7',
                        'month_8', 'month_9', 'month_10',
                        'PUE_lag_one_day']

        for i in range(len(dummies_name)):
            copy_dict[dummies_name[i]] = s_timestamp_dict['DUMMIES'][i]

        # copy_dict = list(copy_dict.values())

        return copy_dict

    def init(self: 'Solution_Rep') -> list:

        speed_list = []

        for i in range(len(self.init_solutions)):
            timestamp_speed = (self.init_solutions[i].init_I_KOMP1_HIZ,
                               self.init_solutions[i].init_II_KOMP1_HIZ,
                               self.init_solutions[i].init_III_KOMP1_HIZ,
                               self.init_solutions[i].init_IV_KOMP1_HIZ)

            speed_list.append(timestamp_speed)

        return speed_list

    def objective(self: 'Solution_Rep',
                  speed_list: list) -> float:

        roman_numbers = ['I', 'II', 'III', 'IV']

        # create a copy of init_solutions
        solutions = copy.deepcopy(self.init_solutions)

        # set speed values and calculate objective
        total_objective = 0
        for i in range(self.no_timestamps):
            s_timestamp = solutions[i]
            s_timestamp.I_KOMP1_HIZ = speed_list[i][0]
            s_timestamp.II_KOMP1_HIZ = speed_list[i][1]
            s_timestamp.III_KOMP1_HIZ = speed_list[i][2]
            s_timestamp.IV_KOMP1_HIZ = speed_list[i][3]

            objective_timestamp = self.objective_timestamp(s_timestamp)

            # if i == 0, every komp1_hiz it should be in range of s_timestamp.init_..._KOMP1_HIZ * 0.85 and
            # s_timestamp.init_..._KOMP1_HIZ * 1.15
            if i == 0:
                for j in range(4):
                    init_speed = s_timestamp.__dict__[f'init_{roman_numbers[j]}_KOMP1_HIZ']
                    speed = s_timestamp.__dict__[f'{roman_numbers[j]}_KOMP1_HIZ']

                    if speed <= 25.1:
                        objective_timestamp += 10 ** 8

                    if speed >= 74.9:
                        objective_timestamp += 10 ** 8

                    '''if init_speed != 0:
                        speed_change_percentage = abs(init_speed - speed) / init_speed
                        if speed_change_percentage > 0.15:
                            objective_timestamp += (10 ** 5) * speed_change_percentage'''

            # else every komp1_hiz it should be im range of s_timestamp_one_before.init_..._KOMP1_HIZ * 0.85 and
            # s_timestamp_one_before.init_..._KOMP1_HIZ * 1.15
            else:
                for j in range(4):
                    s_timestamp_one_before = solutions[i - 1]

                    speed_one_before = s_timestamp_one_before.__dict__[f'{roman_numbers[j]}_KOMP1_HIZ']
                    speed = s_timestamp.__dict__[f'{roman_numbers[j]}_KOMP1_HIZ']

                    if speed_one_before != 0:
                        speed_change_percentage = abs(speed_one_before - speed) / speed_one_before
                        if speed_change_percentage > 0.25:
                            objective_timestamp += (10 ** 1) * speed_change_percentage

            total_objective += objective_timestamp * (self.no_timestamps - i) / 10

        objective = total_objective / sum(range(self.no_timestamps + 1))

        return objective

    @lru_cache(maxsize=128)
    def objective_timestamp(self: 'Solution_Rep',
                            s_timestamp: Solution_Object) -> float:

        PUE, sensor_i_temp, sensor_ii_temp = self.predict_PUE(s_timestamp)

        # if sensor_i_temp or sensor_ii_temp is +-4 than set temperatures, give a high penalty
        sic_set_penalty = 0
        if sensor_i_temp - s_timestamp.SIC_I_SET > 4:
            sic_set_penalty *= 10 ** 8
        if s_timestamp.SIC_I_SET - sensor_i_temp > 4:
            sic_set_penalty += 10 ** 1
        if sensor_ii_temp - s_timestamp.SIC_II_SET > 4:
            sic_set_penalty += 10 ** 8
        if s_timestamp.SIC_II_SET - sensor_ii_temp > 4:
            sic_set_penalty += 10 ** 1

        # if PUE is different from 1, give a penalty as much as difference
        # the more PUE is different than 1, the more penalty with exponential
        pue_penalty = 0
        diff_PUE = PUE - 1
        pue_penalty += 10 ** 4 * (diff_PUE / 0.1)

        objective_timestamp = pue_penalty + sic_set_penalty

        return objective_timestamp

    def predict_PUE(self: 'Solution_Rep',
                    s_timestamp: Solution_Object) -> (float, float, float):

        sensor_i_temp = self.predict_SENSOR_I_TEMP(s_timestamp)
        sensor_ii_temp = self.predict_SENSOR_II_TEMP(s_timestamp)

        PUE_SAMPLE_DICT = self.dict_to_predict(s_timestamp, self.PUE_DICT_OLS)
        PUE_SAMPLE_DICT['SENSOR_I_TEMP'] = sensor_i_temp
        PUE_SAMPLE_DICT['SENSOR_II_TEMP'] = sensor_ii_temp

        PUE = 2
        if type(self.PUE_MODEL) == statsmodels.regression.linear_model.RegressionResultsWrapper:
            PUE = np.exp(self.PUE_MODEL.predict(PUE_SAMPLE_DICT)[0]) + 1
        elif type(self.PUE_MODEL) == sklearn.ensemble._forest.RandomForestRegressor:
            PUE = np.exp(self.PUE_MODEL.predict([list(PUE_SAMPLE_DICT.values())])[0]) + 1
        elif type(self.PUE_MODEL) == sklearn.ensemble._gb.GradientBoostingRegressor:
            PUE = np.exp(self.PUE_MODEL.predict([list(PUE_SAMPLE_DICT.values())])[0]) + 1
        else:
            raise Exception('PUE_MODEL is not a valid model!')

        return PUE, sensor_i_temp, sensor_ii_temp

    def predict_SENSOR_I_TEMP(self: 'Solution_Rep',
                              s_timestamp: Solution_Object) -> float:

        SENSOR_I_SAMPLE_DICT = self.dict_to_predict(s_timestamp, self.SENSOR_I_DICT_OLS)

        SENSOR_I_TEMP = np.exp(self.SENSOR_I_MODEL.predict(SENSOR_I_SAMPLE_DICT)[0])

        return SENSOR_I_TEMP

    def predict_SENSOR_II_TEMP(self: 'Solution_Rep',
                               s_timestamp: Solution_Object) -> float:

        SENSOR_II_SAMPLE_DICT = self.dict_to_predict(s_timestamp, self.SENSOR_II_DICT_OLS)

        SENSOR_II_TEMP = np.exp(self.SENSOR_II_MODEL.predict(SENSOR_II_SAMPLE_DICT)[0])

        return SENSOR_II_TEMP

    def N1(self: 'Solution_Rep',
           speed_list: list) -> (list, tuple):
        """
        Set all speeds in each timestamp to average of them.
        :param speed_list: list of tuples of speeds
        :return: list of tuples of speeds
        :return: move information
        """

        for i in range(self.no_timestamps):
            average_speed = sum(speed_list[i]) / 4
            speed_list[i] = (average_speed, average_speed, average_speed, average_speed)

        return speed_list, (-1, -1)

    def N2(self: 'Solution_Rep',
           speed_list: list) -> (list, tuple):
        """
        Choose a random timestamp and set all speeds in that timestamp to average of them.
        :param speed_list: list of tuples of speeds
        :return: list of tuples of speeds
        :return: move information
        """

        random_timestamp_index = np.random.randint(self.no_timestamps)

        average_speed = sum(speed_list[random_timestamp_index]) / 4
        speed_list[random_timestamp_index] = (average_speed, average_speed, average_speed, average_speed)

        return speed_list, (random_timestamp_index, -1)

    def N3(self: 'Solution_Rep',
           speed_list: list) -> (list, tuple):
        """
        Choose a random timestamp and select a speed in that timestamp and increase it by 10%.
        :param speed_list: list of tuples of speeds
        :return: list of tuples of speeds
        :return: move information
        """

        random_timestamp_index = np.random.randint(self.no_timestamps)
        random_speed_index = np.random.randint(4)

        chosen_speed = speed_list[random_timestamp_index][random_speed_index]

        new_speed = chosen_speed * 1.1

        new_timestamp = []
        for i in range(4):
            if i == random_speed_index:
                new_timestamp.append(new_speed)
            else:
                new_timestamp.append(speed_list[random_timestamp_index][i])

        speed_list[random_timestamp_index] = tuple(new_timestamp)

        return speed_list, (random_timestamp_index, random_speed_index)

    def N4(self: 'Solution_Rep',
           speed_list: list) -> (list, tuple):
        """
        Choose a random timestamp and select a speed in that timestamp and decrease it by 10%.
        :param speed_list: list of tuples of speeds
        :return: list of tuples of speeds
        :return: move information
        """

        random_timestamp_index = np.random.randint(self.no_timestamps)
        random_speed_index = np.random.randint(4)

        chosen_speed = speed_list[random_timestamp_index][random_speed_index]

        new_speed = chosen_speed * 0.9

        new_timestamp = []
        for i in range(4):
            if i == random_speed_index:
                new_timestamp.append(new_speed)
            else:
                new_timestamp.append(speed_list[random_timestamp_index][i])

        speed_list[random_timestamp_index] = tuple(new_timestamp)

        return speed_list, (random_timestamp_index, random_speed_index)

    def N5(self: 'Solution_Rep',
           speed_list: list) -> (list, tuple):
        """
        Choose a random timestamp and decrease all speeds in that timestamp by [5, 10, 15]%.
        :param speed_list: list of tuples of speeds
        :return: list of tuples of speeds
        :return: move information
        """

        random_timestamp_index = np.random.randint(self.no_timestamps)
        random_percentage = np.random.randint(1, 4) * 5

        new_timestamp = []
        for i in range(4):
            new_speed = speed_list[random_timestamp_index][i] * (1 - random_percentage / 100)
            new_timestamp.append(new_speed)

        speed_list[random_timestamp_index] = tuple(new_timestamp)

        return speed_list, (random_timestamp_index, random_percentage)

    def N6(self: 'Solution_Rep',
           speed_list: list) -> (list, tuple):
        """
        Choose a random timestamp and decrease all speeds in that timestamp by [5, 10, 15]%.
        :param speed_list: list of tuples of speeds
        :return: list of tuples of speeds
        :return: move information
        """

        random_timestamp_index = np.random.randint(self.no_timestamps)
        random_percentage = np.random.randint(1, 4) * 5

        new_timestamp = []
        for i in range(4):
            new_speed = speed_list[random_timestamp_index][i] * (1 + random_percentage / 100)
            new_timestamp.append(new_speed)

        speed_list[random_timestamp_index] = tuple(new_timestamp)

        return speed_list, (random_timestamp_index, random_percentage)

    def N7(self: 'Solution_Rep',
           speed_list: list) -> (list, tuple):
        """
        Sets all timestamps to the their initial speeds.
        :param speed_list:
        :return:
        """
        for i in range(self.no_timestamps):
            s_timestamp = self.init_solutions[i]
            speed_list[i] = (s_timestamp.init_I_KOMP1_HIZ,
                             s_timestamp.init_II_KOMP1_HIZ,
                             s_timestamp.init_III_KOMP1_HIZ,
                             s_timestamp.init_IV_KOMP1_HIZ)

        return speed_list, (-1, -1)

    def excel_write(self: 'Solution_Rep',
                    speed_list: list,
                    output_path: str,
                    KS10_REAL_DATA: pd.DataFrame) -> None:

        # create a copy of init_solutions
        solutions = copy.deepcopy(self.init_solutions)

        # set speed values for each timestamp
        for i in range(self.no_timestamps):
            s_timestamp = solutions[i]
            s_timestamp.I_KOMP1_HIZ = speed_list[i][0]
            s_timestamp.II_KOMP1_HIZ = speed_list[i][1]
            s_timestamp.III_KOMP1_HIZ = speed_list[i][2]
            s_timestamp.IV_KOMP1_HIZ = speed_list[i][3]

            # set PUE, SENSOR_I_TEMP and SENSOR_II_TEMP values for each timestamp
            PUE, sensor_i_temp, sensor_ii_temp = self.predict_PUE(s_timestamp)
            s_timestamp.PUE = PUE
            s_timestamp.SENSOR_I_TEMP = sensor_i_temp
            s_timestamp.SENSOR_II_TEMP = sensor_ii_temp

        for i in range(self.no_timestamps):
            s_timestamp = solutions[i]

            real_pue = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] == s_timestamp.timestamp]['PUE'].values[0]

            print('Timestamp: ' + str(s_timestamp.timestamp))
            # print all values with 4 decimal points
            print('I_KOMP1_HIZ: ' + str(format(s_timestamp.I_KOMP1_HIZ, '.4f')) +
                  ' ----- OLD: ' + str(format(s_timestamp.init_I_KOMP1_HIZ, '.4f')))
            print('II_KOMP1_HIZ: ' + str(format(s_timestamp.II_KOMP1_HIZ, '.4f')) +
                  ' ----- OLD: ' + str(format(s_timestamp.init_II_KOMP1_HIZ, '.4f')))
            print('III_KOMP1_HIZ: ' + str(format(s_timestamp.III_KOMP1_HIZ, '.4f')) +
                  ' ----- OLD: ' + str(format(s_timestamp.init_III_KOMP1_HIZ, '.4f')))
            print('IV_KOMP1_HIZ: ' + str(format(s_timestamp.IV_KOMP1_HIZ, '.4f')) +
                  ' ----- OLD: ' + str(format(s_timestamp.init_IV_KOMP1_HIZ, '.4f')))
            print('PUE: ' + str(format(s_timestamp.PUE, '.4f')) +
                  ' ----- Real PUE: ' + str(format(real_pue, '.4f')))

            print('SENSOR_I_TEMP: ' + str(format(s_timestamp.SENSOR_I_TEMP, '.4f')) +
                  ' ----- ' + 'SIC_I_SET: ' + str(format(s_timestamp.SIC_I_SET, '.4f')))
            print('SENSOR_II_TEMP: ' + str(format(s_timestamp.SENSOR_II_TEMP, '.4f')) +
                  ' ----- ' + 'SIC_II_SET: ' + str(format(s_timestamp.SIC_II_SET, '.4f')))
            print()

        # print PUEs
        PUEs = []
        for i in range(self.no_timestamps):
            PUEs.append(solutions[i].PUE)

        print('PUEs: ' + str(PUEs))

        # create a dataframe to write excel file
        OUTPUT_DATA = pd.read_csv(output_path)
        OUTPUT_DATA['Timestamp'] = pd.to_datetime(OUTPUT_DATA['Timestamp'])

        # OUTPUT_DATA columns:
        #       ['Timestamp', 'PUE', 'I_KOMP1_HIZ', 'II_KOMP1_HIZ', 'III_KOMP1_HIZ',
        #        'IV_KOMP1_HIZ', 'CH1_CIKIS_SIC', 'CH1_GIRIS_SIC', 'CH2_CIKIS_SIC',
        #        'CH2_GIRIS_SIC', 'CH3_CIKIS_SIC', 'CH3_GIRIS_SIC', 'SENSOR_I_TEMP',
        #        'SENSOR_II_TEMP', 'OUTLET_TEMP', 'OUTLET_HUMIDITY', 'KS10_UDP_TUKETIM',
        #        'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'hour_0', 'hour_1', 'hour_2',
        #        'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9',
        #        'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15',
        #        'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21',
        #        'hour_22', 'hour_23',
        #        'month_4', 'month_5', 'month_6', 'month_7',
        #        'month_8', 'month_9', 'month_10', 'month_11',
        #        'KS10_REAL_CH1_CIKIS_SIC', 'KS10_REAL_CH1_GIRIS_SIC',
        #        'KS10_REAL_CH2_CIKIS_SIC', 'KS10_REAL_CH2_GIRIS_SIC',
        #        'KS10_REAL_CH3_CIKIS_SIC', 'KS10_REAL_CH3_GIRIS_SIC',
        #        'KS10_REAL_OUTLET_TEMP', 'KS10_REAL_OUTLET_HUMIDITY',
        #        'KS10_REAL_KS10_UDP_TUKETIM', 'solved_for_first_timestamp']

        # insert new timestamps to OUTPUT_DATA
        for i in range(self.no_timestamps):
            s_timestamp = solutions[i]

            if s_timestamp.timestamp not in OUTPUT_DATA['Timestamp'].values:
                OUTPUT_DATA.loc[len(OUTPUT_DATA)] = [s_timestamp.timestamp] + [None] * len(OUTPUT_DATA.columns[1:])

            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'PUE'] = s_timestamp.PUE
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'I_KOMP1_HIZ'] = s_timestamp.I_KOMP1_HIZ
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'II_KOMP1_HIZ'] = s_timestamp.II_KOMP1_HIZ
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'III_KOMP1_HIZ'] = s_timestamp.III_KOMP1_HIZ
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'IV_KOMP1_HIZ'] = s_timestamp.IV_KOMP1_HIZ
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'CH1_CIKIS_SIC'] = s_timestamp.CH1_CIKIS_SIC
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'CH1_GIRIS_SIC'] = s_timestamp.CH1_GIRIS_SIC
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'CH2_CIKIS_SIC'] = s_timestamp.CH2_CIKIS_SIC
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'CH2_GIRIS_SIC'] = s_timestamp.CH2_GIRIS_SIC
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'CH3_CIKIS_SIC'] = s_timestamp.CH3_CIKIS_SIC
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'CH3_GIRIS_SIC'] = s_timestamp.CH3_GIRIS_SIC
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'SENSOR_I_TEMP'] = s_timestamp.SENSOR_I_TEMP
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'SENSOR_II_TEMP'] = s_timestamp.SENSOR_II_TEMP
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'OUTLET_TEMP'] = s_timestamp.OUTLET_TEMP
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'OUTLET_HUMIDITY'] = s_timestamp.OUTLET_HUMIDITY
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'KS10_UDP_TUKETIM'] = s_timestamp.KS10_UDP_TUKETIM
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'Mon'] = s_timestamp.DUMMIES[0]
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'Tue'] = s_timestamp.DUMMIES[1]
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'Wed'] = s_timestamp.DUMMIES[2]
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'Thu'] = s_timestamp.DUMMIES[3]
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'Fri'] = s_timestamp.DUMMIES[4]
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'Sat'] = s_timestamp.DUMMIES[5]
            for h in range(24):
                OUTPUT_DATA.loc[
                    OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, f'hour_{h}'] = s_timestamp.DUMMIES[6 + h]
            for m in range(4, 12):
                OUTPUT_DATA.loc[
                    OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, f'month_{m}'] = s_timestamp.DUMMIES[26 + m]

            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'KS10_REAL_PUE'] = \
                KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] == s_timestamp.timestamp]['PUE'].values[0]
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'KS10_REAL_CH1_CIKIS_SIC'] = \
                KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] == s_timestamp.timestamp]['CH1_CIKIS_SIC'].values[0]
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'KS10_REAL_CH1_GIRIS_SIC'] = \
                KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] == s_timestamp.timestamp]['CH1_GIRIS_SIC'].values[0]
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'KS10_REAL_CH2_CIKIS_SIC'] = \
                KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] == s_timestamp.timestamp]['CH2_CIKIS_SIC'].values[0]
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'KS10_REAL_CH2_GIRIS_SIC'] = \
                KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] == s_timestamp.timestamp]['CH2_GIRIS_SIC'].values[0]
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'KS10_REAL_CH3_CIKIS_SIC'] = \
                KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] == s_timestamp.timestamp]['CH3_CIKIS_SIC'].values[0]
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'KS10_REAL_CH3_GIRIS_SIC'] = \
                KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] == s_timestamp.timestamp]['CH3_GIRIS_SIC'].values[0]
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'KS10_REAL_OUTLET_TEMP'] = \
                KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] == s_timestamp.timestamp]['OUTLET_TEMP'].values[0]
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'KS10_REAL_OUTLET_HUMIDITY'] = \
                KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] == s_timestamp.timestamp]['OUTLET_HUMIDITY'].values[0]
            OUTPUT_DATA.loc[
                OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'KS10_REAL_KS10_UDP_TUKETIM'] = \
                KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] == s_timestamp.timestamp]['KS10_UDP_TUKETIM'].values[0]

            if i == 0:
                OUTPUT_DATA.loc[
                    OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'solved_for_first_timestamp'] = 1
            else:
                OUTPUT_DATA.loc[
                    OUTPUT_DATA['Timestamp'] == s_timestamp.timestamp, 'solved_for_first_timestamp'] = 0

        OUTPUT_DATA.to_csv(output_path, index=False)
