import copy

import numpy as np
import pandas as pd

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

        self.N_List = [self.N1, self.N2, self.N3, self.N4]

        # !!!!!!
        # make sure that name of the keys are same with the name of the attributes in Solution_Object
        # !!!!!!
        dict_keys = ['I_KOMP1_HIZ', 'II_KOMP1_HIZ', 'III_KOMP1_HIZ', 'IV_KOMP1_HIZ',
                     'KS10_UDP_TUKETIM',
                     'CH1_CIKIS_SIC', 'CH1_GIRIS_SIC',
                     'CH2_CIKIS_SIC', 'CH2_GIRIS_SIC',
                     'CH3_CIKIS_SIC', 'CH3_GIRIS_SIC',
                     'SENSOR_I_TEMP', 'SENSOR_II_TEMP',
                     'OUTLET_TEMP', 'OUTLET_HUMIDITY',
                     'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat',
                     'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4',
                     'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9',
                     'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14',
                     'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19',
                     'hour_20', 'hour_21', 'hour_22', 'hour_23',
                     'month_4', 'month_5', 'month_6', 'month_7',
                     'month_8', 'month_9', 'month_10',
                     'PUE_lag_one_day']

        self.PUE_DICT = dict.fromkeys(dict_keys, 0)

        # remove SENSOR_I_TEMP and SENSOR_II_TEMP from dict_keys
        sensor_dict_keys = dict_keys.copy()
        sensor_dict_keys.remove('SENSOR_I_TEMP')
        sensor_dict_keys.remove('SENSOR_II_TEMP')
        self.SENSOR_I_DICT = dict.fromkeys(sensor_dict_keys, 0)
        self.SENSOR_II_DICT = dict.fromkeys(sensor_dict_keys, 0)

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
            total_objective += objective_timestamp

        objective = total_objective / self.no_timestamps

        return objective

    def objective_timestamp(self: 'Solution_Rep',
                            s_timestamp: Solution_Object) -> float:

        PUE, sensor_i_temp, sensor_ii_temp = self.predict_PUE(s_timestamp)

        # if sensor_i_temp or sensor_ii_temp is +-4 than set temperatures, give a high penalty
        sic_set_penalty = 0
        if sensor_i_temp - s_timestamp.SIC_I_SET > 4:
            sic_set_penalty += 10 ** 8
        if s_timestamp.SIC_I_SET - sensor_i_temp > 4:
            sic_set_penalty += 10 ** 4
        if sensor_ii_temp - s_timestamp.SIC_II_SET > 4:
            sic_set_penalty += 10 ** 8
        if s_timestamp.SIC_II_SET - sensor_ii_temp > 4:
            sic_set_penalty += 10 ** 4

        # if I_KOMP1_HIZ or II_KOMP1_HIZ or III_KOMP1_HIZ or IV_KOMP1_HIZ is different from %50
        # initial values, give a penalty
        speed_penalty = 0
        roman_numbers = ['I', 'II', 'III', 'IV']
        for i in range(4):
            init_speed = s_timestamp.__dict__[f'init_{roman_numbers[i]}_KOMP1_HIZ']
            speed = s_timestamp.__dict__[f'{roman_numbers[i]}_KOMP1_HIZ']

            if init_speed != 0:
                if abs(init_speed - speed) / init_speed > 0.5:
                    speed_penalty += 10 ** 2

        # if PUE is different from 1, give a penalty as much as difference
        # the more PUE is different than 1, the more penalty with exponential
        pue_penalty = 0
        diff_PUE = PUE - 1
        pue_penalty += 10 ** 3 * (diff_PUE / 0.01)

        objective_timestamp = pue_penalty + sic_set_penalty + speed_penalty

        return objective_timestamp

    def predict_PUE(self: 'Solution_Rep',
                    s_timestamp: Solution_Object) -> (float, float, float):

        sensor_i_temp = self.predict_SENSOR_I_TEMP(s_timestamp)
        sensor_ii_temp = self.predict_SENSOR_II_TEMP(s_timestamp)

        PUE_SAMPLE_DICT = self.dict_to_predict(s_timestamp, self.PUE_DICT)
        PUE_SAMPLE_DICT['SENSOR_I_TEMP'] = sensor_i_temp
        PUE_SAMPLE_DICT['SENSOR_II_TEMP'] = sensor_ii_temp

        PUE = np.exp(self.PUE_MODEL.predict(PUE_SAMPLE_DICT)[0]) + 1

        return PUE, sensor_i_temp, sensor_ii_temp

    def predict_SENSOR_I_TEMP(self: 'Solution_Rep',
                              s_timestamp: Solution_Object) -> float:

        SENSOR_I_SAMPLE_DICT = self.dict_to_predict(s_timestamp, self.SENSOR_I_DICT)

        SENSOR_I_TEMP = np.exp(self.SENSOR_I_MODEL.predict(SENSOR_I_SAMPLE_DICT)[0])

        return SENSOR_I_TEMP

    def predict_SENSOR_II_TEMP(self: 'Solution_Rep',
                               s_timestamp: Solution_Object) -> float:

        SENSOR_II_SAMPLE_DICT = self.dict_to_predict(s_timestamp, self.SENSOR_II_DICT)

        SENSOR_II_TEMP = np.exp(self.SENSOR_II_MODEL.predict(SENSOR_II_SAMPLE_DICT)[0])

        return SENSOR_II_TEMP

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
            print('Timestamp: ' + str(s_timestamp.timestamp))
            print('I_KOMP1_HIZ: ' + str(s_timestamp.I_KOMP1_HIZ))
            print('II_KOMP1_HIZ: ' + str(s_timestamp.II_KOMP1_HIZ))
            print('III_KOMP1_HIZ: ' + str(s_timestamp.III_KOMP1_HIZ))
            print('IV_KOMP1_HIZ: ' + str(s_timestamp.IV_KOMP1_HIZ))
            print('PUE: ' + str(s_timestamp.PUE))

            real_pue = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp']
                                      == s_timestamp.timestamp]['PUE'].values[0]

            print('Real PUE: ' + str(real_pue))
            print('SENSOR_I_TEMP: ' + str(s_timestamp.SENSOR_I_TEMP) +
                  ' ' + 'SIC_I_SET: ' + str(s_timestamp.SIC_I_SET))
            print('SENSOR_II_TEMP: ' + str(s_timestamp.SENSOR_II_TEMP) +
                  ' ' + 'SIC_II_SET: ' + str(s_timestamp.SIC_II_SET))

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

        :param speed_list: list of tuples of speeds
        :return: list of tuples of speeds
        :return: move information
        """

        pass

    def N6(self: 'Solution_Rep',
           speed_list: list) -> (list, tuple):
        """

        :param speed_list: list of tuples of speeds
        :return: list of tuples of speeds
        :return: move information
        """

        pass
