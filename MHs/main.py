import pandas as pd

import algorithms as alg
from InputHandler import InputHandler
from Solution_Rep import Solution_Rep

import os

REAL_DATA_PATH = '../data/KS VERI/KS10_FINAL_DATA.csv'
OUTPUT_DATA_PATH = '../data/KS VERI/KS10_MH_OUTPUT_DATA_SA.csv'

PICKLES_PATH = 'pickles/'

PUE_MODEL_PATH = PICKLES_PATH + 'PUE_OLS.pkl'
SENSOR_I_MODEL_PATH = PICKLES_PATH + 'SENSOR_I_OLS.pkl'
SENSOR_II_MODEL_PATH = PICKLES_PATH + 'SENSOR_II_OLS.pkl'

CH1_CIKIS_SIC_MODEL_PATH = PICKLES_PATH + 'ARIMA_CH1_CIKIS_SIC_MODEL.pkl'
CH1_GIRIS_SIC_MODEL_PATH = PICKLES_PATH + 'ARIMA_CH1_GIRIS_SIC_MODEL.pkl'
CH2_CIKIS_SIC_MODEL_PATH = PICKLES_PATH + 'ARIMA_CH2_CIKIS_SIC_MODEL.pkl'
CH2_GIRIS_SIC_MODEL_PATH = PICKLES_PATH + 'ARIMA_CH2_GIRIS_SIC_MODEL.pkl'
CH3_CIKIS_SIC_MODEL_PATH = PICKLES_PATH + 'ARIMA_CH3_CIKIS_SIC_MODEL.pkl'
CH3_GIRIS_SIC_MODEL_PATH = PICKLES_PATH + 'ARIMA_CH3_GIRIS_SIC_MODEL.pkl'

KS10_UDP_TUKETIM_MODEL_PATH = PICKLES_PATH + 'ARIMA_KS10_UDP_TUKETIM_MODEL.pkl'

START_TIMESTAMP = '2023-10-18 08:00:00'
NO_TIMESTAMPS = 10

# create KS10_REAL_DATA to start
# The first timestamp of KS10_REAL_DATA is 2023-04-01 00:10:00
# The last timestamp of KS10_REAL_DATA is 2023-10-25 01:05:00
KS10_REAL_DATA = pd.read_csv(REAL_DATA_PATH)
KS10_REAL_DATA['Timestamp'] = pd.to_datetime(KS10_REAL_DATA['Timestamp'])

# create an empty KS10_OUTPUT_DATA
# KS10_REAL_DATA columns:
#       ['Timestamp', 'PUE', 'I_KOMP1_HIZ', 'I_KOMP1_SAAT', 'I_KOMP2_HIZ',
#        'I_KOMP2_SAAT', 'I_NEM_SET', 'I_SIC_SET', 'II_KOMP1_HIZ',
#        'II_KOMP1_SAAT', 'II_KOMP2_HIZ', 'II_KOMP2_SAAT', 'II_NEM_SET',
#        'II_SIC_SET', 'III_KOMP1_HIZ', 'III_KOMP1_SAAT', 'III_KOMP2_HIZ',
#        'III_KOMP2_SAAT', 'III_NEM_SET', 'III_SIC_SET', 'IV_KOMP1_HIZ',
#        'IV_KOMP1_SAAT', 'IV_KOMP2_HIZ', 'IV_KOMP2_SAAT', 'IV_NEM_SET',
#        'IV_SIC_SET', 'CH1_CIKIS_SIC', 'CH1_GIRIS_SIC', 'CH2_CIKIS_SIC',
#        'CH2_GIRIS_SIC', 'CH3_CIKIS_SIC', 'CH3_GIRIS_SIC', 'SENSOR_I_TEMP',
#        'SENSOR_II_TEMP', 'OUTLET_TEMP', 'OUTLET_HUMIDITY', 'KS10_UDP_TUKETIM',
#        'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'hour_0', 'hour_1', 'hour_2',
#        'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9',
#        'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15',
#        'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21',
#        'hour_22', 'hour_23', 'month_1', 'month_2', 'month_3', 'month_4',
#        'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10',
#        'month_11']
# drop I_KOMP1_SAAT, I_KOMP2_SAAT, II_KOMP1_SAAT, II_KOMP2_SAAT,
# III_KOMP1_SAAT, III_KOMP2_SAAT, IV_KOMP1_SAAT, IV_KOMP2_SAAT
# drop I_KOMP2_HIZ, II_KOMP2_HIZ, III_KOMP2_HIZ, IV_KOMP2_HIZ
# drop I_NEM_SET, II_NEM_SET, III_NEM_SET, IV_NEM_SET
# drop I_SIC_SET, II_SIC_SET, III_SIC_SET, IV_SIC_SET
KS10_REAL_DATA = KS10_REAL_DATA.drop(columns=['I_KOMP1_SAAT', 'I_KOMP2_SAAT',
                                              'II_KOMP1_SAAT', 'II_KOMP2_SAAT',
                                              'III_KOMP1_SAAT', 'III_KOMP2_SAAT',
                                              'IV_KOMP1_SAAT', 'IV_KOMP2_SAAT',
                                              'I_KOMP2_HIZ', 'II_KOMP2_HIZ',
                                              'III_KOMP2_HIZ', 'IV_KOMP2_HIZ',
                                              'I_NEM_SET', 'II_NEM_SET',
                                              'III_NEM_SET', 'IV_NEM_SET',
                                              'I_SIC_SET', 'II_SIC_SET',
                                              'III_SIC_SET', 'IV_SIC_SET',
                                              'month_1', 'month_2', 'month_3'])

KS10_OUTPUT_DATA = pd.DataFrame(columns=list(KS10_REAL_DATA.columns) + ['KS10_REAL_PUE',
                                                                        'KS10_REAL_CH1_CIKIS_SIC',
                                                                        'KS10_REAL_CH1_GIRIS_SIC',
                                                                        'KS10_REAL_CH2_CIKIS_SIC',
                                                                        'KS10_REAL_CH2_GIRIS_SIC',
                                                                        'KS10_REAL_CH3_CIKIS_SIC',
                                                                        'KS10_REAL_CH3_GIRIS_SIC',
                                                                        'KS10_REAL_OUTLET_TEMP',
                                                                        'KS10_REAL_OUTLET_HUMIDITY',
                                                                        'KS10_REAL_KS10_UDP_TUKETIM',
                                                                        'solved_for_first_timestamp'])

KS10_OUTPUT_DATA.to_csv(OUTPUT_DATA_PATH, index=False)

TS = alg.TS
SA = alg.SA

InputHandler.set_REAL_DATA(REAL_DATA_PATH)

InputHandler.set_PUE_MODEL(PUE_MODEL_PATH)
InputHandler.set_SENSOR_I_MODEL(SENSOR_I_MODEL_PATH)
InputHandler.set_SENSOR_II_MODEL(SENSOR_II_MODEL_PATH)

InputHandler.set_CH_CIKIS_SIC_MODEL([CH1_CIKIS_SIC_MODEL_PATH,
                                     CH2_CIKIS_SIC_MODEL_PATH,
                                     CH3_CIKIS_SIC_MODEL_PATH],
                                    [1, 2, 3])
InputHandler.set_CH_GIRIS_SIC_MODEL([CH1_GIRIS_SIC_MODEL_PATH,
                                     CH2_GIRIS_SIC_MODEL_PATH,
                                     CH3_GIRIS_SIC_MODEL_PATH],
                                    [1, 2, 3])

InputHandler.set_KS10_UDP_TUKETIM_MODEL(KS10_UDP_TUKETIM_MODEL_PATH)

# create solution object

for i in range(100):
    start_timestamp = str(pd.to_datetime(START_TIMESTAMP) + pd.Timedelta(minutes=5 * i))

    input_h = InputHandler(start_timestamp=start_timestamp,
                           no_timestamps=NO_TIMESTAMPS,
                           output_path=OUTPUT_DATA_PATH)

    sol = Solution_Rep(input_h)

    '''TS_result = TS(sol.init, sol.N_List, sol.objective,
                   tabu_size=10, num_neighbors=30,
                   time_limit=60, ITER=1000000, print_results=False, print_iteration=True)'''

    SA_result = SA(sol.init, sol.N_List, sol.objective,
                   time_limit=20, ITER=1000000, print_results=False, print_iteration=False)

    meta_sol = SA_result
    # objective_meta = sol.objective(meta_sol)

    sol.excel_write(meta_sol, OUTPUT_DATA_PATH, KS10_REAL_DATA)
