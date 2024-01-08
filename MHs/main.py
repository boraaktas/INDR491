import pandas as pd

import algorithms as alg
from InputHandler import InputHandler
from Solution_Rep import Solution_Rep

REAL_DATA_PATH = '../data/KS VERI/KS10_FINAL_DATA.csv'
OUTPUT_DATA_PATH = '../data/KS VERI/KS10_MH_OUTPUT_DATA.csv'

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
NO_TIMESTAMPS = 1

# create KS10_REAL_DATA to start
# The first timestamp of KS10_REAL_DATA is 2023-04-01 00:10:00
# The last timestamp of KS10_REAL_DATA is 2023-10-25 01:05:00
KS10_REAL_DATA = pd.read_csv(REAL_DATA_PATH)
KS10_REAL_DATA['Timestamp'] = pd.to_datetime(KS10_REAL_DATA['Timestamp'])

# create an empty KS10_OUTPUT_DATA
KS10_OUTPUT_DATA = pd.DataFrame(columns=KS10_REAL_DATA.columns)
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

TIMESTAMP_LENGTH = 10

# create solution object
input_h = InputHandler(start_timestamp=START_TIMESTAMP,
                       no_timestamps=NO_TIMESTAMPS,
                       output_path=OUTPUT_DATA_PATH)

sol = Solution_Rep(input_h)

'''TS_result = TS(sol.init, sol.N_List, sol.objective,
                 tabu_size=1, num_neighbors=2,
                 time_limit=4, ITER=1000000, print_results=False, print_iteration=False)

SA_result = SA(sol.init, sol.N_List, sol.objective,
               time_limit=30, ITER=1000000, print_results=False, print_iteration=False)

meta_sol = SA_result
# objective_meta = sol.objective(meta_sol)

sol.excel_write(meta_sol, OUTPUT_DATA_PATH, KS10_REAL_DATA)
'''