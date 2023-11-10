import pandas as pd
import os 
import re

current_path = os.getcwd()
MASTER_DATA_PATH = os.path.join(current_path, 'data/KS VERI/KS10_MASTER_DATA.xlsx')

KS10_UDP_TUKETIM = pd.read_excel(MASTER_DATA_PATH, sheet_name='KS10_UDP_TUKETIM')
KS10_KLIMA_TUKETIM = pd.read_excel(MASTER_DATA_PATH, sheet_name='KS10_KLIMA_TUKETIM')
KS10_KLIMA_VERILERI = pd.read_excel(MASTER_DATA_PATH, sheet_name='KS10_KLIMA_VERILERI')
KS10_SENSOR_I = pd.read_excel(MASTER_DATA_PATH, sheet_name='KS10_SENSOR_I')
KS10_SENSOR_II = pd.read_excel(MASTER_DATA_PATH, sheet_name='KS10_SENSOR_II')
CHILLER_MECHANIC_ROOM = pd.read_excel(MASTER_DATA_PATH, sheet_name='CHILLER_ve_MEKANİK_ROOM')
CHILLER_SICAKLIK = pd.read_excel(MASTER_DATA_PATH, sheet_name='CHILLER')
CHILLER_SICAKLIK_NONAN = pd.read_excel(MASTER_DATA_PATH, sheet_name='CHILLER_NONAN')
CHILLER_SICAKLIK_NONAN_NOEXTREME = pd.read_excel(MASTER_DATA_PATH, sheet_name='CHILLER_NONAN_NOEXTREME_10_35')

print('DATA LOADED')

def get_INNER_TEMP(sensor_number: int,
                   date_time: str) -> float:
    """
    Get the closest inner temperature value to the given date time.
    
    Args:
        sensor_number: The sensor number.
        date_time: The date time.
    
    Returns:
        The inner temperature value.
    """
    
    if sensor_number > 2 or sensor_number < 1:
        raise ValueError('sensor_number must be 1 or 2.')
    
    # check if the date time is in the correct format with regex
    '''if not re.match(r'\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}', date_time):
        raise ValueError('date_time must be in the format of dd.mm.yyyy hh:mm:ss')'''
    
    if sensor_number == 1:
        sensor_df = KS10_SENSOR_I
    else:
        sensor_df = KS10_SENSOR_II
        
    # convert the date time to datetime object
    date_time = pd.to_datetime(date_time, dayfirst=True)
            
    # find the closest date time
    closest_date_time = sensor_df['Date Time'].iloc[(pd.to_datetime(sensor_df['Date Time']) - date_time).abs().argsort()[:1]].values[0]
    
    # get the inner temperature value
    inner_temp = sensor_df[sensor_df['Date Time'] == closest_date_time]['KN-2 SENSOR-' + str(sensor_number) + ' ISI'].values[0]
    
    return inner_temp

def main():
    # get the inner temperature value
    inner_temp: float = get_INNER_TEMP(sensor_number=1,
                                       date_time='26.08.2020 10:16:00')
    
    print(inner_temp)
    
if __name__ == '__main__':
    main()
    