from typing import Optional

import pandas as pd


class Solution_Object:

    def __init__(self: 'Solution_Object',
                 timestamp: pd.Timestamp,
                 PUE: Optional[float],
                 I_KOMP1_HIZ: float,
                 II_KOMP1_HIZ: float,
                 III_KOMP1_HIZ: float,
                 IV_KOMP1_HIZ: float,
                 SENSOR_I_TEMP: Optional[float],
                 SENSOR_II_TEMP: Optional[float],
                 KS10_UDP_TUKETIM: float,
                 CH1_CIKIS_SIC: float,
                 CH1_GIRIS_SIC: float,
                 CH2_CIKIS_SIC: float,
                 CH2_GIRIS_SIC: float,
                 CH3_CIKIS_SIC: float,
                 CH3_GIRIS_SIC: float,
                 OUTLET_TEMP: float,
                 OUTLET_HUMIDITY: float,
                 DUMMIES: list,
                 PUE_lag_one_day: float,
                 SIC_I_SET: float,
                 SIC_II_SET: float) -> None:
        self.timestamp = timestamp
        self.PUE = PUE
        self.I_KOMP1_HIZ = I_KOMP1_HIZ
        self.II_KOMP1_HIZ = II_KOMP1_HIZ
        self.III_KOMP1_HIZ = III_KOMP1_HIZ
        self.IV_KOMP1_HIZ = IV_KOMP1_HIZ
        self.SENSOR_I_TEMP = SENSOR_I_TEMP
        self.SENSOR_II_TEMP = SENSOR_II_TEMP
        self.KS10_UDP_TUKETIM = KS10_UDP_TUKETIM
        self.CH1_CIKIS_SIC = CH1_CIKIS_SIC
        self.CH1_GIRIS_SIC = CH1_GIRIS_SIC
        self.CH2_CIKIS_SIC = CH2_CIKIS_SIC
        self.CH2_GIRIS_SIC = CH2_GIRIS_SIC
        self.CH3_CIKIS_SIC = CH3_CIKIS_SIC
        self.CH3_GIRIS_SIC = CH3_GIRIS_SIC
        self.OUTLET_TEMP = OUTLET_TEMP
        self.OUTLET_HUMIDITY = OUTLET_HUMIDITY
        self.DUMMIES = DUMMIES
        self.PUE_lag_one_day = PUE_lag_one_day
        self.SIC_I_SET = SIC_I_SET
        self.SIC_II_SET = SIC_II_SET

        # first values of the speeds
        self.init_I_KOMP1_HIZ = I_KOMP1_HIZ
        self.init_II_KOMP1_HIZ = II_KOMP1_HIZ
        self.init_III_KOMP1_HIZ = III_KOMP1_HIZ
        self.init_IV_KOMP1_HIZ = IV_KOMP1_HIZ

    def __str__(self):
        return (f"timestamp: {self.timestamp}, "
                f"PUE: {self.PUE}, \n"
                f"I_KOMP1_HIZ: {self.I_KOMP1_HIZ}, "
                f"II_KOMP1_HIZ: {self.II_KOMP1_HIZ}, "
                f"III_KOMP1_HIZ: {self.III_KOMP1_HIZ}, "
                f"IV_KOMP1_HIZ: {self.IV_KOMP1_HIZ}, \n"
                f"KS10_UDP_TUKETIM: {self.KS10_UDP_TUKETIM}, \n"
                f"CH1_CIKIS_SIC: {self.CH1_CIKIS_SIC}, "
                f"CH1_GIRIS_SIC: {self.CH1_GIRIS_SIC}, "
                f"CH2_CIKIS_SIC: {self.CH2_CIKIS_SIC}, "
                f"CH2_GIRIS_SIC: {self.CH2_GIRIS_SIC}, "
                f"CH3_CIKIS_SIC: {self.CH3_CIKIS_SIC}, "
                f"CH3_GIRIS_SIC: {self.CH3_GIRIS_SIC}, \n"
                f"SENSOR_I_TEMP: {self.SENSOR_I_TEMP}, "
                f"SENSOR_II_TEMP: {self.SENSOR_II_TEMP}, \n"
                f"OUTLET_TEMP: {self.OUTLET_TEMP}, "
                f"OUTLET_HUMIDITY: {self.OUTLET_HUMIDITY}, \n"
                f"DUMMIES: {self.DUMMIES}, \n"
                f"PUE_lag_one_day: {self.PUE_lag_one_day}, \n"
                f"SIC_I_SET: {self.SIC_I_SET}, SIC_II_SET: {self.SIC_II_SET}")
