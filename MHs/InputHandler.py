import joblib
import pandas as pd

from Solution_Object import Solution_Object


class InputHandler:
    __KS10_REAL_DATA__: pd.DataFrame

    __PUE_MODEL__: joblib
    __SENSOR_I_MODEL__: joblib
    __SENSOR_II_MODEL__: joblib

    __CH1_CIKIS_SIC_MODEL__: joblib
    __CH1_GIRIS_SIC_MODEL__: joblib
    __CH2_CIKIS_SIC_MODEL__: joblib
    __CH2_GIRIS_SIC_MODEL__: joblib
    __CH3_CIKIS_SIC_MODEL__: joblib
    __CH3_GIRIS_SIC_MODEL__: joblib

    __KS10_UDP_TUKETIM_MODEL__: joblib

    def __init__(self: 'InputHandler',
                 start_timestamp: str = '2023-10-18 08:00:00',
                 no_timestamps: int = 1,
                 output_path: str = 'KS10_OUTPUT_DATA.csv') -> None:

        self.KS10_OUTPUT_DATA = pd.read_csv(output_path)
        self.KS10_OUTPUT_DATA['Timestamp'] = pd.to_datetime(self.KS10_OUTPUT_DATA['Timestamp'])

        self.start_timestamp = start_timestamp
        self.no_timestamps = no_timestamps

        self.solutions = []

        self.creation()

    def creation(self: 'InputHandler') -> None:

        start_timestamp = self.start_timestamp
        no_timestamps = self.no_timestamps

        KS10_REAL_DATA = InputHandler.get_REAL_DATA()

        for i in range(no_timestamps):

            current_timestamp = pd.to_datetime(start_timestamp) + pd.Timedelta(minutes=5 * i)

            ks10_udp_tuketim = self.predict_KS10_UDP_TUKETIM(current_timestamp,
                                                             KS10_REAL_DATA)

            ch1_cikis_sic = self.predict_CHILLER_CIKIS_SIC(1,
                                                           current_timestamp,
                                                           KS10_REAL_DATA)
            ch1_giris_sic = self.predict_CHILLER_GIRIS_SIC(1,
                                                           current_timestamp,
                                                           KS10_REAL_DATA)
            ch2_cikis_sic = self.predict_CHILLER_CIKIS_SIC(2,
                                                           current_timestamp,
                                                           KS10_REAL_DATA)
            ch2_giris_sic = self.predict_CHILLER_GIRIS_SIC(2,
                                                           current_timestamp,
                                                           KS10_REAL_DATA)
            ch3_cikis_sic = self.predict_CHILLER_CIKIS_SIC(3,
                                                           current_timestamp,
                                                           KS10_REAL_DATA)
            ch3_giris_sic = self.predict_CHILLER_GIRIS_SIC(3,
                                                           current_timestamp,
                                                           KS10_REAL_DATA)

            outlet_temp = InputHandler.predict_OUTLET_TEMP(current_timestamp,
                                                           KS10_REAL_DATA)

            outlet_humidity = InputHandler.predict_OUTLET_HUMIDITY(current_timestamp,
                                                                   KS10_REAL_DATA)

            dummies = InputHandler.add_DUMMIES(current_timestamp)

            pue_lag_one_day = InputHandler.add_predict_PUE_lag_one_day(current_timestamp,
                                                                       KS10_REAL_DATA,
                                                                       self.KS10_OUTPUT_DATA)

            sic_i_set = InputHandler.add_SIC_I_SET(current_timestamp,
                                                   KS10_REAL_DATA)
            sic_ii_set = InputHandler.add_SIC_II_SET(current_timestamp,
                                                     KS10_REAL_DATA)

            # if current_timestamp is in KS10_OUTPUT_DATA, start with them
            if current_timestamp in self.KS10_OUTPUT_DATA['Timestamp'].values:
                I_komp1_hiz = self.KS10_OUTPUT_DATA[self.KS10_OUTPUT_DATA['Timestamp']
                                                    == current_timestamp]['I_KOMP1_HIZ'].values[0]
                II_komp1_hiz = self.KS10_OUTPUT_DATA[self.KS10_OUTPUT_DATA['Timestamp']
                                                     == current_timestamp]['II_KOMP1_HIZ'].values[0]
                III_komp1_hiz = self.KS10_OUTPUT_DATA[self.KS10_OUTPUT_DATA['Timestamp']
                                                      == current_timestamp]['III_KOMP1_HIZ'].values[0]
                IV_komp1_hiz = self.KS10_OUTPUT_DATA[self.KS10_OUTPUT_DATA['Timestamp']
                                                     == current_timestamp]['IV_KOMP1_HIZ'].values[0]

            # else start with the speed in 5 minutes ago from start_timestamp
            else:
                old_timestamp = pd.to_datetime(start_timestamp) - pd.Timedelta(minutes=5)
                I_komp1_hiz = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp']
                                             == old_timestamp]['I_KOMP1_HIZ'].values[0]
                II_komp1_hiz = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp']
                                              == old_timestamp]['II_KOMP1_HIZ'].values[0]
                III_komp1_hiz = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp']
                                               == old_timestamp]['III_KOMP1_HIZ'].values[0]
                IV_komp1_hiz = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp']
                                              == old_timestamp]['IV_KOMP1_HIZ'].values[0]

            # create solution object
            solution = Solution_Object(timestamp=current_timestamp,
                                       PUE=None,
                                       I_KOMP1_HIZ=float(I_komp1_hiz),
                                       II_KOMP1_HIZ=float(II_komp1_hiz),
                                       III_KOMP1_HIZ=float(III_komp1_hiz),
                                       IV_KOMP1_HIZ=float(IV_komp1_hiz),
                                       SENSOR_I_TEMP=None,
                                       SENSOR_II_TEMP=None,
                                       KS10_UDP_TUKETIM=float(ks10_udp_tuketim),
                                       CH1_CIKIS_SIC=float(ch1_cikis_sic),
                                       CH1_GIRIS_SIC=float(ch1_giris_sic),
                                       CH2_CIKIS_SIC=float(ch2_cikis_sic),
                                       CH2_GIRIS_SIC=float(ch2_giris_sic),
                                       CH3_CIKIS_SIC=float(ch3_cikis_sic),
                                       CH3_GIRIS_SIC=float(ch3_giris_sic),
                                       OUTLET_TEMP=float(outlet_temp),
                                       OUTLET_HUMIDITY=float(outlet_humidity),
                                       DUMMIES=dummies,
                                       PUE_lag_one_day=float(pue_lag_one_day),
                                       SIC_I_SET=float(sic_i_set),
                                       SIC_II_SET=float(sic_ii_set))

            self.solutions.append(solution)

    def predict_KS10_UDP_TUKETIM(self,
                                 current_timestamp: pd.Timestamp,
                                 KS10_REAL_DATA: pd.DataFrame,
                                 pred_start=0,
                                 no_pred=1) -> float:

        if current_timestamp == self.start_timestamp:
            return KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] == current_timestamp]['KS10_UDP_TUKETIM'].values[0]

        if no_pred < 1:
            raise ValueError('no_pred must be greater than 1')

        data = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] < current_timestamp]['KS10_UDP_TUKETIM']

        preds = []

        model = InputHandler.get_KS10_UDP_TUKETIM_MODEL()

        pred_df = model.predict(start=len(data) + pred_start, end=len(data) + pred_start + no_pred - 1)

        for i in range(len(pred_df)):
            preds.append(pred_df.iloc[i])

        return preds[0]

    def predict_CHILLER_CIKIS_SIC(self,
                                  CHILLER_NUMBER: int,
                                  current_timestamp: pd.Timestamp,
                                  KS10_REAL_DATA: pd.DataFrame,
                                  pred_start=0,
                                  no_pred=1) -> float:

        if current_timestamp == self.start_timestamp:
            return KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] == current_timestamp][(f'CH{CHILLER_NUMBER}'
                                                                                     f'_CIKIS_SIC')].values[0]

        if no_pred < 1:
            raise ValueError('no_pred must be greater than 1')

        data = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] < current_timestamp][f'CH{CHILLER_NUMBER}_CIKIS_SIC']

        preds = []

        model = InputHandler.get_CH_CIKIS_SIC_MODEL(CHILLER_NUMBER)
        pred_df = model.predict(start=len(data) + pred_start, end=len(data) + pred_start + no_pred - 1)
        for i in range(len(pred_df)):
            preds.append(pred_df.iloc[i])

        return preds[0]

    def predict_CHILLER_GIRIS_SIC(self,
                                  CHILLER_NUMBER: int,
                                  current_timestamp: pd.Timestamp,
                                  KS10_REAL_DATA: pd.DataFrame,
                                  pred_start=0,
                                  no_pred=1) -> float:

        if current_timestamp == self.start_timestamp:
            return KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] == current_timestamp][(f'CH{CHILLER_NUMBER}'
                                                                                     f'_GIRIS_SIC')].values[0]

        if no_pred < 1:
            raise ValueError('no_pred must be greater than 1')

        data = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp'] < current_timestamp][f'CH{CHILLER_NUMBER}_GIRIS_SIC']

        preds = []

        model = InputHandler.get_CH_GIRIS_SIC_MODEL(CHILLER_NUMBER)
        pred_df = model.predict(start=len(data) + pred_start, end=len(data) + pred_start + no_pred - 1)
        for i in range(len(pred_df)):
            preds.append(pred_df.iloc[i])

        return preds[0]

    @staticmethod
    def predict_OUTLET_TEMP(current_timestamp: pd.Timestamp,
                            KS10_REAL_DATA: pd.DataFrame) -> float:
        outlet_temp = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp']
                                     == current_timestamp]['OUTLET_TEMP'].values[0]
        return outlet_temp

    @staticmethod
    def predict_OUTLET_HUMIDITY(current_timestamp: pd.Timestamp,
                                KS10_REAL_DATA: pd.DataFrame) -> float:
        outlet_humidity = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp']
                                         == current_timestamp]['OUTLET_HUMIDITY'].values[0]
        return outlet_humidity

    @staticmethod
    def add_DUMMIES(current_timestamp: pd.Timestamp) -> list:

        REAL_DATA = InputHandler.get_REAL_DATA()

        dummies = []
        mon = REAL_DATA[REAL_DATA['Timestamp'] == current_timestamp]['Mon'].values[0]
        tue = REAL_DATA[REAL_DATA['Timestamp'] == current_timestamp]['Tue'].values[0]
        wed = REAL_DATA[REAL_DATA['Timestamp'] == current_timestamp]['Wed'].values[0]
        thu = REAL_DATA[REAL_DATA['Timestamp'] == current_timestamp]['Thu'].values[0]
        fri = REAL_DATA[REAL_DATA['Timestamp'] == current_timestamp]['Fri'].values[0]
        sat = REAL_DATA[REAL_DATA['Timestamp'] == current_timestamp]['Sat'].values[0]
        dummies.extend([mon, tue, wed, thu, fri, sat])

        for i in range(24):
            hour = REAL_DATA[REAL_DATA['Timestamp'] == current_timestamp][f'hour_{i}'].values[0]
            dummies.append(hour)

        for i in range(4, 12):
            month = REAL_DATA[REAL_DATA['Timestamp'] == current_timestamp][f'month_{i}'].values[0]
            dummies.append(month)

        return dummies

    @staticmethod
    def add_predict_PUE_lag_one_day(current_timestamp: pd.Timestamp,
                                    KS10_REAL_DATA: pd.DataFrame,
                                    KS10_OUTPUT_DATA: pd.DataFrame) -> float:
        # if lag_one_day_timestamp is in KS10_OUTPUT_DATA, use PUE from KS10_OUTPUT_DATA
        # else use PUE from KS10_REAL_DATA
        lag_one_day_timestamp = current_timestamp - pd.Timedelta(days=1)

        if lag_one_day_timestamp in KS10_OUTPUT_DATA['Timestamp'].values:
            pue_lag_one_day = KS10_OUTPUT_DATA[KS10_OUTPUT_DATA['Timestamp']
                                               == lag_one_day_timestamp]['PUE'].values[0]
        else:
            pue_lag_one_day = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp']
                                             == lag_one_day_timestamp]['PUE'].values[0]

        return pue_lag_one_day

    @staticmethod
    def add_SIC_I_SET(current_timestamp: pd.Timestamp,
                      KS10_REAL_DATA: pd.DataFrame) -> float:

        I_SIC_SET = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp']
                                   == current_timestamp]['I_SIC_SET'].values[0]
        II_SIC_SET = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp']
                                    == current_timestamp]['II_SIC_SET'].values[0]

        return (I_SIC_SET + II_SIC_SET) / 2

    @staticmethod
    def add_SIC_II_SET(current_timestamp: pd.Timestamp,
                       KS10_REAL_DATA: pd.DataFrame) -> float:
        III_SIC_SET = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp']
                                     == current_timestamp]['III_SIC_SET'].values[0]
        IV_SIC_SET = KS10_REAL_DATA[KS10_REAL_DATA['Timestamp']
                                    == current_timestamp]['IV_SIC_SET'].values[0]

        return (III_SIC_SET + IV_SIC_SET) / 2

    @classmethod
    def set_REAL_DATA(cls,
                      path: str) -> None:
        cls.__KS10_REAL_DATA__ = pd.read_csv(path)
        cls.__KS10_REAL_DATA__['Timestamp'] = pd.to_datetime(cls.__KS10_REAL_DATA__['Timestamp'])

    @classmethod
    def set_PUE_MODEL(cls,
                      path: str) -> None:
        cls.__PUE_MODEL__ = joblib.load(path)

    @classmethod
    def set_SENSOR_I_MODEL(cls,
                           path: str) -> None:
        cls.__SENSOR_I_MODEL__ = joblib.load(path)

    @classmethod
    def set_SENSOR_II_MODEL(cls,
                            path: str) -> None:
        cls.__SENSOR_II_MODEL__ = joblib.load(path)

    @classmethod
    def set_CH_CIKIS_SIC_MODEL(cls,
                               paths: list[str] | str,
                               chiller_nos: list[int] | int) -> None:

        if isinstance(chiller_nos, list):
            for i in chiller_nos:
                cls.set_CH_CIKIS_SIC_MODEL(paths[i - 1], i)

        else:
            if chiller_nos == 1:
                cls.__CH1_CIKIS_SIC_MODEL__ = joblib.load(paths)

            elif chiller_nos == 2:
                cls.__CH2_CIKIS_SIC_MODEL__ = joblib.load(paths)

            elif chiller_nos == 3:
                cls.__CH3_CIKIS_SIC_MODEL__ = joblib.load(paths)

            else:
                raise ValueError('chiller_no must be 1, 2 or 3')

    @classmethod
    def set_CH_GIRIS_SIC_MODEL(cls,
                               paths: list[str] | str,
                               chiller_nos: list[int] | int) -> None:

        if isinstance(chiller_nos, list):
            for i in chiller_nos:
                cls.set_CH_GIRIS_SIC_MODEL(paths[i - 1], i)

        else:
            if chiller_nos == 1:
                cls.__CH1_GIRIS_SIC_MODEL__ = joblib.load(paths)

            elif chiller_nos == 2:
                cls.__CH2_GIRIS_SIC_MODEL__ = joblib.load(paths)

            elif chiller_nos == 3:
                cls.__CH3_GIRIS_SIC_MODEL__ = joblib.load(paths)

            else:
                raise ValueError('chiller_no must be 1, 2 or 3')

    @classmethod
    def set_KS10_UDP_TUKETIM_MODEL(cls,
                                   path: str) -> None:
        cls.__KS10_UDP_TUKETIM_MODEL__ = joblib.load(path)

    @classmethod
    def get_REAL_DATA(cls) -> pd.DataFrame:
        return cls.__KS10_REAL_DATA__

    @classmethod
    def get_PUE_MODEL(cls) -> joblib:
        return cls.__PUE_MODEL__

    @classmethod
    def get_SENSOR_I_MODEL(cls) -> joblib:
        return cls.__SENSOR_I_MODEL__

    @classmethod
    def get_SENSOR_II_MODEL(cls) -> joblib:
        return cls.__SENSOR_II_MODEL__

    @classmethod
    def get_CH_CIKIS_SIC_MODEL(cls,
                               chiller_no: int) -> joblib:

        if chiller_no == 1:
            return cls.__CH1_CIKIS_SIC_MODEL__

        elif chiller_no == 2:
            return cls.__CH2_CIKIS_SIC_MODEL__

        elif chiller_no == 3:
            return cls.__CH3_CIKIS_SIC_MODEL__

        else:
            raise ValueError('chiller_no must be 1, 2 or 3')

    @classmethod
    def get_CH_GIRIS_SIC_MODEL(cls,
                               chiller_no: int) -> joblib:

        if chiller_no == 1:
            return cls.__CH1_GIRIS_SIC_MODEL__

        elif chiller_no == 2:
            return cls.__CH2_GIRIS_SIC_MODEL__

        elif chiller_no == 3:
            return cls.__CH3_GIRIS_SIC_MODEL__

        else:
            raise ValueError('chiller_no must be 1, 2 or 3')

    @classmethod
    def get_KS10_UDP_TUKETIM_MODEL(cls) -> joblib:
        return cls.__KS10_UDP_TUKETIM_MODEL__

