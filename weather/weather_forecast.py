import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry


def fetch_past_weather_data(start_date, end_date, frequency):
    """
    Fetch weather data from Open-Meteo API for the given date range and frequency.

    Parameters:
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data in 'YYYY-MM-DD' format.
    frequency (str): The frequency of the data, either 'hourly' or 'daily'.
    
    returns dataframe
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 41.0138,
        "longitude": 28.9497,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Europe/Moscow"
    }

    # Add parameters based on the frequency
    if frequency == 'hourly':
        params['hourly'] = ["temperature_2m", "relative_humidity_2m"]
    elif frequency == 'daily':
        params['daily'] = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"]
    else:
        raise ValueError("Frequency must be either 'hourly' or 'daily'.")

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    print(f"Coordinates {response.Latitude()}°E {response.Longitude()}°N")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")

    if frequency == 'hourly':
        hourly = response.Hourly()
        temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
        ), "temperature": temperature_2m, "relative_humidity": relative_humidity_2m}

        hourly_data["date"] = hourly_data["date"][1:]

        return pd.DataFrame(data=hourly_data)

    elif frequency == 'daily':
        daily = response.Daily()
        temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
        temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
        temperature_2m_mean = daily.Variables(2).ValuesAsNumpy()

        daily_data = {"date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s"),
            end=pd.to_datetime(daily.TimeEnd(), unit="s"),
            freq='D',  # Daily frequency
        ), "temperature_max": temperature_2m_max, "temperature_min": temperature_2m_min,
            "temperature_mean": temperature_2m_mean}

        # Adjust the date to match the data length
        daily_data["date"] = daily_data["date"][1:]

        return pd.DataFrame(data=daily_data)


def getWeatherForecast(forecast_type="daily", forecast_days=7):
    """
    Fetches and returns weather data for the specified forecast type and number of days.
    
    Parameters:
    forecast_type (str): Type of forecast to return, "current", "minutely_15", "hourly", or "daily".
    forecast_days (int): Number of days to forecast, could be 1, 3, 7, 14, 16.
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 41.0138,
        "longitude": 28.9497,
        "current": ["temperature_2m", "relative_humidity_2m"],
        "minutely_15": ["temperature_2m", "relative_humidity_2m"],
        "hourly": ["temperature_2m", "relative_humidity_2m"],
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "timezone": "Europe/Moscow",
        "forecast_days": forecast_days
    }
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]

    if forecast_type == "daily":
        daily = response.Daily()
        daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()

        daily_data = {"date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s"),
            end=pd.to_datetime(daily.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=daily.Interval()),
        ), "temperature_2m_max": daily_temperature_2m_max, "temperature_2m_min": daily_temperature_2m_min}

        daily_data["date"] = daily_data["date"][1:]
        daily_dataframe = pd.DataFrame(data=daily_data)

        return daily_dataframe

    elif forecast_type == "hourly":
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
        ), "temperature_2m": hourly_temperature_2m, "relative_humidity_2m": hourly_relative_humidity_2m}

        hourly_data["date"] = hourly_data["date"][1:]
        hourly_dataframe = pd.DataFrame(data=hourly_data)

        return hourly_dataframe

    elif forecast_type == "minutely_15":
        minutely_15 = response.Minutely15()
        minutely_15_temperature_2m = minutely_15.Variables(0).ValuesAsNumpy()
        minutely_15_relative_humidity_2m = minutely_15.Variables(1).ValuesAsNumpy()

        minutely_15_data = {"date": pd.date_range(
            start=pd.to_datetime(minutely_15.Time(), unit="s"),
            end=pd.to_datetime(minutely_15.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=minutely_15.Interval()),
        ), "temperature_2m": minutely_15_temperature_2m, "relative_humidity_2m": minutely_15_relative_humidity_2m}

        minutely_15_data["date"] = minutely_15_data["date"][1:]
        minutely_15_dataframe = pd.DataFrame(data=minutely_15_data)

        return minutely_15_dataframe

    elif forecast_type == "current":
        current = response.Current()
        current_data = {
            "time": [pd.to_datetime(current.Time(), unit='s')],
            "temperature": [current.Variables(0).Value()],
            "relative_humidity": [current.Variables(1).Value()]
        }
        current_dataframe = pd.DataFrame(data=current_data)
        return current_dataframe
    else:
        raise ValueError("Invalid forecast type specified. Choose 'daily', 'hourly', or 'minutely_15'.")
