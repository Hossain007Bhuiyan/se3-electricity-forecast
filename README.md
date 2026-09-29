# SE3 Electricity Price Forecast

> Work in progress. More details will be added when the project is finished.

This project forecasts tomorrow's hourly electricity prices in Sweden's SE3 price zone (Stockholm region).

The official day-ahead prices for the next day are published around 13:00 Swedish time. The aim is to predict all 24 hourly prices for the next day with a machine learning model before they are published. The forecasts are based on historical prices, calendar effects such as weekdays and Swedish holidays and weather data.

The goal is to build a complete forecasting system, from collecting the data and training the model to publishing new forecasts automatically every day.

The best model so far is about 34% more accurate than a simple baseline forecast ("same hour yesterday") on a full year of unseen test data (Oct 2025 – Sep 2026).

Data sources: [elprisetjustnu.se](https://www.elprisetjustnu.se) for electricity prices and [Open-Meteo](https://open-meteo.com) for weather.