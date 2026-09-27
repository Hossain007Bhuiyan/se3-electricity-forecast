# SE3 Electricity Price Forecast

> Work in progress. More details will be added when the project is finished.

This project forecasts tomorrow's hourly electricity prices in Sweden's SE3 price zone (Stockholm region).

Every morning, before the official day-ahead prices are published, a machine learning model predicts all 24 hourly prices for the next day. The forecasts are based on historical prices, calendar effects such as weekdays and Swedish holidays, and weather data.

The goal is to build a complete forecasting system: from collecting the data and training the model to publishing new forecasts automatically every day.

Data sources: [elprisetjustnu.se](https://www.elprisetjustnu.se) for electricity prices and [Open-Meteo](https://open-meteo.com) for weather.