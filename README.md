# SE3 Electricity Price Forecast

> Work in progress. More details will be added when the project is finished.

This project forecasts tomorrow's hourly electricity prices in Sweden's SE3 price zone (Stockholm region).

The official day-ahead prices for the next day are published around 13:00 Swedish time. The aim is to predict all 24 hourly prices for the next day with a machine learning model before they are published. The forecasts are based on historical prices, calendar effects such as weekdays and Swedish holidays and weather data.

The goal is to build a complete forecasting system, from collecting the data and training the model to publishing new forecasts automatically every day.

The best model so far has a 34% lower average error (MAE) than a simple baseline forecast ("same hour yesterday") on a full year of unseen test data (Oct 2025 – Sep 2026).

## What drives the forecasts

To understand how the models reach their forecasts, they were analysed with SHAP values and permutation importance on the test year.

- The most important information is the price development over the last week, yesterday's price at the same hour, the temperature and the time of day.
- Cold weather, high recent prices and the morning and evening hours raise the forecast. Wind, sunshine, weekends and holidays lower it.
- Extreme price spikes remain the hardest part. For the most expensive hour of the test year (19 February 2026 at 08:00, 4.90 SEK/kWh), the best model forecast 1.82 SEK/kWh: it recognised the hour as expensive but predicted less than half of the actual price.

SHAP values show how a model uses its inputs, not proven causes. Inputs that carry similar information, such as yesterday's price and the price history of the last week, share their importance between them. The SHAP charts explain a version of each model trained once before the test year, so their forecasts can differ slightly from the monthly retrained models above (for the hour above: 1.76 instead of 1.82 SEK/kWh). The charts are in the `figures` folder.

Data sources: [elprisetjustnu.se](https://www.elprisetjustnu.se) for electricity prices and [Open-Meteo](https://open-meteo.com) for weather.