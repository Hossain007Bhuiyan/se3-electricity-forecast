# SE3 Electricity Price Forecast

<a href="https://github.com/Hossain007Bhuiyan/se3-electricity-forecast/actions/workflows/tests.yml"><img src="https://github.com/Hossain007Bhuiyan/se3-electricity-forecast/actions/workflows/tests.yml/badge.svg" alt="tests" height="38"></a>
> Status: work in progress. Daily automated forecasts and a dashboard are being added.

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

## Experiment tracking

All model runs are tracked with MLflow: the baselines, the four LightGBM versions and the LSTM, on both the validation year and the test year. For every run, MLflow records its settings, its results (MAE, RMSE and the error for every month), its hourly predictions, the Git commit of the code and a fingerprint of the data. This makes all runs comparable side by side, and every result can be traced back to the exact code and data behind it. The tracking data is stored locally and is not part of this repository; the screenshots show the tracked runs.

<img width="1652" height="1162" alt="Image" src="https://github.com/user-attachments/assets/74cb362d-e7ad-4497-8034-e60c8b53a42e" />
<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/f0ef61ed-9f10-4a70-ac3b-f2eb1d914c53" />

## Tests

27 automated tests (pytest) check the most important parts of the project: that features and LSTM input sequences only use information available at forecast time, the time-based split, the summer/winter time changes, Swedish holidays, MLflow tracking and that model training is repeatable. They use small synthetic data and run automatically with GitHub Actions on every push.

To run them locally:

    uv run pytest
    uv run pytest tests_torch

The PyTorch tests run as a separate command, because LightGBM and PyTorch must not be loaded in the same Python process on macOS.

## Next steps

- Daily forecasts: every morning, new data and a weather forecast are downloaded, tomorrow's 24 hourly prices are predicted and the forecasts are later compared with the actual prices
- A dashboard showing the latest forecast and the live accuracy
- Final documentation with the full results, limitations and an architecture diagram

Data sources: [elprisetjustnu.se](https://www.elprisetjustnu.se) for electricity prices and [Open-Meteo](https://open-meteo.com) for weather.