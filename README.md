# Live forecasts

This branch is written automatically every morning by the daily forecast workflow.
- forecasts.csv: every forecast hour, with the LSTM and weekly_naive forecasts and, once published, the real price
- summary.json: the live accuracy so far (only forecasts made before 12:00 Swedish time count)
- model/lstm.pt: the LSTM trained for the current month
