# The pages of the dashboard and the building blocks they share: data loading, chart style,
# cards and section titles. app.py draws the frame around them (title, live badge, clock,
# navigation menu and footer); each *_page function below draws one page.

import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import live_data

CYAN, VIOLET, AMBER, PINK, GREY = "#22d3ee", "#a78bfa", "#fbbf24", "#f472b6", "#64748b"

# Readable names for the models in results/test_results.csv
MODEL_NAMES = {
    "lstm": "LSTM",
    "lgbm_price_l1": "LightGBM",
    "weekly_naive": "Baseline: weekly_naive",
    "same_hour_yesterday": "Baseline: same hour yesterday",
    "yesterday_mean": "Baseline: yesterday's average",
    "same_hour_last_week": "Baseline: same hour last week",
}

# Phones get charts without drag-to-zoom and without the chart toolbar, so a finger moving over
# a chart scrolls the page instead of zooming the chart. The browser's User-Agent header tells
# which kind of device asked for the page; computers keep the full chart controls.
def is_mobile(agent=None):
    if agent is None:
        agent = st.context.headers.get("User-Agent", "")
    return "Mobi" in agent or "Android" in agent


def show_chart(fig, key):
    if is_mobile():
        fig.update_layout(dragmode=False)
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        fig.update_scenes(dragmode=False)  # the 3D landscape: turned only with the Rotate button
        st.plotly_chart(fig, theme=None, key=key, config={"displayModeBar": False})
    else:
        st.plotly_chart(fig, theme=None, key=key)

# Data is cached for 10 minutes, so the page stays fast and picks up each new morning run.
# The real prices for the 3D view change at most once a day, so they are cached for an hour.
# The record comes with the time it was downloaded, so the page can show how fresh it is.
@st.cache_data(ttl=600)
def get_record():
    return live_data.load_record(), pd.Timestamp.now(tz=live_data.TZ)


@st.cache_data(ttl=3600)
def get_test_results():
    return live_data.load_test_results()


@st.cache_data(ttl=3600)
def get_mlflow_runs():
    return live_data.load_mlflow_runs(), live_data.load_mlflow_monthly()

@st.cache_data(ttl=600)
def get_monitoring():
    return live_data.load_monitoring()


@st.cache_data(ttl=600)
def get_run_log():
    return live_data.load_run_log()

@st.cache_data(ttl=3600)
def get_recent_prices():
    return live_data.fetch_recent_prices(days=30)


# The same dark, see-through style for every chart
def style(fig, height=420):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"), margin=dict(l=70, r=20, t=40, b=50),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"), hovermode="x unified",
        hoverlabel=dict(bgcolor="#111a33", bordercolor="#22d3ee", font=dict(color="#f8fafc", size=13), namelength=-1),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False, automargin=True)
    return fig


def section(kicker, title, text):
    st.markdown(f'<div class="section"><div class="kicker">{kicker}</div><h2>{title}</h2><p>{text}</p></div>',
                unsafe_allow_html=True)


def card(label, value, unit="", note=""):
    return (f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value">{value}<span class="unit">{unit}</span></div><div class="note">{note}</div></div>')


# Line chart of one day: the two forecasts, and the real prices once they are published
def day_chart(rows, height=430):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rows["time_local"], y=rows["lstm"], name="LSTM forecast", mode="lines",
                             line=dict(color=CYAN, width=3, shape="spline"),
                             fill="tozeroy", fillcolor="rgba(34,211,238,.10)"))
    fig.add_trace(go.Scatter(x=rows["time_local"], y=rows["weekly_naive"], name="Baseline (weekly_naive)",
                             mode="lines", line=dict(color=VIOLET, width=2, dash="dot", shape="spline")))
    if rows["actual"].notna().any():
        fig.add_trace(go.Scatter(x=rows["time_local"], y=rows["actual"], name="Real price", mode="lines+markers",
                                 line=dict(color=AMBER, width=2.5), marker=dict(size=6)))
    fig.update_traces(hovertemplate="%{y:.3f} SEK/kWh")
    fig.update_yaxes(title="SEK/kWh")
    fig.update_xaxes(tickformat="%H:%M")
    return style(fig, height)


# 3D surface of the real prices: days front to back, hours left to right, price as height.
# The "Rotate" button turns the camera once around the landscape.
def landscape_chart(table):
    days = [d.strftime("%d %b") for d in table.index]
    surface = go.Surface(z=table.to_numpy(), x=list(table.columns), y=days,
                         colorscale=[[0, "#1e1b4b"], [0.35, CYAN], [0.7, VIOLET], [1, PINK]],
                         colorbar=dict(title="SEK/kWh", thickness=12),
                         hovertemplate="%{y} %{x}:00<br>%{z:.2f} SEK/kWh<extra></extra>")
    steps = 72
    angles = [2 * math.pi * i / steps for i in range(steps + 1)]
    frames = [go.Frame(layout=dict(scene_camera=dict(eye=dict(x=1.9 * math.cos(a), y=1.9 * math.sin(a), z=1.25))))
              for a in angles]
    fig = go.Figure(data=[surface], frames=frames)
    axis = dict(gridcolor="rgba(255,255,255,.12)", backgroundcolor="rgba(0,0,0,0)", showbackground=True)
    fig.update_layout(
        scene=dict(xaxis=dict(title="Hour", tickvals=[0, 6, 12, 18, 23], **axis),
                   yaxis=dict(title="", tickvals=days[::5], ticktext=days[::5], tickfont=dict(size=11), **axis),
                   zaxis=dict(title="SEK/kWh", **axis),
                   camera=dict(eye=dict(x=1.9, y=0, z=1.25)), aspectratio=dict(x=1.4, y=1.4, z=0.7)),
        updatemenus=[dict(type="buttons", x=0.02, y=0.98, showactive=False, bgcolor="rgba(34,211,238,.15)",
                          font=dict(color="#e2e8f0"),
                          buttons=[dict(label="Rotate", method="animate",
                                        args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="Stop", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
    )
    style(fig, 620)
    fig.update_layout(hovermode="closest")
    return fig


# The live record, or a clear message (and nothing more on the page) if it cannot be loaded
def load_record():
    try:
        return get_record()
    except Exception as error:
        st.error("The live forecasts could not be loaded from GitHub right now. Please try again in a few minutes.")
        st.caption(f"Details: {error}")
        st.stop()


def show_problem(message, error):
    st.info(message)
    st.caption(f"Details: {error}")


# Tomorrow's forecast: the key numbers and the 24 hourly prices
def tomorrow_page():
    record, _ = load_record()
    day, rows = live_data.latest_day(record)
    overview = live_data.day_overview(rows)
    errors = live_data.daily_errors(record)
    if errors.empty:
        live_value, live_unit, live_note = "Collecting", "", "the first counted day appears after two morning runs"
    else:
        lstm_mae = (errors["lstm_mae"] * errors["hours"]).sum() / errors["hours"].sum()
        naive_mae = (errors["weekly_naive_mae"] * errors["hours"]).sum() / errors["hours"].sum()
        live_value, live_unit = f"{lstm_mae:.3f}", "SEK/kWh"
        live_note = f"live MAE over {len(errors)} days, baseline {naive_mae:.3f}"

    st.markdown('<div class="cards">'
                + card(f"Average forecast, {day:%d %b}", f"{overview['mean']:.2f}", "SEK/kWh", f"{overview['hours']} hours")
                + card("Most expensive hour", f"{overview['peak_time']:%H:%M}", "", f"{overview['peak_price']:.2f} SEK/kWh")
                + card("Cheapest hour", f"{overview['low_time']:%H:%M}", "", f"{overview['low_price']:.2f} SEK/kWh")
                + card("Live accuracy", live_value, live_unit, live_note)
                + '</div>', unsafe_allow_html=True)

    section("Newest forecast", f"{day:%A %d %B %Y}",
            "The LSTM forecast and the simple weekly_naive baseline for every hour (Swedish time). "
            "The real prices appear here once they are published, around 13:00 the day before.")
    show_chart(day_chart(rows), "newest")


# Live accuracy: the overall live error, its development over time, the last 30 counted days,
# and any earlier day in detail. Only the last 30 days are shown as bars, so the page stays
# readable when the record grows; the lines show the trend without the noise of single days.
def live_accuracy_page():
    record, _ = load_record()
    errors = live_data.daily_errors(record)
    section("Live accuracy", "Forecast vs. reality",
            "Only forecasts made before 12:00 Swedish time on the day before count, so every number here "
            "was forecast before the prices were known. Lower is better.")
    if errors.empty:
        st.info("No counted days yet. Each morning's forecast is checked once the real prices are published, "
                "so the first counted day appears after the second scheduled morning run.")
    else:
        lstm_mae = (errors["lstm_mae"] * errors["hours"]).sum() / errors["hours"].sum()
        naive_mae = (errors["weekly_naive_mae"] * errors["hours"]).sum() / errors["hours"].sum()
        st.markdown('<div class="cards">'
                    + card("Counted days", f"{len(errors)}", "", f"{int(errors['hours'].sum())} hours")
                    + card("Live MAE, LSTM", f"{lstm_mae:.3f}", "SEK/kWh", "all counted days")
                    + card("Live MAE, baseline", f"{naive_mae:.3f}", "SEK/kWh", "weekly_naive, same days")
                    + card("Test year, LSTM", "0.209", "SEK/kWh", "for comparison; needs several weeks of live data")
                    + '</div>', unsafe_allow_html=True)

        week = live_data.rolling_errors(errors, 7)
        if week.empty:
            st.caption(f"The 7-day trend appears after 7 counted days ({len(errors)} so far).")
        else:
            month = live_data.rolling_errors(errors, 28)
            trend = go.Figure()
            trend.add_trace(go.Scatter(x=week["day"], y=week["lstm_mae"], name="LSTM, 7 days", mode="lines",
                                       line=dict(color=CYAN, width=3)))
            trend.add_trace(go.Scatter(x=week["day"], y=week["weekly_naive_mae"], name="Baseline, 7 days",
                                       mode="lines", line=dict(color=VIOLET, width=2, dash="dot")))
            if not month.empty:
                trend.add_trace(go.Scatter(x=month["day"], y=month["lstm_mae"], name="LSTM, 28 days", mode="lines",
                                           line=dict(color=AMBER, width=2)))
            trend.update_traces(hovertemplate="%{y:.3f} SEK/kWh")
            trend.update_yaxes(title="MAE (SEK/kWh)")
            section("Trend", "Average error over the last 7 and 28 counted days", "")
            show_chart(style(trend, 360), "live_trend")

        last = errors.tail(30)
        bars = go.Figure()
        labels = [d.strftime("%d %b") for d in last["day"]]
        bars.add_trace(go.Bar(x=labels, y=last["lstm_mae"], name="LSTM", marker_color=CYAN))
        bars.add_trace(go.Bar(x=labels, y=last["weekly_naive_mae"], name="Baseline (weekly_naive)", marker_color=VIOLET))
        bars.update_yaxes(title="MAE (SEK/kWh)")
        section("Day by day", "The last 30 counted days", "")
        show_chart(style(bars, 380), "live_errors")

    # Any earlier day with real prices can be looked at in detail
    known_days = sorted(record.loc[record["actual"].notna(), "time_local"].dt.date.unique(), reverse=True)
    if known_days:
        chosen = st.selectbox("Look at a day", known_days, format_func=lambda d: d.strftime("%A %d %B %Y"))
        chosen_rows = record[record["time_local"].dt.date == chosen]
        counted = bool(chosen_rows["issued_before_noon"].all())
        st.caption("This forecast counts in the live accuracy." if counted
                   else "This forecast was made after 12:00 Swedish time, so it is shown but does not count.")
        show_chart(day_chart(chosen_rows, 380), "chosen_day")


# The latest monitoring run: deadline, live error and input drift
def monitoring_page():
    section("Monitoring", "Automatic daily checks",
            "Every day after 12:00 Swedish time, three checks run on the live forecasts. Every alert opens an "
            "issue on GitHub. The limits come from the test year and were fixed before the checks started.")
    try:
        status = get_monitoring()
    except Exception as error:
        show_problem("The monitoring result could not be loaded from GitHub right now.", error)
        return
    labels = {"ok": "OK", "alert": "Alert", "waiting": "Waiting"}
    names = {"deadline": "Forecast before 12:00", "error": "Live error", "drift": "Input drift"}
    checks = status["checks"]
    st.markdown('<div class="cards">'
                + "".join(card(names[name], labels[check["status"]], "", check["message"]) for name, check in checks.items())
                + '</div>', unsafe_allow_html=True)
    checked = pd.Timestamp(status["checked_at_utc"]).tz_convert(live_data.TZ)
    st.caption(f"Last check: {checked:%a %d %b %Y, %H:%M} Swedish time.")

    if "psi" in checks["drift"]:
        drift = pd.DataFrame({"input": list(checks["drift"]["psi"]), "psi": list(checks["drift"]["psi"].values())})
        drift = drift.sort_values("psi", ascending=False)
        section("Input drift", "Population stability index per input",
                f"The inputs of the last {checks['drift']['days']} forecast days compared with the same month in the "
                f"test year. Above {checks['drift']['limit']} counts as drift.")
        st.dataframe(drift, hide_index=True,
                     column_config={"input": "Input", "psi": st.column_config.NumberColumn("PSI", format="%.3f")})

    section("Run log", "The last 14 daily forecast runs",
            "Every run of the daily forecast adds one line: when it started, what it did and how long it took. "
            "A failed run is logged with its error message.")
    try:
        runs = get_run_log().head(14)
    except Exception as error:
        show_problem("The run log could not be loaded from GitHub right now.", error)
        return
    table = pd.DataFrame({
        "started": runs["started"].dt.strftime("%a %d %b, %H:%M"),
        "status": runs["status"],
        "forecast_day": runs.get("forecast_day"),
        "new_hours": runs.get("new_hours"),
        "retrained": runs.get("retrained"),
        "duration_s": runs["duration_s"],
        "error": runs.get("error"),
    })
    st.dataframe(table, hide_index=True, column_config={
        "started": "Started (Swedish time)", "status": "Status", "forecast_day": "Forecast day",
        "new_hours": "New hours", "retrained": "Retrained", "error": "Error",
        "duration_s": st.column_config.NumberColumn("Duration (s)", format="%.0f"),
    })

# The real prices of the last 30 days as a 3D landscape
def landscape_page():
    section("Price landscape", "The last 30 days in 3D",
            "Real SE3 prices for every hour of the last 30 days. Drag to turn the landscape, scroll to zoom, "
            "or press Rotate. Daily peaks in the morning and evening, and cheap nights and middays, stand out as ridges and valleys.")
    try:
        table = live_data.price_landscape(get_recent_prices())
        show_chart(landscape_chart(table), "landscape")
    except Exception as error:
        show_problem("The recent prices could not be loaded from elprisetjustnu.se right now. Please try again later.", error)


# The one-time evaluation on the test year
def test_results_page():
    section("Evaluation", "Results on the test year",
            "Average error (MAE) on 8,664 unseen hours from 1 October 2025 to 26 September 2026, with every model "
            "retrained each month. rel_mae compares with the weekly_naive baseline: below 1 is better.")
    try:
        results = get_test_results().sort_values("mae", ascending=False)
        colours = [CYAN if m == "lstm" else VIOLET if m.startswith("lgbm") else GREY for m in results["model"]]
        names = [MODEL_NAMES.get(m, m) for m in results["model"]]
        test_fig = go.Figure(go.Bar(x=results["mae"], y=names, orientation="h", marker_color=colours,
                                    text=[f"MAE {m:.3f}  |  rel_mae {r:.2f}" for m, r in zip(results["mae"], results["rel_mae"])],
                                    textposition="auto", hovertemplate="%{y}: MAE %{x:.3f} SEK/kWh<extra></extra>"))
        test_fig.update_xaxes(title="MAE (SEK/kWh)")
        style(test_fig, 380)
        test_fig.update_layout(hovermode="closest", showlegend=False, margin=dict(b=70))
        show_chart(test_fig, "test_results")
    except Exception as error:
        show_problem("The test results could not be loaded from GitHub right now.", error)


# Every MLflow run (exported from mlflow.db) and the monthly MAE on the test year
def experiments_page():
    section("Experiment tracking", "Every run, tracked with MLflow",
            "Every baseline, validation and test run was tracked with MLflow. This table is an export of the "
            "project's MLflow database: each run records the exact code version (click to open the commit on "
            "GitHub) and a fingerprint of the exact data it used, so every result can be traced and reproduced.")
    try:
        runs, monthly = get_mlflow_runs()
        table = runs.assign(model_name=[MODEL_NAMES.get(m, m) for m in runs["run_name"]],
                            commit=live_data.REPO_URL + "/commit/" + runs["git_commit"].astype(str))
        st.dataframe(
            table[["stage", "model_name", "mae", "rmse", "rel_mae", "hours", "data_fingerprint", "commit"]],
            hide_index=True,
            column_config={
                "stage": "Stage", "model_name": "Model", "hours": "Hours", "data_fingerprint": "Data fingerprint",
                "mae": st.column_config.NumberColumn("MAE", format="%.4f"),
                "rmse": st.column_config.NumberColumn("RMSE", format="%.4f"),
                "rel_mae": st.column_config.NumberColumn("rel_mae", format="%.4f"),
                "commit": st.column_config.LinkColumn("Code commit", display_text=r"/commit/([0-9a-f]{7})"),
            },
        )
        test_months = monthly[monthly["stage"] == "test"]
        lines = go.Figure()
        for name, rows_of_run in test_months.groupby("run_name", sort=False):
            colour = CYAN if name == "lstm" else VIOLET if name.startswith("lgbm") else GREY
            lines.add_trace(go.Scatter(x=rows_of_run["month"], y=rows_of_run["mae"], name=MODEL_NAMES.get(name, name),
                                       mode="lines+markers", line=dict(color=colour, width=3 if name == "lstm" else 1.5),
                                       hovertemplate="%{y:.3f} SEK/kWh"))
        lines.update_yaxes(title="MAE (SEK/kWh)")
        lines.update_xaxes(title="Month of the test year")
        show_chart(style(lines, 420), "mlflow_monthly")
        st.caption("Monthly MAE of every model on the test year, from the same MLflow runs. The full MLflow "
                   "interface runs locally; this page shows its exported data.")
    except Exception as error:
        show_problem("The MLflow runs could not be loaded from GitHub right now.", error)


# How the project works: how the model was built and chosen, and what runs every day
def how_it_works_page():
    section("How it works", "Building the model",
            "Several models were built and compared in exactly the same way. The best one on the validation "
            "year was chosen, then tested once on a year it had never been tuned on.")
    st.markdown('<div class="cards">'
                + card("1 &middot; Data", "Since Nov 2022", "", "hourly SE3 prices, Stockholm weather and Swedish holidays")
                + card("2 &middot; Inputs", "18", "features", "only what is known the morning before: recent prices, calendar, holidays, weather forecast")
                + card("3 &middot; Models", "6", "compared", "four simple baselines, LightGBM and an LSTM neural network")
                + card("4 &middot; Evaluation", "Walk-forward", "", "every month forecast by a model trained only on earlier data; the LSTM was best on the validation and the test year")
                + '</div>', unsafe_allow_html=True)
    st.markdown('<div class="cards">'
                + card("5 &middot; Explainability", "SHAP", "", "SHAP values and permutation importance show which inputs drive each forecast")
                + card("6 &middot; Tracking", "MLflow", "", "every run is saved with its settings, results, code commit and data fingerprint")
                + card("7 &middot; Testing", "CI", "", "automated tests run with GitHub Actions on every change to the code")
                + '</div>', unsafe_allow_html=True)

    section("How it works", "Running it every day",
            "The chosen model, the LSTM, runs every morning and is checked every day, fully automatically.")
    st.markdown('<div class="cards">'
                + card("8 &middot; Daily forecast", "06:05", "", "a timer starts GitHub Actions: new prices and a weather forecast, the LSTM retrained once a month, tomorrow's 24 hours forecast before the 12:00 deadline")
                + card("9 &middot; Live record", "Every hour", "", "each forecast is saved and later compared with the real price; only forecasts made before 12:00 count")
                + card("10 &middot; Monitoring", "12:40", "", "daily checks of the deadline, the live error and input drift; every alert opens a GitHub issue")
                + card("11 &middot; Run log", "Every run", "", "each run records what it did, how long it took and any error")
                + '</div>', unsafe_allow_html=True)

# Where the data comes from, the rules that keep the results honest, and links
def about_page():
    section("Data and sources", "Where the numbers come from",
            "All data is public, and every result can be reproduced from the code on GitHub.")
    st.markdown(f"""
<div class="cards">
{card("Prices", "SE3", "", "day-ahead spot prices in SEK/kWh from elprisetjustnu.se: the market price itself, before VAT, taxes and fees. Since October 2025 they are 15-minute prices, averaged per hour.")}
{card("Weather", "Open-Meteo", "", "Stockholm weather. Validation, test and live forecasts use weather forecasts, never the weather that really happened.")}
{card("Holidays", "Sweden", "", "Swedish public holidays, plus Midsommarafton, Julafton and Nyårsafton, when most workplaces are closed.")}
</div>
""", unsafe_allow_html=True)
    section("Honest evaluation", "The rules behind every number",
            "These rules make sure no result uses information that would not have been available in time.")
    st.markdown(f"""
<div class="cards">
{card("Deadline", "12:00", "", "only forecasts made before 12:00 Swedish time on the day before count in the live accuracy")}
{card("Inputs", "No leakage", "", "every input is known the morning before the forecast day")}
{card("Test year", "Kept apart", "", "all model choices were made on the validation year; the test year was only used for the final evaluation")}
{card("Tracking", "MLflow", "", "every run records its code commit and a fingerprint of its data")}
</div>
""", unsafe_allow_html=True)

    section("Quality and automation", "Tested and run automatically",
            "Continuous integration (CI) with GitHub Actions: on every push, GitHub installs the exact package "
            "versions from uv.lock on a clean Linux machine and runs all automated tests. The badges below show "
            "the live status of the latest runs.")
    st.markdown(f"""
<p>
  <a href="{live_data.REPO_URL}/actions/workflows/tests.yml" target="_blank"><img alt="tests" height="28"
    src="https://img.shields.io/github/actions/workflow/status/{live_data.REPO}/tests.yml?branch=main&style=for-the-badge&label=tests&logo=githubactions&logoColor=white"></a>
  <a href="{live_data.REPO_URL}/actions/workflows/daily_forecast.yml" target="_blank"><img alt="daily forecast" height="28"
    src="https://img.shields.io/github/actions/workflow/status/{live_data.REPO}/daily_forecast.yml?branch=main&style=for-the-badge&label=daily%20forecast&logo=githubactions&logoColor=white"></a>
</p>
<div class="cards">
{card("Tests", "pytest", "", "no future information in any input, clock changes, holidays, MLflow tracking, the daily forecast and every dashboard page")}
{card("CI", "Every push", "", "GitHub Actions runs the main tests and the PyTorch tests in separate processes, with the packages locked in uv.lock")}
{card("Daily run", "Every morning", "", "a timer starts the forecast workflow at 06:05 and 08:05, with GitHub's own schedule as a backup")}
</div>
""", unsafe_allow_html=True)

    section("Links", "Code and data", "")
    st.markdown(f"""
<div class="cards">
{card("Code", "GitHub", "", f'<a href="{live_data.REPO_URL}" target="_blank">{live_data.REPO_URL}</a>')}
{card("Live record", "forecast-data", "", f'<a href="{live_data.REPO_URL}/tree/forecast-data" target="_blank">every forecast and real price, updated each morning</a>')}
</div>
""", unsafe_allow_html=True)