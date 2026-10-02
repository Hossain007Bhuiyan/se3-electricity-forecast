# SE3 electricity price dashboard. Shows the newest forecast from the daily GitHub Actions run,
# how earlier forecasts compared with the real prices, a 3D view of recent prices, and the
# results of the test year. Run locally with:
#     uv run streamlit run dashboard/app.py

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import live_data

st.set_page_config(page_title="SE3 Electricity Price Forecast", layout="wide")

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

# Page style: dark background, animated title, glass-like cards with a slight 3D tilt that
# straighten up when the mouse is over them, and a short fade-in for cards and charts.
# "backwards" means the start animation only applies before it runs, so the hover effect still works.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, .stApp, .stMarkdown { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(circle at 15% 0%, #1e1b4b 0%, #0b1020 45%, #050814 100%); color: #e2e8f0; }
.block-container { padding-top: 2rem; max-width: 1250px; }
[data-testid="stHeader"] { background: transparent; }
.hero h1 { font-size: 2.8rem; font-weight: 800; margin: 0;
  background: linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6, #22d3ee); background-size: 300% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent; animation: shine 8s linear infinite; }
.hero p { color: #94a3b8; font-size: 1.05rem; margin: .4rem 0 1rem; }
.hero p.next { font-size: .85rem; margin: .6rem 0 0; }
@keyframes shine { to { background-position: 300% 0; } }
.live { display: inline-flex; align-items: center; gap: 10px; padding: 6px 14px; border-radius: 999px;
  background: rgba(34,197,94,.1); border: 1px solid rgba(34,197,94,.35); color: #86efac; font-size: .9rem; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: #22c55e; animation: pulse 2s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(34,197,94,.7); } 70% { box-shadow: 0 0 0 12px rgba(34,197,94,0); }
  100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); } }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 18px; perspective: 1000px; margin: 1.2rem 0 .6rem; }
.card { background: linear-gradient(145deg, rgba(255,255,255,.09), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.12); border-radius: 18px; padding: 18px 20px;
  box-shadow: 0 22px 40px -22px rgba(0,0,0,.9), inset 0 1px 0 rgba(255,255,255,.08);
  transform: rotateX(8deg); transform-origin: bottom; transition: transform .45s ease, box-shadow .45s ease;
  animation: rise .9s ease backwards; }
.card:hover { transform: rotateX(0deg) translateY(-6px) scale(1.02); box-shadow: 0 30px 60px -24px rgba(34,211,238,.45); }
.card:nth-child(2) { animation-delay: .12s; } .card:nth-child(3) { animation-delay: .24s; } .card:nth-child(4) { animation-delay: .36s; }
@keyframes rise { from { opacity: 0; transform: translateY(40px) rotateX(35deg); } }
.card .label { color: #94a3b8; font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; }
.card .value { font-size: 2rem; font-weight: 800; color: #f8fafc; margin-top: 6px; }
.card .unit { font-size: .95rem; font-weight: 600; color: #94a3b8; margin-left: 4px; }
.card .note { color: #94a3b8; font-size: .85rem; margin-top: 4px; }
.section { margin: 2.2rem 0 .4rem; }
.section .kicker { color: #22d3ee; font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; font-weight: 600; }
.section h2 { color: #f8fafc; font-size: 1.5rem; font-weight: 800; margin: .2rem 0; padding: 0; }
.section p { color: #94a3b8; margin: 0; }
[data-testid="stPlotlyChart"] { animation: fadein 1.2s ease backwards; border-radius: 18px;
  background: linear-gradient(145deg, rgba(255,255,255,.05), rgba(255,255,255,.01)); border: 1px solid rgba(255,255,255,.08); }
@keyframes fadein { from { opacity: 0; transform: translateY(20px); } }
.footer { color: #64748b; font-size: .85rem; margin-top: 2.5rem; border-top: 1px solid rgba(255,255,255,.08); padding-top: 1rem; }
.footer a { color: #22d3ee; }
</style>
""", unsafe_allow_html=True)


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


# Load the live record; without it the page cannot show anything
try:
    record, checked_at = get_record()
except Exception as error:
    st.error("The live forecasts could not be loaded from GitHub right now. Please try again in a few minutes.")
    st.caption(f"Details: {error}")
    st.stop()

day, rows = live_data.latest_day(record)
overview = live_data.day_overview(rows)
errors = live_data.daily_errors(record)
last_run = record["issued_at_utc"].max().tz_convert(live_data.TZ)

# Title, short description, when the last forecast was made and when the data was last checked
st.markdown(f"""
<div class="hero">
  <h1>SE3 Electricity Price Forecast</h1>
  <p>Tomorrow's hourly day-ahead prices for Sweden's SE3 zone (Stockholm region), forecast every morning
  by an LSTM neural network, before the 12:00 bidding deadline.</p>
  <span class="live"><span class="dot"></span>Live &middot; last forecast made {last_run:%a %d %b %Y, %H:%M:%S}
  Swedish time &middot; data checked {checked_at:%H:%M:%S}</span>
  <p class="next">A new forecast is made every morning at about 07:17 Swedish time (06:17 in winter);
  GitHub sometimes starts scheduled runs a little later. The data on this page is refreshed every 10 minutes.</p>
</div>
""", unsafe_allow_html=True)


# A live clock in Swedish time. It runs as a small script in the visitor's browser, so it ticks every
# second without reloading the page, and always shows Swedish time, wherever the visitor is.
# The script is wrapped in a function, and each rerun of the page replaces the old timer.
st.html("""
<div style="margin-top: 10px;">
  <span id="se3-clock" style="display: inline-flex; align-items: center; padding: 6px 14px; border-radius: 999px;
    background: rgba(34,211,238,.08); border: 1px solid rgba(34,211,238,.35); color: #e2e8f0;
    font-family: 'Inter', sans-serif; font-size: .9rem; font-variant-numeric: tabular-nums;"></span>
</div>
<script>
(() => {
  const format = new Intl.DateTimeFormat("en-GB", {timeZone: "Europe/Stockholm", weekday: "short", day: "2-digit",
    month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23"});
  const tick = () => {
    const clock = document.getElementById("se3-clock");
    if (clock) clock.textContent = "Now in Sweden: " + format.format(new Date());
  };
  clearInterval(window.se3ClockTimer);
  window.se3ClockTimer = setInterval(tick, 1000);
  tick();
})();
</script>
""", unsafe_allow_javascript=True)


# Key numbers of the newest forecast and the live accuracy so far
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
st.plotly_chart(day_chart(rows), theme=None, key="newest")

section("Live accuracy", "Forecast vs. reality, day by day",
        "Only forecasts made before 12:00 Swedish time on the day before count, so every number here "
        "was forecast before the prices were known. Lower is better.")
if errors.empty:
    st.info("No counted days yet. Each morning's forecast is checked once the real prices are published, "
            "so the first counted day appears after the second scheduled morning run.")
else:
    bars = go.Figure()
    labels = [d.strftime("%d %b") for d in errors["day"]]
    bars.add_trace(go.Bar(x=labels, y=errors["lstm_mae"], name="LSTM", marker_color=CYAN))
    bars.add_trace(go.Bar(x=labels, y=errors["weekly_naive_mae"], name="Baseline (weekly_naive)", marker_color=VIOLET))
    bars.update_yaxes(title="MAE (SEK/kWh)")
    st.plotly_chart(style(bars, 380), theme=None, key="live_errors")

# Any earlier day with real prices can be looked at in detail
known_days = sorted(record.loc[record["actual"].notna(), "time_local"].dt.date.unique(), reverse=True)
if known_days:
    chosen = st.selectbox("Look at a day", known_days, format_func=lambda d: d.strftime("%A %d %B %Y"))
    chosen_rows = record[record["time_local"].dt.date == chosen]
    counted = bool(chosen_rows["issued_before_noon"].all())
    st.caption("This forecast counts in the live accuracy." if counted
               else "This forecast was made after 12:00 Swedish time, so it is shown but does not count.")
    st.plotly_chart(day_chart(chosen_rows, 380), theme=None, key="chosen_day")

section("Price landscape", "The last 30 days in 3D",
        "Real SE3 prices for every hour of the last 30 days. Drag to turn the landscape, scroll to zoom, "
        "or press Rotate. Daily peaks in the morning and evening, and cheap nights and middays, stand out as ridges and valleys.")
try:
    table = live_data.price_landscape(get_recent_prices())
    st.plotly_chart(landscape_chart(table), theme=None, key="landscape")
except Exception as error:
    st.info("The recent prices could not be loaded from elprisetjustnu.se right now. Please try again later.")
    st.caption(f"Details: {error}")

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
    st.plotly_chart(test_fig, theme=None, key="test_results")
except Exception as error:
    st.info("The test results could not be loaded from GitHub right now.")
    st.caption(f"Details: {error}")

section("How it works", "From raw prices to a daily forecast",
        "The same four steps run every morning, fully automatically.")
st.markdown('<div class="cards">'
            + card("1 &middot; Data", "Since Nov 2022", "", "hourly SE3 prices and Stockholm weather")
            + card("2 &middot; Inputs", "18", "features", "only what is known the morning before: recent prices, calendar, holidays, weather forecast")
            + card("3 &middot; Model", "LSTM", "", "reads the last 168 hours of prices; retrained every month")
            + card("4 &middot; Live", "Daily", "", "runs on GitHub Actions before the 12:00 deadline and is checked against the real prices")
            + '</div>', unsafe_allow_html=True)

# Where the data comes from, and a link to the code
st.markdown(f"""
<div class="footer">
Prices are SE3 day-ahead spot prices in SEK/kWh from elprisetjustnu.se: the market price itself, before VAT, taxes and fees.
Weather forecasts from Open-Meteo. The model is retrained at the start of every month.
Code, method and full results: <a href="{live_data.REPO_URL}" target="_blank">{live_data.REPO_URL}</a>
</div>
""", unsafe_allow_html=True)