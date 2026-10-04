# SE3 electricity price dashboard. This file draws the frame of every page: the title, the live
# badge, the clock, the navigation menu at the top and the footer. The pages themselves are in
# views.py, and the data comes from live_data.py. Run locally with:
#     uv run streamlit run dashboard/app.py

import streamlit as st

import live_data
import views

st.set_page_config(page_title="SE3 Electricity Price Forecast", layout="wide")

# Page style: dark background, animated title, glass-like cards with a slight 3D tilt that
# straighten up when the mouse is over them, and a short fade-in for cards and charts.
# "backwards" means the start animation only applies before it runs, so the hover effect still works.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, .stApp, .stMarkdown { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(circle at 15% 0%, #1e1b4b 0%, #0b1020 45%, #050814 100%); color: #e2e8f0; }
.block-container { padding-top: 2rem; max-width: 1250px; }
[data-testid="stHeader"] { background: rgba(11,16,32,.88); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(34,211,238,.25); }
[data-testid="stTopNavLink"], [data-testid="stTopNavSection"] { color: #e2e8f0; }
[data-testid="stTopNavPopoverBody"] { background: #111a33; border: 1px solid rgba(34,211,238,.25); border-radius: 12px; }
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
.card a { color: #22d3ee; }
.mobile-menu { display: none; }
[data-testid="stExpandSidebarButton"] { width: auto !important; padding: 6px 14px !important; border-radius: 999px !important;
  background: rgba(34,211,238,.15) !important; border: 1px solid #22d3ee !important; }
[data-testid="stExpandSidebarButton"] > * { display: none !important; }
[data-testid="stExpandSidebarButton"]::after { content: "☰  Menu"; color: #f8fafc; font-weight: 600; font-size: .9rem; white-space: nowrap; }
@media (max-width: 900px) {
  .mobile-menu { display: flex; gap: 8px; overflow-x: auto; margin-top: 14px; padding-bottom: 6px; -webkit-overflow-scrolling: touch; }
  .mobile-menu a { flex: 0 0 auto; padding: 8px 14px; border-radius: 999px; font-size: .85rem; text-decoration: none;
    color: #e2e8f0; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.14); }
  .mobile-menu a.active { background: rgba(34,211,238,.18); border-color: #22d3ee; color: #f8fafc; font-weight: 600; }
}
</style>
""", unsafe_allow_html=True)

# Title, short description, when the last forecast was made and when the data was last checked
record, checked_at = views.load_record()
last_run = record["issued_at_utc"].max().tz_convert(live_data.TZ)
st.markdown(f"""
<div class="hero">
  <h1>SE3 Electricity Price Forecast</h1>
  <p>Tomorrow's hourly day-ahead prices for Sweden's SE3 zone (Stockholm region), forecast every morning
  by an LSTM neural network, before the 12:00 bidding deadline.</p>
  <span class="live"><span class="dot"></span>Live &middot; last forecast made {last_run:%a %d %b %Y, %H:%M:%S}
  Swedish time &middot; page data refreshed {checked_at:%H:%M:%S}</span>
  <p class="next">A new forecast is made early every morning; the first forecast made before 12:00 counts.
  The data on this page is refreshed every 10 minutes.</p>
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

# The menu at the top: three drop-down menus, each with its pages
menu = {
    "Forecast": [
        st.Page(views.tomorrow_page, title="Tomorrow's forecast", url_path="forecast", default=True),
        st.Page(views.live_accuracy_page, title="Live accuracy", url_path="live-accuracy"),
        st.Page(views.landscape_page, title="Price landscape (3D)", url_path="price-landscape"),
    ],
    "Model": [
        st.Page(views.test_results_page, title="Test-year results", url_path="test-results"),
        st.Page(views.experiments_page, title="Experiment tracking (MLflow)", url_path="experiments"),
    ],
    "Project": [
        st.Page(views.how_it_works_page, title="How it works", url_path="how-it-works"),
        st.Page(views.about_page, title="Data and sources", url_path="data-and-sources"),
    ],
}
pages = st.navigation(menu, position="top")

# On phones, Streamlit folds the top menu into a side panel behind a small arrow in the corner,
# which is easy to miss. Phones therefore also get a row of page buttons under the title (swipe
# sideways for more); the CSS hides this row on larger screens, so computers look the same as before.
links = "".join(
    f'<a href="/{page.url_path}" target="_self" class="{"active" if page.url_path == pages.url_path else ""}">{page.title}</a>'
    for group in menu.values() for page in group
)
st.markdown(f'<nav class="mobile-menu">{links}</nav>', unsafe_allow_html=True)

pages.run()

# Where the data comes from, and a link to the code, at the bottom of every page
st.markdown(f"""
<div class="footer">
Prices are SE3 day-ahead spot prices in SEK/kWh from elprisetjustnu.se: the market price itself, before VAT, taxes and fees.
Weather forecasts from Open-Meteo. The model is retrained at the start of every month.
Code, method and full results: <a href="{live_data.REPO_URL}" target="_blank">{live_data.REPO_URL}</a>
</div>
""", unsafe_allow_html=True)