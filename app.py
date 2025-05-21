from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pvlib
import plotly.express as px
import plotly.graph_objs as go

app = Flask(__name__)

# --------------------------------------------------------------------
# PARAMETRY SYSTEMU PV I CENNIK (Polska 2025)
# --------------------------------------------------------------------
LICZBA_MODULI      = 27
MODUL_MOC_W        = 190        # [W]
INSTALLED_KWP      = LICZBA_MODULI * MODUL_MOC_W / 1000  # [kWp]
MODULE_EFFICIENCY  = 0.149      # 14.9%
PERFORMANCE_RATIO  = 0.80       # 80% strat
MODULE_AREA        = 1.6        # [m²]
TILT_ANGLE         = 40         # ° nachylenie
AZIMUTH_ANGLE      = 135        # ° azymut (180° = południe)

PRICE_PEAK         = 0.70       # PLN/kWh – oszczędność autokonsumpcji
PRICE_GRID         = 0.80       # PLN/kWh – cena zakupu z sieci
NET_METERING_RATIO = 0.70       # 70% kredytu za oddaną energię
COST_PER_KWP       = 6000       # PLN/kWp – koszt instalacji

LATITUDE  = 51.1087
LONGITUDE = 17.0597

# --------------------------------------------------------------------
# 1) Parsowanie CSV SunnyPortal (_3=_power, _4=_voltage, _5=_current)
# --------------------------------------------------------------------
def parse_measurement_file(path, base_date, measurement_col):
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";")
                if len(parts) < 1:
                    continue

                time_day_str = parts[0].strip()
                tokens = time_day_str.split()
                if len(tokens) < 2:
                    continue

                time_str = tokens[0].rstrip(".")
                day_str  = tokens[1].rstrip(".")

                try:
                    day_num = int(day_str)
                except:
                    continue

                try:
                    time_obj = datetime.strptime(time_str, "%H:%M").time()
                except:
                    continue

                # tutaj nowa logika dla absolutnej daty
                year  = base_date.year
                month = base_date.month
                if day_num < base_date.day:
                    month += 1
                    if month == 13:
                        month = 1
                        year += 1
                full_dt = datetime(
                    year, month, day_num,
                    time_obj.hour, time_obj.minute
                )

                # parsowanie wartości jak dotąd...
                measurement = np.nan
                if len(parts) > 1:
                    val_str = parts[1].strip().strip('"').replace(",", ".")
                    try:
                        measurement = float(val_str) if val_str != "" else np.nan
                    except:
                        measurement = np.nan

                data.append({"timestamp": full_dt, measurement_col: measurement})
    except Exception as e:
        print(f"Błąd parsowania {path}: {e}")

    return pd.DataFrame(data)

def fetch_sunnyportal(start, end):
    d1, d2 = start.strftime("%d.%m.%Y"), end.strftime("%d.%m.%Y")
    p = parse_measurement_file(f"data/{d1}-{d2}_3.csv", start, "power")
    v = parse_measurement_file(f"data/{d1}-{d2}_4.csv", start, "voltage")
    c = parse_measurement_file(f"data/{d1}-{d2}_5.csv", start, "current")
    if p.empty or v.empty or c.empty:
        return pd.DataFrame()
    df = p.merge(v, on="timestamp").merge(c, on="timestamp")
    mask = (df.timestamp >= start) & (df.timestamp <= end)
    df = df.loc[mask].copy()
    df["production_kWh"] = df.power * 0.25 / 1000.0
    df["date"] = df.timestamp.dt.date
    df["hour"] = df.timestamp.dt.hour
    return df

# --------------------------------------------------------------------
# 2) Pobranie irradiance forecast z Open-Meteo (GHI, DHI, DNI)
# --------------------------------------------------------------------
def fetch_radiation_forecast(start, end):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "shortwave_radiation,diffuse_radiation,direct_normal_irradiance",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date":   end.strftime("%Y-%m-%d"),
        "timezone":   "UTC"
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(h["time"]),
        "ghi": h["shortwave_radiation"],
        "dhi": h["diffuse_radiation"],
        "dni": h["direct_normal_irradiance"]
    })
    df[["ghi","dhi","dni"]] = df[["ghi","dhi","dni"]].clip(lower=0)
    df["date"] = df.timestamp.dt.date
    df["hour"] = df.timestamp.dt.hour
    return df

# --------------------------------------------------------------------
# 3) Symulacja POA z pvlib
# --------------------------------------------------------------------
def simulate_pv_poa(df_rad):
    if df_rad.empty:
        return pd.DataFrame()
    df = df_rad.set_index("timestamp").copy()
    solpos = pvlib.solarposition.get_solarposition(df.index, LATITUDE, LONGITUDE)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=TILT_ANGLE,
        surface_azimuth=AZIMUTH_ANGLE,
        ghi=df["ghi"],
        dni=df["dni"],
        dhi=df["dhi"],
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"]
    )["poa_global"]
    df["poa_irradiance"] = poa.values
    df["production_kWh"] = (
        df.poa_irradiance * MODULE_AREA * MODULE_EFFICIENCY * PERFORMANCE_RATIO
        / 1000.0 * LICZBA_MODULI
    )
    df["production_kWh"] *= 0.5
    df = df.reset_index()
    df["date"] = df.timestamp.dt.date
    df["hour"] = df.timestamp.dt.hour
    return df

# --------------------------------------------------------------------
# 4) Dynamiczne, godzinowe skalowanie + analiza ekonomii
# --------------------------------------------------------------------
def analyze_week(sp_df, pv_df):

    sp_h = sp_df.groupby(["date","hour"])["production_kWh"].sum().reset_index(name="sp_sum")
    pv_h = pv_df.groupby(["date","hour"])["production_kWh"].sum().reset_index(name="pv_sum")

    s = pd.merge(pv_h, sp_h, on=["date","hour"])
    s["scale"] = s["sp_sum"] / s["pv_sum"]
    scale_map = s.set_index(["date","hour"])["scale"].to_dict()

    pv_df["production_kWh"] = pv_df.apply(
        lambda r: r["production_kWh"] * 0.3 * scale_map.get((r["date"],r["hour"]), 0.0),
        axis=1
    )

    sp_daily = sp_df.groupby("date")["production_kWh"].sum().reset_index()
    pv_daily = pv_df.groupby("date")["production_kWh"].sum().reset_index()
    avg_h    = pv_df.groupby("hour")["production_kWh"].mean().reset_index()

    econ = sp_df.copy()
    econ["save_auto"]  = econ.production_kWh * PRICE_PEAK
    econ["credit_kWh"] = econ.production_kWh * NET_METERING_RATIO
    econ["value_feed"] = econ.credit_kWh * PRICE_GRID
    econ["net_feed"]   = econ.value_feed

    total_auto = econ.save_auto.sum()
    total_feed = econ.net_feed.sum()
    peak = econ[econ.hour.between(10,14)]["production_kWh"].sum()
    off  = econ.production_kWh.sum() - peak

    days      = (econ.timestamp.max()-econ.timestamp.min()).days + 1
    annual_sav = total_auto/days*365
    payback   = (INSTALLED_KWP*COST_PER_KWP)/annual_sav if annual_sav>0 else np.nan

    return sp_daily, pv_daily, avg_h, econ, total_auto, total_feed, peak, off, payback

# --------------------------------------------------------------------
# 5) Wykresy i trasy Flask
# --------------------------------------------------------------------
def plot_line(df, x, y, title, ylab):
    if df.empty: return "<p>Brak danych</p>"
    fig = px.line(df, x=x, y=y, title=title)
    fig.update_yaxes(title=ylab)
    return fig.to_html(full_html=False)

def plot_bar(df, x, y, title, xl, yl):
    if df.empty: return "<p>Brak danych</p>"
    fig = px.bar(df, x=x, y=y, title=title, labels={x:xl, y:yl})
    return fig.to_html(full_html=False)

def plot_pie(labels, vals, title):
    fig = go.Figure(data=[go.Pie(labels=labels, values=vals, hole=0.4)])
    fig.update_layout(title=title)
    return fig.to_html(full_html=False)

def plot_comparison(sp_df, pv_df):
    if sp_df.empty or pv_df.empty:
        return "<p>Brak danych</p>"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sp_df.timestamp, y=sp_df.production_kWh,
        mode='lines+markers', name='SunnyPortal'
    ))
    fig.add_trace(go.Scatter(
        x=pv_df.timestamp, y=pv_df.production_kWh,
        mode='lines+markers', name='POA skalowana'
    ))
    fig.update_layout(title='Pomiar vs skalowana symulacja POA',
                      xaxis_title='Czas', yaxis_title='Energia [kWh]')
    return fig.to_html(full_html=False)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results", methods=["POST"])
def results():
    start = datetime.strptime(request.form["start_date"], "%Y-%m-%d")
    end   = datetime.strptime(request.form["end_date"],   "%Y-%m-%d")
    if (end-start).days != 6:
        return "<p>Proszę wybrać dokładnie 7 kolejnych dni.</p>"
    end = end + timedelta(hours=23, minutes=59)

    sp_df = fetch_sunnyportal(start, end)
    rad   = fetch_radiation_forecast(start, end)
    pv_df = simulate_pv_poa(rad)

    sp_d, pv_d, avg_h, econ_df, total_auto, total_feed, peak, off, payback = analyze_week(sp_df, pv_df)

    return render_template("results.html",
        comp_chart    = plot_comparison(sp_df, pv_df),
        sp_chart      = plot_bar(sp_d,  "date","production_kWh","SunnyPortal wg dnia","Dzień","kWh"),
        pv_chart      = plot_bar(pv_d,  "date","production_kWh","POA skalowana wg dnia","Dzień","kWh"),
        profile_chart = plot_line(avg_h,"hour","production_kWh","Średni profil godzinowy","kWh"),
        peak_pie      = plot_pie(["10–14","Pozostałe"], [peak,off], "Godziny dla bezrobotnych"),
        save_line     = plot_line(econ_df,"timestamp","save_auto","Oszczędności autokonsumpcji","PLN"),
        feed_line     = plot_line(econ_df,"timestamp","net_feed","Przychód z net-metering","PLN"),
        INSTALLED_KWP = INSTALLED_KWP,
        TOTAL_AUTO    = total_auto,
        TOTAL_FEED    = total_feed,
        PAYBACK       = payback,
        NET_RATIO     = int(NET_METERING_RATIO*100)
    )

if __name__ == "__main__":
    app.run(debug=True)
