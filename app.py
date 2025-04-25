from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pvlib
import plotly.express as px
import plotly.graph_objs as go

app = Flask(__name__)

LICZBA_MODULI      = 27
MODUL_MOC_W        = 190
INSTALLED_KWP      = LICZBA_MODULI * MODUL_MOC_W / 1000
MODULE_EFFICIENCY  = 0.149
PERFORMANCE_RATIO  = 0.80
MODULE_AREA        = 1.6
TILT_ANGLE         = 30
AZIMUTH_ANGLE      = 210

PRICE_PEAK         = 0.70
PRICE_GRID         = 0.80
NET_METERING_RATIO = 0.80
COST_PER_KWP       = 3500

LATITUDE  = 51.0631
LONGITUDE = 17.0334

def parse_measurement_file(path, base_date, col):
    if not pd.io.common.file_exists(path):
        return pd.DataFrame()
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split(";")
            head = parts[0].split()
            if len(head) < 2:
                continue
            t_str, d_str = head[0].rstrip("."), head[1].rstrip(".")
            try:
                day = int(d_str)
                tm = datetime.strptime(t_str, "%H:%M").time()
                ts = datetime.combine(base_date + timedelta(days=day-1), tm)
            except:
                continue
            val = np.nan
            if len(parts) > 1 and parts[1].strip():
                v = parts[1].strip().strip('"').replace(",", ".")
                try:
                    val = float(v)
                except:
                    val = np.nan
            rows.append({"timestamp": ts, col: val})
    return pd.DataFrame(rows)

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

def simulate_pv_poa(df_rad):
    if df_rad.empty:
        return pd.DataFrame()
    # ustawiamy timestamp jako indeks
    df = df_rad.set_index("timestamp").copy()
    solpos = pvlib.solarposition.get_solarposition(df.index, LATITUDE, LONGITUDE)
    irrads = pvlib.irradiance.get_total_irradiance(
        surface_tilt=TILT_ANGLE,
        surface_azimuth=AZIMUTH_ANGLE,
        ghi=df["ghi"],
        dni=df["dni"],
        dhi=df["dhi"],
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"]
    )["poa_global"]
    df["poa_irradiance"] = irrads.values
    df["production_kWh"] = (
        df["poa_irradiance"] * MODULE_AREA * MODULE_EFFICIENCY * PERFORMANCE_RATIO
        / 1000.0 * LICZBA_MODULI
    )
    df = df.reset_index()
    df["date"] = df.timestamp.dt.date
    df["hour"] = df.timestamp.dt.hour
    return df[["timestamp","date","hour","production_kWh"]]

def analyze_week(sp_df, pv_df):
    sp_daily  = sp_df.groupby("date")["production_kWh"].sum().reset_index()
    pv_daily  = pv_df.groupby("date")["production_kWh"].sum().reset_index()
    avg_hour  = pv_df.groupby("hour")["production_kWh"].mean().reset_index()

    econ = sp_df.copy()
    econ["save_auto"]  = econ.production_kWh * PRICE_PEAK
    econ["credit_kWh"] = econ.production_kWh * NET_METERING_RATIO
    econ["value_feed"] = econ.credit_kWh * PRICE_GRID
    econ["net_feed"]   = econ.value_feed

    total_auto = econ.save_auto.sum()
    total_feed = econ.net_feed.sum()
    peak = econ[econ.hour.between(10,14)]["production_kWh"].sum()
    off  = econ.production_kWh.sum() - peak

    days      = (econ.timestamp.max() - econ.timestamp.min()).days + 1
    annual_sav = total_auto/days*365
    payback   = (INSTALLED_KWP * COST_PER_KWP) / annual_sav if annual_sav>0 else np.nan

    return sp_daily, pv_daily, avg_hour, econ, total_auto, total_feed, peak, off, payback


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
        mode='lines+markers', name='Pomiar SunnyPortal'
    ))
    fig.add_trace(go.Scatter(
        x=pv_df.timestamp, y=pv_df.production_kWh,
        mode='lines+markers', name='Symulacja POA'
    ))
    fig.update_layout(
        title='Pomiar vs Symulacja POA',
        xaxis_title='Czas', yaxis_title='Energia [kWh]'
    )
    return fig.to_html(full_html=False)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results", methods=["POST"])
def results():
    try:
        start = datetime.strptime(request.form["start_date"], "%Y-%m-%d")
        end   = datetime.strptime(request.form["end_date"],   "%Y-%m-%d")
    except:
        return "<p>Błędny format dat</p>"
    if (end - start).days != 6:
        return "<p>Proszę wybrać dokładnie 7 kolejnych dni.</p>"
    end = end + timedelta(hours=23, minutes=59)

    sp_df = fetch_sunnyportal(start, end)
    rad   = fetch_radiation_forecast(start, end)
    pv_df = simulate_pv_poa(rad)
    sp_d, pv_d, avg_h, econ_df, total_auto, total_feed, peak, off, payback = analyze_week(sp_df, pv_df)

    return render_template("results.html",
        comp_chart    = plot_comparison(sp_df, pv_df),
        sp_chart      = plot_bar(sp_d,   "date","production_kWh","Pomiary PV wg dnia","Dzień","kWh"),
        pv_chart      = plot_bar(pv_d,   "date","production_kWh","Symulacja POA wg dnia","Dzień","kWh"),
        profile_chart = plot_line(avg_h,"hour","production_kWh","Średni profil godzinowy PV","kWh"),
        peak_pie      = plot_pie(["10–14","Pozostałe"], [peak,off], "Godziny dla bezrobotnych"),
        save_line     = plot_line(econ_df, "timestamp","save_auto","Oszczędności autokonsumpcji","PLN"),
        feed_line     = plot_line(econ_df, "timestamp","net_feed","Przychód z net-metering","PLN"),
        INSTALLED_KWP = INSTALLED_KWP,
        TOTAL_AUTO    = total_auto,
        TOTAL_FEED    = total_feed,
        PAYBACK       = payback,
        NET_RATIO     = int(NET_METERING_RATIO*100)
    )

if __name__ == "__main__":
    app.run(debug=True)
