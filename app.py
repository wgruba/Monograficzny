from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# ====================================================================
# Parametry instalacji – specyfikacja paneli
# ====================================================================
LICZBA_MODULI = 27
MODUL_MOC_W = 190  # Moc pojedynczego modułu [W]
INSTALLED_CAPACITY_KWP = (LICZBA_MODULI * MODUL_MOC_W) / 1000  # kWp (np. ok. 5.13 kWp)
MODULE_EFFICIENCY = 0.149  # Sprawność modułu (14,9%)
PERFORMANCE_RATIO = 0.80  # Ogólny współczynnik strat (80%)
MODULE_AREA = 1.6  # Powierzchnia modułu [m²] – wartość przybliżona


# ====================================================================
# Funkcja pomocnicza: parsowanie pliku z danymi pomiarowymi
# ====================================================================
def parse_measurement_file(file_path, base_date, measurement_col):
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";")
                if len(parts) < 1:
                    continue
                # Pierwsze pole, np. "08:15. 1"
                time_day_str = parts[0].strip()
                tokens = time_day_str.split()
                if len(tokens) < 2:
                    continue

                time_str = tokens[0].strip()
                day_str = tokens[1].strip().replace(".", "")

                # Wyznacz numer dnia
                try:
                    day_num = int(day_str)
                except:
                    day_num = 1

                # Usuń kropkę z końca, jeśli jest
                if time_str.endswith("."):
                    time_str = time_str[:-1]

                try:
                    time_obj = datetime.strptime(time_str, "%H:%M").time()
                except Exception:
                    continue

                full_dt = datetime.combine(base_date + timedelta(days=day_num - 1), time_obj)

                measurement = np.nan
                if len(parts) > 1:
                    val_str = parts[1].strip()
                    if val_str:
                        if val_str.startswith("\"") and val_str.endswith("\""):
                            val_str = val_str[1:-1]
                        val_str = val_str.replace(",", ".")
                        try:
                            measurement = float(val_str) if val_str != "" else np.nan
                        except:
                            measurement = np.nan

                data.append({"timestamp": full_dt, measurement_col: measurement})

    except Exception as e:
        print(f"Błąd przy otwieraniu/parsowaniu pliku {file_path}: {e}")

    return pd.DataFrame(data)


# ====================================================================
# Klasa pobierania danych z SunnyPortal – dynamiczny wybór plików
# ====================================================================
class SunnyPortalDataFetcher:
    def __init__(self, username, password):
        self.username = username
        self.password = password

    def fetch_data(self, start_date, end_date):
        """
        Wczytuje dane z trzech plików CSV:
          - *_1.csv (moc)
          - *_2.csv (napięcie)
          - *_3.csv (prąd)

        Następnie je scala i oblicza 'production_kWh' dla 15-minutowych interwałów.
        """
        start_str = start_date.strftime("%d.%m.%Y")
        end_str = end_date.strftime("%d.%m.%Y")

        # Zmień ścieżki, jeżeli trzeba
        file_power = f"data/{start_str}-{end_str}_3.csv"
        file_voltage = f"data/{start_str}-{end_str}_4.csv"
        file_current = f"data/{start_str}-{end_str}_5.csv"

        df_power = parse_measurement_file(file_power, base_date=start_date, measurement_col="power")
        df_voltage = parse_measurement_file(file_voltage, base_date=start_date, measurement_col="grid_voltage")
        df_current = parse_measurement_file(file_current, base_date=start_date, measurement_col="grid_current")

        if df_power.empty or df_voltage.empty or df_current.empty:
            print("Brakuje danych w plikach CSV (pusty DataFrame).")
            return pd.DataFrame()

        # Łączymy
        df_merged = pd.merge(df_power, df_voltage, on="timestamp", how="inner")
        df_merged = pd.merge(df_merged, df_current, on="timestamp", how="inner")

        # Filtrowanie
        df_merged = df_merged[
            (df_merged["timestamp"] >= start_date) &
            (df_merged["timestamp"] <= end_date)
        ]

        # Produkcja w kWh (15 minut to 0.25 h)
        df_merged["production_kWh"] = (df_merged["power"] * 0.25) / 1000.0
        df_merged["grid_power_calculated"] = df_merged["grid_voltage"] * df_merged["grid_current"]

        return df_merged


# ====================================================================
# Pobieranie danych z NASA POWER API
# ====================================================================
def fetch_nasa_data(start_date, end_date, latitude=51.1087, longitude=17.0597):
    """
    Pobiera dane godzinowe promieniowania (ALLSKY_SFC_SW_DWN) z NASA POWER.
    Domyślnie współrzędne budynku D2 PWr (~51.1087, 17.0597).
    """
    base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }
    try:
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data_json = response.json()
            nasa_data = data_json["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
            df = pd.DataFrame(list(nasa_data.items()), columns=["timestamp", "irradiance"])
            # NASA zwraca klucz w formie YYYYMMDDHH
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H")
            return df
        else:
            print("Błąd połączenia z NASA POWER API, status code:", response.status_code)
            return None
    except Exception as ex:
        print("Wyjątek przy pobieraniu danych NASA:", ex)
        return None


# ====================================================================
# Symulacja produkcji PV na podstawie danych NASA + specyfikacji modułów
# ====================================================================
def simulate_pv_production(nasa_df):
    """
    1. Zerujemy ujemne wartości nasłonecznienia (noc -> -999, itp.).
    2. Zamieniamy W/m² na energię w kWh w danej godzinie (multiplikujemy przez czas=1h).
    3. Przeliczamy przez powierzchnię, sprawność i liczbę modułów.
    """
    # Ustawiamy wszystkie wartości < 0 na 0:
    nasa_df.loc[nasa_df["irradiance"] < 0, "irradiance"] = 0

    # Mamy godzinowe wartości irradiance w [W/m^2].
    # Dla 1 h: [W/m^2] * 1 h => [Wh/m^2].
    # 1 [W] = 1 [J/s], ale uproszczenie – NASA daje nam średnią godzinową w W/m^2.
    # Tutaj do kWh: / 1000 => kW i * 1 h => kWh.
    time_interval_hours = 1.0

    nasa_df["production_per_module"] = (
        nasa_df["irradiance"]          # [W/m^2]
        * MODULE_AREA                  # [m^2] => [W]
        * MODULE_EFFICIENCY
        * PERFORMANCE_RATIO
        * time_interval_hours          # 1 godzina
        / 1000.0                       # konwersja W->kW => Wh->kWh
    )
    # Łączna produkcja
    nasa_df["total_production"] = nasa_df["production_per_module"] * LICZBA_MODULI

    return nasa_df


# ====================================================================
# Moduł analizy ekonomicznej (bazujący na danych SunnyPortal)
# ====================================================================
class EconomicAnalyzer:
    def __init__(self, data):
        self.data = data.copy()
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self.data["hour"] = self.data["timestamp"].dt.hour
        self.data["month"] = self.data["timestamp"].dt.month

    def calculate_economics(self, cost_peak, feed_in_rate, grid_usage_rate):
        """
        cost_peak: cena energii kupowanej z sieci (PLN/kWh),
                   którą oszczędzamy przez autokonsumpcję
        feed_in_rate: stawka za sprzedaż nadwyżki (PLN/kWh)
        grid_usage_rate: hipotetyczny koszt odkupu (PLN/kWh)
        """
        # oszczędności autokonsumpcyjne
        self.data["savings_autoconsumption"] = self.data["production_kWh"] * cost_peak
        # przychód z oddawania do sieci
        self.data["revenue_feed_in"] = self.data["production_kWh"] * feed_in_rate
        # koszt, gdybyśmy tę energię musieli odkupić
        self.data["cost_buy_energy"] = self.data["production_kWh"] * grid_usage_rate
        # zysk feed-in = przychód - koszt
        self.data["net_feed_in"] = self.data["revenue_feed_in"] - self.data["cost_buy_energy"]

        total_savings_auto = self.data["savings_autoconsumption"].sum()
        total_net_feed_in = self.data["net_feed_in"].sum()
        return total_savings_auto, total_net_feed_in


# ====================================================================
# Funkcje generujące wykresy (Plotly)
# ====================================================================
def create_production_chart(data):
    if data.empty:
        return "<p>Brak danych do wykresu produkcji.</p>"
    fig = px.line(data, x="timestamp", y="production_kWh", title="Produkcja energii (dane SunnyPortal)")
    fig.update_layout(xaxis_title="Czas", yaxis_title="Produkcja [kWh]")
    return fig.to_html(full_html=False)


def create_nasa_chart(nasa_data):
    if nasa_data.empty:
        return "<p>Brak danych NASA do wykresu.</p>"
    fig = px.line(
        nasa_data,
        x="timestamp",
        y="irradiance",
        title="Promieniowanie słoneczne (NASA POWER)"
    )
    # Clamp osi Y do 0 od dołu
    min_val = max(0, nasa_data["irradiance"].min())
    max_val = nasa_data["irradiance"].max() * 1.1
    fig.update_yaxes(range=[min_val, max_val])
    fig.update_layout(xaxis_title="Czas", yaxis_title="Irradiance [W/m²]")
    return fig.to_html(full_html=False)


def create_spec_simulation_chart(spec_df):
    if spec_df.empty:
        return "<p>Brak danych do symulacji.</p>"
    fig = px.line(
        spec_df,
        x="timestamp",
        y="total_production",
        title="Symulacja produkcji PV (specyfikacja modułów)"
    )
    min_val = max(0, spec_df["total_production"].min())
    max_val = spec_df["total_production"].max() * 1.1
    fig.update_yaxes(range=[min_val, max_val])
    fig.update_layout(xaxis_title="Czas", yaxis_title="Produkcja [kWh]")
    return fig.to_html(full_html=False)


def create_economics_chart(data):
    if data.empty:
        return "<p>Brak danych do analizy ekonomicznej.</p>"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["timestamp"],
        y=data["savings_autoconsumption"],
        mode='lines', name='Oszczędności autokonsumpcji'
    ))
    fig.add_trace(go.Scatter(
        x=data["timestamp"],
        y=data["net_feed_in"],
        mode='lines', name='Wynik feed-in'
    ))
    fig.update_layout(
        title="Analiza ekonomiczna",
        xaxis_title="Czas",
        yaxis_title="Wartość [PLN]"
    )
    return fig.to_html(full_html=False)


# ====================================================================
# Trasy aplikacji (Flask)
# ====================================================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/results", methods=["POST"])
def results():
    start_date_str = request.form.get("start_date")
    end_date_str = request.form.get("end_date")
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").replace(hour=0, minute=0, second=0)
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    except Exception as e:
        return f"Błędny format dat: {e}"

    # Pobierz dane z plików CSV
    fetcher = SunnyPortalDataFetcher(username="twoj_username", password="twoje_haslo")
    sunny_data = fetcher.fetch_data(start_date, end_date)

    # NASA
    nasa_data = fetch_nasa_data(start_date, end_date)
    if nasa_data is None or nasa_data.empty:
        nasa_chart = "<p>Nie udało się pobrać danych NASA.</p>"
        spec_sim_chart = "<p>Brak symulacji PV na podstawie danych NASA.</p>"
        total_spec_production = None
    else:
        nasa_chart = create_nasa_chart(nasa_data)
        spec_sim_df = simulate_pv_production(nasa_data.copy())
        spec_sim_chart = create_spec_simulation_chart(spec_sim_df)
        total_spec_production = spec_sim_df["total_production"].sum()

    # Analiza ekonomiczna
    cost_peak = 0.80      # PLN/kWh – koszt kupowanej energii w szczycie
    feed_in_rate = 0.50   # PLN/kWh – stawka za oddaną energię
    grid_usage_rate = 0.40  # PLN/kWh – koszt ewentualnego odkupu
    analyzer = EconomicAnalyzer(sunny_data)
    total_auto, total_feed_in = 0, 0
    if not sunny_data.empty:
        total_auto, total_feed_in = analyzer.calculate_economics(cost_peak, feed_in_rate, grid_usage_rate)

    production_chart = create_production_chart(sunny_data)
    economics_chart = create_economics_chart(analyzer.data)

    # Tekstowe podsumowanie
    analysis_summary = f"""
    <h2>Podsumowanie szczegółowej analizy</h2>
    <p><strong>Zainstalowana moc:</strong> {INSTALLED_CAPACITY_KWP:.2f} kWp</p>
    """

    if total_spec_production is not None:
        analysis_summary += (
            f"<p><strong>Symulowana całkowita produkcja (dane NASA + spec. modułów):</strong> "
            f"{total_spec_production:.2f} kWh</p>"
        )

    analysis_summary += f"""
    <p><strong>Oszczędności autokonsumpcyjne (SunnyPortal):</strong> {total_auto:.2f} PLN</p>
    <p><strong>Wynik feed-in (SunnyPortal):</strong> {total_feed_in:.2f} PLN</p>
    """

    if sunny_data.empty:
        analysis_summary += "<p>Nie znaleziono danych z plików CSV lub są puste.</p>"

    return render_template(
        "results.html",
        production_chart=production_chart,
        economics_chart=economics_chart,
        nasa_chart=nasa_chart,
        spec_sim_chart=spec_sim_chart,
        analysis_summary=analysis_summary
    )


if __name__ == "__main__":
    app.run(debug=True)
