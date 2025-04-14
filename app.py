from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta

app = Flask(__name__)

# -------------------------------------------------------------------
# Moduł symulujący pobieranie danych z SunnyPortal
# -------------------------------------------------------------------
class SunnyPortalDataFetcher:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        # W produkcji wykorzystaj requests.Session i odpowiednią autentykację.

    def fetch_data(self, start_date, end_date):
        """
        Funkcja symuluje pobieranie danych (produkowanej energii co godzinę)
        w okresie od start_date do end_date.
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        np.random.seed(42)
        production = []
        for dt in date_range:
            # Ustalanie profilu godzinowego: brak produkcji w nocy, szczyt w okolicach południa
            hourly_factor = max(0, np.sin((dt.hour - 6) / 12 * np.pi) + np.random.normal(0, 0.2))
            # Efekt sezonowy: wyższa produkcja latem, niższa zimą
            if dt.month in [6, 7, 8]:
                seasonal_factor = 1.5
            elif dt.month in [12, 1, 2]:
                seasonal_factor = 0.7
            else:
                seasonal_factor = 1.0
            production.append(max(0, hourly_factor * seasonal_factor * 5))  # Maksymalnie ok. 5 kWh
        data = pd.DataFrame({
            "timestamp": date_range,
            "production_kWh": production
        })
        return data

# -------------------------------------------------------------------
# Moduł analizy ekonomicznej – oblicza scenariusze: autokonsumpcja vs feed-in
# -------------------------------------------------------------------
class EconomicAnalyzer:
    def __init__(self, data):
        self.data = data.copy()
        self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
        self.data["hour"] = self.data["timestamp"].dt.hour
        self.data["month"] = self.data["timestamp"].dt.month

    def calculate_economics(self, cost_peak, feed_in_rate, grid_usage_rate):
        """
        Obliczenia:
          - Autokonsumpcja: oszczędność odpowiada cenie energii w godzinach szczytu.
          - Feed-in: przychód z eksportu minus koszt ponownego zakupu energii.
        """
        self.data["savings_autoconsumption"] = self.data["production_kWh"] * cost_peak
        self.data["revenue_feed_in"] = self.data["production_kWh"] * feed_in_rate
        self.data["cost_buy_energy"] = self.data["production_kWh"] * grid_usage_rate
        self.data["net_feed_in"] = self.data["revenue_feed_in"] - self.data["cost_buy_energy"]

        total_savings_auto = self.data["savings_autoconsumption"].sum()
        total_net_feed_in = self.data["net_feed_in"].sum()
        return total_savings_auto, total_net_feed_in

# -------------------------------------------------------------------
# Funkcja generująca wykres produkcji energii za pomocą Plotly
# -------------------------------------------------------------------
def create_production_chart(data):
    fig = px.line(data, x="timestamp", y="production_kWh", title="Produkcja energii [kWh]")
    fig.update_layout(xaxis_title="Czas", yaxis_title="Produkcja [kWh]")
    return fig.to_html(full_html=False)

# -------------------------------------------------------------------
# Funkcja generująca wykres analizy ekonomicznej
# -------------------------------------------------------------------
def create_economics_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["savings_autoconsumption"],
                             mode='lines', name='Oszczędności autokonsumpcji'))
    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["net_feed_in"],
                             mode='lines', name='Wynik feed-in'))
    fig.update_layout(title="Analiza ekonomiczna", xaxis_title="Czas", yaxis_title="Wartość [PLN]")
    return fig.to_html(full_html=False)

# -------------------------------------------------------------------
# Trasa główna – formularz wyboru dat
# -------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# -------------------------------------------------------------------
# Trasa wyników – przetwarza wybrany zakres dat oraz prezentuje wyniki analizy
# -------------------------------------------------------------------
@app.route("/results", methods=["POST"])
def results():
    # Pobieramy dane z formularza
    start_date_str = request.form.get("start_date")
    end_date_str = request.form.get("end_date")

    try:
        # Parsowanie dat (format: YYYY-MM-DD)
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        # Ustawiamy godzinę początkową na początek dnia
        start_date = start_date.replace(hour=0, minute=0, second=0)
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        # Ustawiamy godzinę końcową na ostatnią godzinę dnia
        end_date = end_date.replace(hour=23, minute=59, second=59)
    except Exception as e:
        return f"Błędny format dat: {e}"

    # Inicjalizujemy moduł pobierania danych
    # W tym przykładzie dane są symulowane; w produkcji podaj właściwe dane logowania.
    fetcher = SunnyPortalDataFetcher(username="twoj_username", password="twoje_haslo")
    production_data = fetcher.fetch_data(start_date, end_date)

    # Parametry analizy ekonomicznej – przykładowe wartości (PLN/kWh)
    cost_peak = 0.80      # koszt energii w godzinach szczytu
    feed_in_rate = 0.50   # stawka za eksport energii do sieci
    grid_usage_rate = 0.40  # koszt ponownego zakupu energii

    analyzer = EconomicAnalyzer(production_data)
    total_auto, total_feed_in = analyzer.calculate_economics(cost_peak, feed_in_rate, grid_usage_rate)

    # Generujemy wykresy do osadzenia w HTML
    production_chart = create_production_chart(production_data)
    economics_chart = create_economics_chart(analyzer.data)

    # Przekazujemy wyniki do szablonu
    return render_template(
        "results.html",
        total_auto=f"{total_auto:.2f}",
        total_feed_in=f"{total_feed_in:.2f}",
        production_chart=production_chart,
        economics_chart=economics_chart
    )

if __name__ == "__main__":
    app.run(debug=True)
