import psutil
from datetime import datetime
import csv
import time


# Umbral de temperatura en grados Celsius
TEMP_UMBRAL = 60.0  # Ajusta según el hardware
CSV_FILE = "time_on_waiting.csv"

class Temperature:

    def __init__(self):
        pass

    def obtener_temperatura(self):
        """ Obtiene la temperatura de la CPU """
        try:
            temperaturas = psutil.sensors_temperatures()
            if "coretemp" in temperaturas:
                return max(temp.current for temp in temperaturas["coretemp"])
            elif "cpu_thermal" in temperaturas:
                return temperaturas["cpu_thermal"][0].current
            else:
                print("No se pudo obtener la temperatura.")
                return None
        except AttributeError:
            print("El sistema no soporta la obtención de temperatura.")
            return None

    def print_temp(self, temp):
        if temp is not None:
            print(f"Temperatura actual: {temp}°C")

    def registrar_tiempo(self, inicio, fin):
        """ Registra en un archivo CSV el tiempo de espera por alta temperatura """
        diferencia = (fin - inicio).total_seconds()

        try:
            with open(CSV_FILE, "x", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["init", "end", "difference"])
        except FileExistsError:
            pass

        # Registrar la información en el CSV
        with open(CSV_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([inicio, fin, diferencia])

    def init_main(self):
        temp = self.obtener_temperatura()
        self.print_temp(temp)

        if temp >= TEMP_UMBRAL:
            inicio_espera = datetime.now()
            while temp >= TEMP_UMBRAL:
                temp = self.obtener_temperatura()
                self.print_temp(temp)
                time.sleep(10)  # Esperar 10 segundos antes de volver a verificar
            fin_espera = datetime.now()
            self.registrar_tiempo(inicio_espera, fin_espera)

        print("Temperatura dentro del rango, reanudando ejecución.")