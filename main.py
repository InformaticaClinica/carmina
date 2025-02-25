import sys
sys.stdout.flush()
import time
import psutil
import csv
from datetime import datetime
import re
# Models
from llm import LLMContext
#from llm import ChatGPTModel, ChatGPTminiModel
# from llm import Llama3_1_90b_Model, Llama3_1_405b_Model, Llama3_1_11b_Model
# from llm import Haiku3Model, OpusModel
# from llm import BigMistralModel, Sonet3_5Model
# from llm import  Llama3_2_90b_Model
from llm import  Llama3_3_70b_Model
from metrics import Metrics, MetricsDict
import os
from utils.utils import Utils as U

PATH = './data/carmen/'

def first_iteration(metrics, filename, llm, name_model):
    context = LLMContext(llm)
    data = {}
    data["system"] = U.read_text("prompts/system_prompt1.txt")
    data["user"] = U.read_text(f'{PATH}txt/replaced/{filename}')
    text_generated = context.generate_response(data)
    U.store_text(text_generated, filename, "first/" + name_model)
    ground_truth = U.read_text(f'{PATH}/masked/{filename}')
    metrics.set_filename(filename)
    metrics.calculate(ground_truth, text_generated)
    metrics.store_metrics()
    return metrics, text_generated


def second_iteration(metrics_second, text_generated, llm, name_model, filename):
    prompt_filename = 'prompts/system_prompt2_beta.txt'

    labels = re.findall(r'\[\*\*(.*?)\*\*\]', text_generated)º
    labels = list(map(lambda x: f'"{x}"', labels))
    input_text = '\n'.join(labels)

    prompt = {"system": U.read_text(prompt_filename), "user": input_text}
    context = LLMContext(llm)
    output = context.generate_response(prompt)

    df = U.output_to_csv(output)

    metrics_second.set_filename(filename)
    metrics_second.calculate_classification_metrics(
        filename,
        df
    )

    directorio = f"./data/anon/raw/json/{name_model}"
    os.makedirs(directorio, exist_ok=True)
    output_path = os.path.join(directorio, f"{filename}.csv")
    df.to_csv(output_path)
    return df

def third_iteration(metrics_thrid, text_generated, dictionary, name_model, filename):
    ground_truth = U.read_text(f'{PATH}txt/masked/{filename}')
    text_generated = U.post_processing_replace_text(text_generated, dictionary)
    metrics_thrid.set_filename(filename)
    metrics_thrid.calculate(
        ground_truth, 
        text_generated, 
        classification_bool = False
        )
    metrics_thrid.store_metrics()
    U.store_text(text_generated, filename, "thrid/" + name_model+"_3rd")
    return metrics_thrid

def anonimized_loop(llm, name_model):
    start_time = time.time()
    # counter = 0
    metrics = Metrics(name_model)
    metrics_second = MetricsDict(name_model+"_2rd")
    metrics_thrid = Metrics(name_model+"_3rd")
    for filename in sorted(os.listdir(f'{PATH}txt/replaced/')):
        try:
            init_main()
            metrics, text_generated = first_iteration(metrics, filename, llm, name_model)
            dataframe_label = second_iteration(metrics_second, text_generated, llm, name_model, filename)
            metrics_thrid = third_iteration(metrics_thrid, text_generated, dataframe_label, name_model, filename)
        except Exception as e:
            print(filename)
            print(e)
    metrics.save_metrics()
    metrics_second.save_metrics()
    metrics_thrid.save_metrics()
    U.save_time_to_file(name_model, start_time)

def main():
    anonimized_loop(Llama3_3_70b_Model(), "Llama3_3_70b_Model")
    # anonimized_loop(Llama3_2_90b_Model(), "Llama3_2_3b_Model")
    # anonimized_loop(Llama3_1_90b_Model(), "Llama3_1_70b_Model")
    # anonimized_loop(Llama3_1_405b_Model(), "Llama3_1_405b_Model")
    # anonimized_loop(Llama3_1_11b_Model(), "Llama3_1_8b_Model")
    # anonimized_loop(Haiku3Model(), "Haiku3Model")
    # anonimized_loop(OpusModel(), "OpusModel")
    # anonimized_loop(Sonet3_5Model(), "Sonet3_5Model")
    # anonimized_loop(BigMistralModel(), "BigMistralModel")





# Umbral de temperatura en grados Celsius
TEMP_UMBRAL = 60.0  # Ajusta según el hardware
CSV_FILE = "time_on_waiting.csv"

def obtener_temperatura():
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

def print_temp(temp):
    if temp is not None:
        print(f"Temperatura actual: {temp}°C")

def registrar_tiempo(inicio, fin):
    """ Registra en un archivo CSV el tiempo de espera por alta temperatura """
    diferencia = (fin - inicio).total_seconds()

    # Crear el archivo CSV con encabezados si no existe
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

def init_main():
    temp = obtener_temperatura()
    print_temp(temp)

    if temp >= TEMP_UMBRAL:
        inicio_espera = datetime.now()
        while temp >= TEMP_UMBRAL:
            temp = obtener_temperatura()
            print_temp(temp)
            time.sleep(10)  # Esperar 10 segundos antes de volver a verificar
        fin_espera = datetime.now()
        registrar_tiempo(inicio_espera, fin_espera)

    print("Temperatura dentro del rango, reanudando ejecución.")

if __name__ == "__main__":
    main()
