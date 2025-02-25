import sys
sys.stdout.flush()
import time
import psutil
import csv
from datetime import datetime

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
import re
import json
from io import StringIO
import csv
import pandas as pd 

PATH = './data/carmen/'

# TODO: Move this function on utils.py
def read_text(filename=None):
    with open(filename, 'r') as archivo:
        return archivo.read()


def output_to_csv(data):
    if data.strip() == 'None':
        return None

    datos_io = StringIO(data)
    lector_csv = csv.reader(datos_io, delimiter=',', quotechar='"', skipinitialspace=True)
    filas  = []
    for fila in lector_csv:
        filas.append(fila)

    # Pequeño procesamiento para eliminar los espacios en blanco que se generan despues de cada entidad
    df = pd.DataFrame(filas)
    df[0] = list(map(lambda x: x.rstrip(), df[0]))
    df[1] = list(map(lambda x: x.rstrip(), df[1]))
    
    return df

def save_time_to_file(block, start_time):
    end_time_1 = time.time()
    execution_time = end_time_1 - start_time
    filename = f"./data/metrics/time/{block}.txt"
    with open(filename, 'a') as file:
        file.write(f"Execution time of {block}: {execution_time} seconds\n")

def store_text(text, filename, name_model):
    folder_path = f"./data/anon/raw/{name_model}"
    file_path = os.path.join(folder_path, f"{filename}")
    os.makedirs(folder_path, exist_ok=True)
    with open(file_path, 'w') as file:
        file.write(text)

def dataframe_to_dict(dataframe):
    return dict(zip(dataframe.iloc[:, 0].str.rstrip(), dataframe.iloc[:, 1].str.rstrip()))


def post_processing_replace_text(text, dataframe):
    original_dict = dataframe_to_dict(dataframe)
    # for d in dictionary_list:
    #         combined_dictionary.update(d)
    def replacement_match(match):
        keyword = match.group(1)
        return original_dict.get(keyword, keyword)
    
    modified_text = re.sub(r'\[\*\*(.*?)\*\*\]', replacement_match, text)
    return modified_text


def post_processing_replace_text_csv(text, df):
    # Comprobar que el DataFrame tiene dos columnas
    if df.shape[1] != 2:
        raise ValueError("El DataFrame debe tener exactamente dos columnas.")

    # Crear un diccionario a partir del DataFrame
    combined_dictionary = dict(zip(df[1], df[0]))

    # Función de reemplazo
    def replacement_match(match):
        keyword = match.group(1)
        return combined_dictionary.get(keyword, keyword)  # Si no se encuentra, devuelve la palabra original

    # Realiza la sustitución en el texto
    modified_text = re.sub(r'\[\*\*(.*?)\*\*\]', replacement_match, text)
    return modified_text


def first_iteration(metrics, filename, llm, name_model):
    context = LLMContext(llm)
    data = {}
    data["system"] = read_text("prompts/system_prompt1.txt")
    data["user"] = read_text(f'{PATH}txt/replaced/{filename}')
    print("before")
    text_generated = context.generate_response(data)
    print(" ==finish call==")
    print(text_generated)
    store_text(text_generated, filename, "first/" + name_model)
    ground_truth = read_text(f'{PATH}/masked/{filename}')
    metrics.set_filename(filename)
    metrics.calculate(ground_truth, text_generated)
    metrics.store_metrics()
    return metrics, text_generated

# TODO: implement logic when there is no labels to classify (input_text = '')
def second_iteration(metrics_second, text_generated, llm, name_model, filename):
    prompt_filename = 'prompts/system_prompt2_beta.txt'

    # Process text_generated to create a list of labels
    labels = re.findall(r'\[\*\*(.*?)\*\*\]', text_generated)
    labels = list(map(lambda x: f'"{x}"', labels))
    input_text = '\n'.join(labels)

    prompt = {"system": read_text(prompt_filename), "user": input_text}
    context = LLMContext(llm)
    output = context.generate_response(prompt)

    df = output_to_csv(output)

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
    ground_truth = read_text(f'{PATH}txt/masked/{filename}')
    text_generated = post_processing_replace_text(text_generated, dictionary)
    metrics_thrid.set_filename(filename)
    metrics_thrid.calculate(
        ground_truth, 
        text_generated, 
        classification_bool = False
        )
    metrics_thrid.store_metrics()
    store_text(text_generated, filename, "thrid/" + name_model+"_3rd")
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
            print("first")
            metrics, text_generated = first_iteration(metrics, filename, llm, name_model)
            print("gere")
            dataframe_label = second_iteration(metrics_second, text_generated, llm, name_model, filename)
            metrics_thrid = third_iteration(metrics_thrid, text_generated, dataframe_label, name_model, filename)
            # dictionary = second_iteration_hips(None, text_generated, llm, name_model, filename)
            # print("hello")
            # _ = third_iteration_hips(None, text_generated, dictionary, name_model, filename)
            # print("hello")
            # counter += 1
            # if counter == 11:
        except Exception as e:
            print(filename)
            print(e)
    metrics.save_metrics()
    metrics_second.save_metrics()
    metrics_thrid.save_metrics()
    save_time_to_file(name_model, start_time)

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
