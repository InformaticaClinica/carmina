import csv
from io import StringIO
import re
import pandas as pd 
import os
import time

class Utils:
    def __init__(self):
        pass
    def read_text(self, filename=None):
        with open(filename, 'r') as archivo:
            return archivo.read()
    
    def output_to_csv(self, data):
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
    

    def dataframe_to_dict(self, dataframe):
        return dict(zip(dataframe.iloc[:, 0].str.rstrip(), dataframe.iloc[:, 1].str.rstrip()))


    def post_processing_replace_text(self, text, dataframe):
        original_dict = self.dataframe_to_dict(dataframe)
        # for d in dictionary_list:
        #         combined_dictionary.update(d)
        def replacement_match(match):
            keyword = match.group(1)
            return original_dict.get(keyword, keyword)
        
        modified_text = re.sub(r'\[\*\*(.*?)\*\*\]', replacement_match, text)
        return modified_text
    
    
    def store_text(self, text, filename, name_model):
        folder_path = f"./data/anon/raw/{name_model}"
        file_path = os.path.join(folder_path, f"{filename}")
        os.makedirs(folder_path, exist_ok=True)
        with open(file_path, 'w') as file:
            file.write(text)

    def save_time_to_file(self, block, start_time):
        end_time_1 = time.time()
        execution_time = end_time_1 - start_time
        filename = f"./data/metrics/time/{block}.txt"
        with open(filename, 'a') as file:
            file.write(f"Execution time of {block}: {execution_time} seconds\n")
