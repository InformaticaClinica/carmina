import sys
sys.stdout.flush()

import time
import re
import os

from utils.utils import Utils 
from utils.temperature import Temperature
T = Temperature()
U = Utils()

# Models
from llm import LLMContext
# from llm import ChatGPTModel, ChatGPTminiModel
# from llm import Llama3_1_90b_Model, Llama3_1_405b_Model, Llama3_1_11b_Model
# from llm import Haiku3Model, OpusModel
# from llm import BigMistralModel, Sonet3_5Model
# from llm import  Llama3_2_90b_Model
# from llm import  Deepseek_70b
from llm import Deepseek_70b_azure

from metrics import Metrics, MetricsDict

from dotenv import load_dotenv
load_dotenv()

PATH = './data/carmen/'

def first_iteration(metrics, filename, llm, name_model, use_cache: bool = False):
    context = LLMContext(llm)
    data = {}
    data["system"] = U.read_text("prompts/system_prompt1.txt")
    data["user"] = U.read_text(f'{PATH}txt/replaced/{filename}')
    if use_cache and os.path.exists(f'./data/anon/raw/first/{name_model}/{filename}'):
        # Si existe el archivo, lo leemos
        # y lo guardamos en text_generated
        text_generated = U.read_text(f'./data/anon/raw/first/{name_model}/{filename}')
    else:
        text_generated = context.generate_response(data)
        U.store_text(text_generated, filename, "first/" + name_model)
    ground_truth = U.read_text(f'{PATH}txt/masked/{filename}')
    metrics.set_filename(filename)
    metrics.calculate(ground_truth, text_generated)
    metrics.store_metrics()
    return metrics, text_generated


def second_iteration(metrics_second, text_generated, llm, name_model, filename):
    prompt_filename = 'prompts/system_prompt2_beta.txt'

    labels = re.findall(r'\[\*\*(.*?)\*\*\]', text_generated)
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
    if not os.path.exists(f'./data/anon/raw/first/{name_model}'):
        os.makedirs(f'./data/anon/raw/first/{name_model}', exist_ok=True)
    start_time = time.time()
    metrics = Metrics(name_model)
    metrics_second = MetricsDict(name_model+"_2rd")
    metrics_thrid = Metrics(name_model+"_3rd")
    use_cache = True
    for filename in sorted(os.listdir(f'{PATH}txt/replaced/')):
        metrics, text_generated = first_iteration(metrics, filename, llm, name_model, use_cache)
        dataframe_label = second_iteration(metrics_second, text_generated, llm, name_model, filename)
        metrics_thrid = third_iteration(metrics_thrid, text_generated, dataframe_label, name_model, filename)
    metrics.save_metrics()
    metrics_second.save_metrics()
    metrics_thrid.save_metrics()
    U.save_time_to_file(name_model, start_time)

def main():
    anonimized_loop(Deepseek_70b_azure(), "Deepseek_70b")
    # anonimized_loop(Llama3_2_90b_Model(), "Llama3_2_3b_Model")
    # anonimized_loop(Llama3_1_90b_Model(), "Llama3_1_70b_Model")
    # anonimized_loop(Llama3_1_405b_Model(), "Llama3_1_405b_Model")
    # anonimized_loop(Llama3_1_11b_Model(), "Llama3_1_8b_Model")
    # anonimized_loop(Haiku3Model(), "Haiku3Model")
    # anonimized_loop(OpusModel(), "OpusModel")
    # anonimized_loop(Sonet3_5Model(), "Sonet3_5Model")
    # anonimized_loop(BigMistralModel(), "BigMistralModel")


if __name__ == "__main__":
    main()
