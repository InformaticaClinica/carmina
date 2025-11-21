import os
from dotenv import load_dotenv
from src.carmina.model_executor import ModelExecutor
from src.carmina.tools.benchmark_summary import BenchmarkSummary
from src.carmina.metrics.timer import measure_time
from src.carmina.metrics.recorder import MetricsRecorder

class BenchmarkRunner:
    """
    The BenchmarkRunner class is responsible for orchestrating the execution of benchmarks
    for various models. It loads configuration from environment variables, sets up necessary
    directories, and executes the benchmarking process for each specified model.

    Attributes:
        models (list): A list of model names to benchmark, loaded from the "MODELS" environment variable.
        anonymization__mode (str): The strategy to use during execution, loaded from the "STRATEGY" environment variable.
        input_path (str): The path to the input data file, loaded from the "INPUT" environment variable.
        output_dir (str): The directory where output files will be saved, loaded from the "OUTPUT_DIR" environment variable.
        metrics_dir (str): The directory where metrics files will be saved, loaded from the "METRICS_DIR" environment variable.
    """
    def __init__(self):
        load_dotenv()
        self.models = os.getenv("MODELS").split(",")
        self.anonymization_mode = os.getenv("ANONYMIZATION_MODE")
        self.cloud_provider = os.getenv("CLOUD_PROVIDER")
        self.input_path = os.getenv("INPUT_DIR")
        self.output_dir = os.getenv("OUTPUT_DIR")
        self.metrics_dir = os.getenv("METRICS_DIR")
        self.debug = os.getenv("DEBUG")
        self.debug_dir = os.getenv("DEBUG_DIR")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

    def run(self):
        """
        Executes the benchmarking process for a list of models.

        This method iterates through the provided models, creating a `ModelExecutor`
        instance for each model. The `ModelExecutor` is responsible for executing
        the model using the specified strategy, input path, output directory, and
        metrics directory. After executing all models, a `BenchmarkSummary` is
        generated to summarize the benchmarking results.

        Attributes:
            models (list): A list of model names to be benchmarked.
            anonymization_mode (str): The execution strategy to be used for the models.
            input_path (str): The path to the input data for the models.
            output_dir (str): The directory where the model outputs will be stored.
            metrics_dir (str): The directory where the benchmarking metrics will be stored.
        """
        benchmark_recorder = MetricsRecorder("benchmark")
        with measure_time("total_benchmark_time", benchmark_recorder):
            for model in self.models:
                executor = ModelExecutor(
                    model_name=model,
                    anonymization_mode=self.anonymization_mode,
                    cloud_provider=self.cloud_provider,
                    input_path=self.input_path,
                    output_dir=self.output_dir,
                    metrics_dir=self.metrics_dir,
                    debug_dir=self.debug_dir,
                )
                executor.execute()

        BenchmarkSummary(self.models, self.metrics_dir, benchmark_recorder).generate()
