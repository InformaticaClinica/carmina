import os
import json
from data_sources.data_loader import load_dataset
#from pipeline.anon_pipeline import AnonymizationPipeline
#from llm.factory import get_llm_strategy
#from metrics.recorder import MetricsRecorder
#from metrics.timer import measure_time
#from metrics.evaluator import evaluate_identification, evaluate_substitution

class ModelExecutor:
    """
    Class that executes an anonymization model on a dataset.

    This class is responsible for loading the data, executing the selected
    anonymization model with the specified strategy, evaluating the results,
    and saving both the anonymized data and evaluation metrics.

    Attributes:
        model_name (str): Name of the model to use.
        strategy (str): Execution strategy for the model.
        input_path (str): Path to the input data file.
        output_dir (str): Directory where anonymized results will be saved
        metrics_dir (str): Directory where evaluation metrics will be saved.
    """
    def __init__(self, model_name, strategy, input_path, output_dir, metrics_dir):
        """
        Initializes the ModelExecutor with the provided parameters.

        Args:
            model_name (str): Name of the model to use.
            strategy (str): Execution strategy for the model.
            input_path (str): Path to the input data file.
            output_dir (str): Directory where anonymized results will be saved.
            metrics_dir (str): Directory where evaluation metrics will be saved.
        """
        self.model_name = model_name
        self.strategy = strategy
        self.input_path = input_path
        self.output_dir = output_dir
        self.metrics_dir = metrics_dir

    def execute(self):
        """
        Executes the complete anonymization and evaluation process.

        This method performs the following operations:
        1. Loads the input data.
        2. Initializes a metrics recorder.
        3. Retrieves the selected LLM strategy.
        4. Creates and executes the anonymization pipeline.
        5. Saves the anonymized records to the output directory.
        6. Evaluates the results by comparing true labels with predicted labels.
        7. Records and exports the evaluation metrics.

        Returns:
            None: Results are saved to the specified files.
        """
        print(f"\n🚀 Executing with model: {self.model_name} and strategy: {self.strategy}")
        records = load_dataset(self.input_path)
        # Save loaded records to a JSON file for inspection
        debug_path = os.path.join(self.output_dir, "loaded_records_debug.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        # print(f"Records loaded and saved to {debug_path} for inspection")
        # recorder = MetricsRecorder()

        # llm = get_llm_strategy(model_name=self.model_name, strategy=self.strategy)
        # pipeline = AnonymizationPipeline(llm)

        # with measure_time("anon_pipeline_time", recorder):
        #     anonymized_records = pipeline.run(records)

        # output_path = os.path.join(self.output_dir, f"output_{self.model_name}.json")
        # with open(output_path, "w", encoding="utf-8") as f:
        #     json.dump(anonymized_records, f, indent=2)

        # y_true = [r["labels"] for r in records]
        # y_pred = [r["predicted_labels"] for r in anonymized_records]

        # recorder.record_all({
        #     **evaluate_identification(y_true, y_pred),
        #     **evaluate_substitution(y_true, y_pred)
        # })

        # metrics_path = os.path.join(self.metrics_dir, f"results_{self.model_name}.json")
        # recorder.export_to_json(metrics_path)

        # print(f"✅ Model {self.model_name} completed. Results in {output_path} and {metrics_path}")
