import os
import json
import dotenv
import logging
from src.carmina.data_sources.data_loader import load_dataset
from src.carmina.pipeline.anon_pipeline import AnonymizationPipeline
from src.carmina.llm.factory import LLMFactory
from src.carmina.metrics.recorder import MetricsRecorder
from src.carmina.metrics.timer import measure_time
from src.carmina.metrics.compare_line import extract_all_metrics

# Load environment variables
dotenv.load_dotenv()

class ModelExecutor:
    """
    Class that executes an anonymization model on a dataset.

    This class is responsible for loading the data, executing the selected
    anonymization model with the specified strategy, evaluating the results,
    and saving both the anonymized data and evaluation metrics.

    Attributes:
        model_name (str): Name of the model to use.
        anonymization_mode (str): Execution strategy for the model.
        input_path (str): Path to the input data file.
        output_dir (str): Directory where anonymized results will be saved
        metrics_dir (str): Directory where evaluation metrics will be saved.
    """
    def __init__(self, model_name, anonymization_mode, cloud_provider, input_path, output_dir, metrics_dir, debug_dir):
        """
        Initializes the ModelExecutor with the provided parameters.

        Args:
            model_name (str): Name of the model to use.
            anonymization_mode (str): Execution strategy for the model.
            cloud_provider (str): Cloud service provider for the model execution.
            input_path (str): Path to the input data file.
            output_dir (str): Directory where anonymized results will be saved.
            metrics_dir (str): Directory where evaluation metrics will be saved.
        """
        self.model_name = model_name
        self.anonymization_mode = anonymization_mode
        self.cloud_provider = cloud_provider
        self.input_path = input_path
        self.output_dir = output_dir
        self.metrics_dir = metrics_dir
        self.debug_dir = debug_dir

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
        print(f"\n🚀 Executing with model: {self.model_name} and anonymization_mode: {self.anonymization_mode}")
        records = load_dataset(self.input_path)

        # Save loaded records to a JSON file for inspection
        debug_path = os.path.join(self.debug_dir, "loaded_records_debug.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        recorder = MetricsRecorder(self.model_name)
        
        llm = LLMFactory.create(
            model_name=self.model_name,
            cloud_provider=self.cloud_provider,
            strategy_kwargs={"anonymization_mode": self.anonymization_mode},
            provider_kwargs={}
        )

        pipeline = AnonymizationPipeline(llm)

        with measure_time("anon_pipeline_time", recorder):
            anonymized_records = pipeline.run(records)

        # Capture execution time for inclusion in results
        execution_time = recorder.get_current_metrics().get("anon_pipeline_time", 0)

        # ======== SAVE ONLY NON-EMPTY ANONYMIZED_TEXT =========
        output_dir = os.getenv("OUTPUT_DIR")
        if not output_dir:
            raise ValueError("OUTPUT_DIR not specified in .env file")

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save only the 'anonymized_text' field for each record if not empty
        #saved_count = 0
        #for record in anonymized_records:
        #    anonymized_text = record.get("anonymized_text", "")
        #    if anonymized_text.strip():  # Only save if not empty or whitespace
        #        record_id = record.get("id", "unknown")
        #        output_path = os.path.join(output_dir, f"{record_id}.txt")
        #        with open(output_path, "w", encoding="utf-8") as f:
        #            f.write(anonymized_text)
        #        saved_count += 1
#
        #print(f"✅ {saved_count} anonymized_text files saved to {output_dir} (non-empty only)")
        # ======== END SAVE ONLY NON-EMPTY ANONYMIZED_TEXT =========

        debug_path = os.path.join(self.debug_dir, f"output_{self.model_name}.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(anonymized_records, f, indent=2)

        print(f"✅ Anonymization completed. Results saved ")
        
        ground_truth_identity_texts = [entity.get("identify", None) for entity in anonymized_records]
        if None in ground_truth_identity_texts:
            logging.warning("Some records are missing the 'identify' key.")
            
        # Reconstruct texts from chunks if present, otherwise use direct value (backward compatibility)
        prediction_identity_texts = []
        prediction_texts = []
        
        for entity in anonymized_records:
            chunks = entity.get("chunks")
            # chunks may be a list of dicts, a string repr, or absent
            if isinstance(chunks, list) and chunks and isinstance(chunks[0], dict):
                pred_ident = " ".join([c.get("identified_text", "") for c in chunks])
                pred_anon = " ".join([c.get("anonymized_text", "") for c in chunks])
                prediction_identity_texts.append(pred_ident)
                prediction_texts.append(pred_anon)
            else:
                # Fallback: chunks absent, a string, or non-dict list
                prediction_identity_texts.append(entity.get("identify", entity.get("identified_text", "")))
                prediction_texts.append(entity.get("masked_text", entity.get("anonymized_text", "")))

        ground_truth_texts = [entity.get("masked_text", "") for entity in anonymized_records]
        filenames = [entity.get("id", "unknown") for entity in anonymized_records]
        languages = [self.get_language(entity.get("id", "")) for entity in anonymized_records] #TODO: improve software design

        
        recorder.record_all(
            extract_all_metrics(
                ground_truth_identity_texts=ground_truth_identity_texts,
                prediction_identity_texts=prediction_identity_texts,
                ground_truth_texts=ground_truth_texts,
                prediction_texts=prediction_texts,
                filenames=filenames,
                languages=languages,
            )
        )

        # Add execution time to the results
        recorder.current_metrics["execution_time"] = execution_time

        metrics_path = os.path.join(self.metrics_dir, f"results_{self.model_name}.json")
        recorder.export_to_json(metrics_path)

        print(f"✅ Model {self.model_name} completed. Results in {self.output_dir} and {metrics_path}")

        # Retrieve output directory from .env
        

    def get_language(self, id):
        """
        Get the language for a given file ID from CARMEN1_mappings.tsv
        
        Args:
            id (str): File ID (e.g., "CARMEN-I_CC_1.txt")
        
        Returns:
            str: Language code (es, bi, cat) or None if not found
        """
        import os
        import pandas as pd
        
        # Remove .txt extension if present
        if id.endswith(".txt"):
            id = id[:-4]
        
        # Path to the mapping file
        # Path to the mapping file is outside of 'src' directory
        mapping_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                       "data", "inputs", "CARMEN1_mappings.tsv")
        try:
            # Read the TSV file using pandas for efficiency
            df = pd.read_csv(mapping_file, sep='\t')
            
            # Find the matching row
            match = df[df['filename'] == id]
            
            if not match.empty:
                return match['language'].values[0]
            else:
                return None
                
        except Exception as e:
            print(f"Error accessing mapping file: {e}")
            return None
