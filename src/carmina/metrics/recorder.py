import json
import os
import pandas as pd
from typing import Dict, Any, List, Optional


class MetricsRecorder:
    """
    Records and manages evaluation metrics for text anonymization.
    
    This class provides a centralized way to record various metrics
    and export them to different formats. It maintains metrics for 
    the current record and a history of all recorded metrics.
    
    Attributes:
        current_metrics (Dict[str, Any]): Metrics for current record
        results (List[Dict[str, Any]]): History of all recorded metrics
    """
    
    def __init__(self, model_name: str = "Unassigned"):
        """
        Initialize an empty metrics recorder.
        
        Args:
            model_name: Name of the model being evaluated
        """
        self.current_metrics = {}
        self.results = []
        self.model_name = model_name
    
    def record(self, metric_name: str, value: Any) -> 'MetricsRecorder':
        """
        Record a single metric value.
        
        Args:
            metric_name: Name of the metric to record
            value: Value of the metric
            
        Returns:
            self: For method chaining
        """
        self.current_metrics[metric_name] = value
        return self
    
    def record_all(self, metrics_dict):
        """
        Record multiple metrics at once.
        
        Args:
            metrics_dict: Dictionary of metrics to record
            
        Returns:
            self: For method chaining
        """
        self.current_metrics = metrics_dict
        return self
        
    def set_filename(self, filename: str) -> 'MetricsRecorder':
        """
        Set the filename for the current metrics record.
        
        Args:
            filename: Name of the file being processed
            
        Returns:
            self: For method chaining
        """
        self.current_metrics["filename"] = filename
        return self
        
    def save_current(self) -> None:
        """
        Save current metrics to results history and reset current metrics.
        
        This is typically called after processing each file to store
        its metrics and prepare for the next file.
        """
        if self.current_metrics:
            self.results.append(self.current_metrics.copy())
            self.current_metrics = {}
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get the current metrics.
        
        Returns:
            Dict[str, Any]: Current metrics dictionary
        """
        return self.current_metrics
    
    def record_token_metrics(self, token_counts: Dict[str, int], anonymization_mode: str = None) -> 'MetricsRecorder':
        """
        Record token usage metrics.
        
        Args:
            token_counts: Dictionary with system, user, and total token counts
            anonymization_mode: The anonymization mode used (optional)
            
        Returns:
            self: For method chaining
        """
        prefix = f"{anonymization_mode}_" if anonymization_mode else ""
        
        self.current_metrics[f"{prefix}tokens_system"] = token_counts.get("system", 0)
        self.current_metrics[f"{prefix}tokens_user"] = token_counts.get("user", 0)
        self.current_metrics[f"{prefix}tokens_total"] = token_counts.get("total", 0)
        
        return self
    
    def record_cost_metrics(self, input_cost: float = 0.0, output_cost: float = 0.0, 
                           total_cost: float = None) -> 'MetricsRecorder':
        """
        Record cost metrics based on token usage.
        
        Args:
            input_cost: Cost for input tokens
            output_cost: Cost for output tokens  
            total_cost: Total cost (if None, calculated as input_cost + output_cost)
            
        Returns:
            self: For method chaining
        """
        self.current_metrics["cost_input"] = input_cost
        self.current_metrics["cost_output"] = output_cost
        self.current_metrics["cost_total"] = total_cost if total_cost is not None else input_cost + output_cost
        
        return self
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """
        Get all recorded metrics.
        
        Returns:
            List[Dict[str, Any]]: All recorded metrics
        """
        return self.results
    
    def export_to_json(self, filepath: str) -> None:
        """
        Export all recorded metrics to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save current metrics before exporting if they exist
        if self.current_metrics:
            self.save_current()
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
    
    def export_to_csv(self, filepath: str) -> None:
        """
        Export all recorded metrics to a CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save current metrics before exporting if they exist
        if self.current_metrics:
            self.save_current()
            
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
    
    def calculate_summary(self) -> Dict[str, Any]:
        """
        Calculate summary statistics across all results.
        
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not self.results:
            return {}
            
        # Save current metrics if they exist
        if self.current_metrics:
            self.save_current()
            
        # Convert results to DataFrame for easier aggregation
        df = pd.DataFrame(self.results)
        
        # Calculate numeric summaries
        numeric_cols = df.select_dtypes(include=['number']).columns
        summary = {}
        
        for col in numeric_cols:
            if col in ['precision', 'recall', 'f1', 'cosine_sim', 'inv_levenshtein', 'overall']:
                summary[f"{col}_mean"] = df[col].mean()
                summary[f"{col}_median"] = df[col].median()
                summary[f"{col}_min"] = df[col].min()
                summary[f"{col}_max"] = df[col].max()
                
        # Count by language if available
        if 'language' in df.columns:
            language_counts = df['language'].value_counts().to_dict()
            summary['language_counts'] = language_counts
            
            # Calculate metrics by language
            for lang in language_counts.keys():
                lang_df = df[df['language'] == lang]
                lang_metrics = {}
                
                for col in numeric_cols:
                    if col in ['precision', 'recall', 'f1', 'cosine_sim', 'inv_levenshtein', 'overall']:
                        lang_metrics[f"{col}_mean"] = lang_df[col].mean()
                
                summary[f"metrics_{lang}"] = lang_metrics
        
        return summary