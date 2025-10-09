import os
import json

def format_execution_time(seconds):
    """
    Format execution time from seconds to a human-readable string.
    Examples: "2h 30m", "45m", "30s"
    """
    if not isinstance(seconds, (int, float)):
        return str(seconds)

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m"
    else:
        return f"{secs}s"

class BenchmarkSummary:
    def __init__(self, models, metrics_dir, benchmark_recorder=None):
        self.models = models
        self.metrics_dir = metrics_dir
        self.benchmark_recorder = benchmark_recorder

    def generate(self):
        print("\n📊 Generando resumen de benchmark...")
        summary = {}
        aggregated = {}

        for model in self.models:
            path = os.path.join(self.metrics_dir, f"results_{model}.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    summary[model] = data

                # Print individual model metrics per file
                print(f"\n🔍 Métricas para {model}:")
                for file_result in summary[model]:
                    file_name = file_result.get('file', 'unknown')
                    print(f"  Archivo: {file_name}")
                    for key, value in file_result.items():
                        if key == 'file':
                            continue
                        if key == 'execution_time':
                            value = format_execution_time(value)
                        print(f"    {key}: {value}")

                # Aggregate metrics across all files for this model
                for file_result in data:
                    for key, value in file_result.items():
                        if key == 'file':
                            continue
                        if isinstance(value, (int, float)):
                            if key not in aggregated:
                                aggregated[key] = []
                            aggregated[key].append(value)
            else:
                print(f"⚠️ No se encontró archivo de métricas para {model}")

        # Calculate aggregated summary
        aggregated_summary = {}
        for key, values in aggregated.items():
            if key == "execution_time":
                aggregated_summary[key] = sum(values)
            else:
                aggregated_summary[key] = sum(values) / len(values) if values else 0

        # Format aggregated execution_time
        if "execution_time" in aggregated_summary:
            aggregated_summary["execution_time"] = format_execution_time(aggregated_summary["execution_time"])

        # Print aggregated summary
        print("\n📈 Resumen agregado:")
        for key, value in aggregated_summary.items():
            print(f"  {key}: {value}")

        summary_path = os.path.join(self.metrics_dir, "benchmark_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Resumen generado en {summary_path}")

        # Print total execution time if available
        if self.benchmark_recorder:
            total_time = self.benchmark_recorder.get_current_metrics().get("total_benchmark_time")
            if total_time:
                print(f"⏱️ Total execution time: {total_time:.4f}s")
