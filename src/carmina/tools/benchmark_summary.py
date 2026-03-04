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
                # data is a list; the first element is the full metrics dict
                # produced by extract_all_metrics: {"metrics": {...}, "files": [...]}
                metrics_entry = data[0] if isinstance(data, list) and data else data
                summary[model] = metrics_entry

                print(f"\n🔍 Métricas para {model}:")
                top_metrics = metrics_entry.get("metrics", {}) if isinstance(metrics_entry, dict) else {}
                for section, values in top_metrics.items():
                    print(f"  [{section}]")
                    if isinstance(values, dict):
                        for k, v in values.items():
                            if k == "execution_time":
                                v = format_execution_time(v)
                            print(f"    {k}: {v}")

                # Aggregate numeric leaf metrics across models
                for section, values in top_metrics.items():
                    if not isinstance(values, dict):
                        continue
                    for key, value in values.items():
                        if isinstance(value, (int, float)):
                            agg_key = f"{section}_{key}"
                            if agg_key not in aggregated:
                                aggregated[agg_key] = []
                            aggregated[agg_key].append(value)
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
