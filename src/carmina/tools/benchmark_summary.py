import os
import json

class BenchmarkSummary:
    def __init__(self, models, metrics_dir, benchmark_recorder=None):
        self.models = models
        self.metrics_dir = metrics_dir
        self.benchmark_recorder = benchmark_recorder

    def generate(self):
        print("\n📊 Generando resumen de benchmark...")
        summary = {}

        for model in self.models:
            path = os.path.join(self.metrics_dir, f"results_{model}.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    summary[model] = json.load(f)
            else:
                print(f"⚠️ No se encontró archivo de métricas para {model}")

        summary_path = os.path.join(self.metrics_dir, "benchmark_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"✅ Resumen generado en {summary_path}")

        # Print total execution time if available
        if self.benchmark_recorder:
            total_time = self.benchmark_recorder.get_current_metrics().get("total_benchmark_time")
            if total_time:
                print(f"⏱️ Total execution time: {total_time:.4f}s")
