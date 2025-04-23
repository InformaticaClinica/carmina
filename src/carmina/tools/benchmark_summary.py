import os
import json

class BenchmarkSummary:
    def __init__(self, models, metrics_dir):
        self.models = models
        self.metrics_dir = metrics_dir

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
