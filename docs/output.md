# Carmina Output Directory Documentation

## Overview

This document details the structure and contents of the output directory used by the Carmina project. The output directory contains various JSON files that store model outputs, debugging information, metrics, and logs generated during inference runs.

## Directory Structure

The output directory is located at:
```
data/outputs/
```

### Primary Subdirectories

- `logs/` - Contains detailed execution logs
- `metrics/` - Stores performance and usage metrics
- Root directory - Contains primary output JSON files and debug information

## JSON File Types and Structures

### Model Output Files

Files follow the naming pattern `output_[model-name].json` (e.g., `output_deepseek-r1.json`, `output_claude-3.7-sonnet.json`).

#### Structure

The structure may vary by model type. For entity identification models like `output_deepseek-v3.json`:

```json
[
  {
    "id": "DOCUMENT_ID.txt",
    "text": "Original unprocessed text content of the document",
    "identify": "Text with identified entities marked with [**entity**] brackets",
    "masked_text": "Text with entities replaced by category placeholders like [**CATEGORIA**]",
    "labels": [
      {
        "type": "ENTITY_TYPE",
        "text": "actual entity text",
        "start": 50,
        "end": 65
      },
      // Additional entities...
    ],
    "gt_raw_entities": [
      "entity1",
      "entity2",
      // List of raw entity strings
    ],
    "gt_masked_entities": [
      "ENTITY_TYPE1",
      "ENTITY_TYPE2",
      // List of entity category labels
    ],
    "identified_text": "Text with all identified entities marked with brackets",
    "entities_identified": [
      "entity1",
      "entity2",
      // List of all identified entity strings
    ],
    "anonymized_text": "Text with entities replaced by appropriate category labels",
    "entities_anonymized": [
      "ENTITY_TYPE1",
      "ENTITY_TYPE2",
      // List of anonymized entity categories
    ]
  }
]
```

### Debug Files

`loaded_records_debug.json` provides information about text processing results.

#### Structure

```json
[
  {
    "id": "DOCUMENT_ID.txt",
    "text": "Original unprocessed text content of the document",
    "identify": "Text with identified entities marked with [**entity**] brackets",
    "masked_text": "Text with entities replaced by category placeholders like [**CATEGORIA**]",
    "labels": [
      // Array that may contain entity labels (often empty in early processing stages)
    ]
  },
  // Additional records...
]
```

Each record represents a document being processed through the entity identification pipeline, showing both the original text and versions with different levels of entity processing.

## Logs Directory

The `logs/` directory contains timestamped log files following the pattern `carmina_YYYYMMDD_HHMMSS.log`. These files track:

- Inference requests and responses
- Error messages and exceptions
- System performance metrics
- API interactions with model providers
- Configuration information

Example log entry:
```
[2025-04-15 12:58:14] INFO: Starting inference with model deepseek-r1
[2025-04-15 12:58:15] DEBUG: Sending request to Azure endpoint
[2025-04-15 12:58:17] INFO: Response received, 456 tokens generated in 2.45 seconds
```

## Metrics Directory

The `metrics/` directory contains aggregated performance data:

### Token Usage Files

`token_usage_[timeframe].json`:
```json
{
  "period": "2025-04-15 to 2025-04-21",
  "models": {
    "deepseek-r1": {
      "input_tokens": 12500,
      "output_tokens": 28700,
      "total_tokens": 41200,
      "estimated_cost": 0.82
    },
    "claude-3.7-sonnet": {
      "input_tokens": 9800,
      "output_tokens": 15600,
      "total_tokens": 25400,
      "estimated_cost": 1.27
    }
  }
}
```

### Performance Metrics

`performance_metrics_[timeframe].json`:
```json
{
  "period": "2025-04-15 to 2025-04-21",
  "models": {
    "deepseek-r1": {
      "average_latency": 2.8,
      "median_latency": 2.5,
      "p95_latency": 4.2,
      "successful_requests": 245,
      "failed_requests": 3
    },
    "claude-3.7-sonnet": {
      "average_latency": 1.9,
      "median_latency": 1.7,
      "p95_latency": 3.1,
      "successful_requests": 198,
      "failed_requests": 1
    }
  }
}
```

## Working with Output Files

### Common Operations

- **Analyzing Results**: Parse the output JSON files to compare model performances
- **Debugging Issues**: Review log files for detailed information about errors or unexpected behaviors
- **Monitoring Usage**: Track token usage and costs through metrics files
- **Performance Tuning**: Use latency metrics to optimize system performance

### Best Practices

**Note:** These practices should be verified for effectiveness in your specific deployment environment.

1. Regularly archive output files to prevent disk space issues
2. Implement version control for output directories to track changes over time
3. Use the debug files to identify and address data processing issues quickly
4. Reference log files when troubleshooting API connection problems
5. Analyze metrics files to optimize cost and performance

## Example Usage Scenarios

### Scenario 1: Comparing Model Outputs

```python
import json

# Load outputs from different models
with open('data/outputs/output_deepseek-r1.json', 'r') as f:
    deepseek_outputs = json.load(f)

with open('data/outputs/output_claude-3.7-sonnet.json', 'r') as f:
    claude_outputs = json.load(f)

# Compare outputs for the same input
for ds_output in deepseek_outputs:
    record_id = ds_output['id']
    for cl_output in claude_outputs:
        if cl_output['id'] == record_id:
            print(f"Record {record_id}:")
            print(f"DeepSeek output: {ds_output['output']['completion'][:100]}...")
            print(f"Claude output: {cl_output['output']['completion'][:100]}...")
            print("-" * 50)
            break
```

### Scenario 2: Cost Analysis

```python
import json
import matplotlib.pyplot as plt

# Load token usage metrics
with open('data/outputs/metrics/token_usage_monthly.json', 'r') as f:
    usage_data = json.load(f)

# Plot cost comparison
models = []
costs = []

for model, data in usage_data['models'].items():
    models.append(model)
    costs.append(data['estimated_cost'])

plt.bar(models, costs)
plt.title('Monthly Cost by Model')
plt.xlabel('Model')
plt.ylabel('Estimated Cost ($)')
plt.savefig('model_cost_comparison.png')
```
