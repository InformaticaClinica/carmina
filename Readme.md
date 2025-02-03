# Interfaces
## LLM
### Strategy Design Pattern
The LLMContext interface has been designed using the Strategy design pattern. This pattern allows defining a family of algorithms, encapsulating them, and making them interchangeable. In our case, LLMContext enables dynamically switching the language model (LLM) used to generate responses without modifying the client code that uses it.

### Interface Usage
To use the LLMContext interface, simply instantiate it with a specific language model (llm). You can then generate a response by calling the generate_response method with the desired input data:

> context = LLMContext(llm)

> text_generated = context.generate_response(data)

### Handling Design Pattern
Within the Strategy pattern of LLMContext, a secondary design pattern called Handling has been implemented. This pattern is responsible for managing the different ways of interacting with the language models. The Handling pattern allows LLMContext to be compatible with various models and processing approaches, providing flexible and scalable handling of different text generation strategies.

The Handling pattern ensures that regardless of the underlying model used, the LLMContext interface can manage the model call uniformly and efficiently. This facilitates the integration and expansion of new models without affecting existing functionality.

## Metrics
A class named Metrics has been created to serve as an interface for calculating various performance metrics related to language model outputs.

### Initialization
To initialize the Metrics class, you need to provide the name of the model:
> metrics = Metrics(name_model)

### Public functions
The Metrics class includes several public functions for calculating performance metrics:
> metrics.calculate(ground_truth, text_generated)

This method is responsible for calculating the following metrics:
- Precision: The ratio of correctly predicted positive observations to the total predicted positives.
- Recall: The ratio of correctly predicted positive observations to all actual positives.
- F1 Score: The harmonic mean of precision and recall.
Cosine Similarity: A measure of similarity between two non-zero vectors of an inner product space.
- Levenshtein Distance: A metric for measuring the difference between two sequences.
- Inverse Levenshtein Distance: The inverse of the Levenshtein distance, which can be useful for certain calculations.
- Overall: A general metric that can summarize the performance based on the calculated values.

Once the metrics are calculated, if you wish to store them for future reference, you need to use the following method:
> metrics.store_metrics()

This allows you to continue calculating additional metrics with different data within the same model.

### Saving Metrics
Finally, when you have iterated enough times and want to retrieve a CSV file containing all the information, you can use the function:
> metrics.save_metrics()

This function is responsible for saving the list of all experiments conducted up to that point in a CSV format.

# MEASURES
- Nombres
- Calle
- Número de telefonos
- Edad
- Sexo
- Profesión
- Territorio
    - CIP
    - Provincia
- Hospital
- FAMILIARES
- Fechas
- País
