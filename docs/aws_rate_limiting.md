# AWS Rate Limiting and Throttling Management

## Problema
Al usar AWS Bedrock con alta frecuencia, puedes encontrar errores de throttling:
```
ThrottlingException: Too many tokens, please wait before trying again.
```

## Solución Implementada

### 1. Retry con Backoff Exponencial
El `AWSProvider` ahora incluye un mecanismo de retry automático que:
- Detecta automáticamente `ThrottlingException`
- Espera con backoff exponencial antes de reintentar
- Incluye jitter aleatorio para evitar el "thundering herd"
- Configurable a través de variables de entorno

### 2. Rate Limiting
- Delays automáticos entre requests consecutivos
- Delays configurables entre elementos de batch processing
- Logging detallado del progreso

## Configuración

Añade estas variables a tu archivo `.env`:

```bash
# AWS Rate Limiting Configuration
AWS_MAX_RETRIES=5        # Número máximo de reintentos
AWS_BASE_DELAY=1.0       # Delay base en segundos
AWS_MAX_DELAY=60.0       # Delay máximo en segundos
AWS_REQUEST_DELAY=0.1    # Delay entre requests en segundos
```

### Valores Recomendados por Caso de Uso

#### Desarrollo/Testing (requests ocasionales)
```bash
AWS_MAX_RETRIES=3
AWS_BASE_DELAY=0.5
AWS_MAX_DELAY=30.0
AWS_REQUEST_DELAY=0.05
```

#### Producción con volumen moderado
```bash
AWS_MAX_RETRIES=5
AWS_BASE_DELAY=1.0
AWS_MAX_DELAY=60.0
AWS_REQUEST_DELAY=0.1
```

#### Batch processing intensivo
```bash
AWS_MAX_RETRIES=8
AWS_BASE_DELAY=2.0
AWS_MAX_DELAY=120.0
AWS_REQUEST_DELAY=0.5
```

## Características del Sistema de Retry

### Backoff Exponencial
El delay entre reintentos se calcula como:
```
delay = min(base_delay * (2 ** attempt) + random(0, 1), max_delay)
```

### Logging
El sistema registra:
- Advertencias cuando ocurre throttling
- Número de intento actual
- Tiempo de espera antes del siguiente intento
- Progreso en batch processing

### Manejo de Errores
- Solo reintenta en `ThrottlingException`
- Otros errores se propagan inmediatamente
- Después de agotar todos los reintentos, se propaga el error original

## Ejemplos de Uso

### Procesamiento Individual
```python
aws_provider = AWSProvider()
try:
    result = aws_provider.run_inference(
        model_id="claude-3.5-haiku",
        messages={"messages": [{"role": "user", "content": "Hello"}]},
        inference_params={"max_tokens": 100, "temperature": 0.7}
    )
except ClientError as e:
    if e.response['Error']['Code'] == 'ThrottlingException':
        print("Throttling error after all retries exhausted")
    else:
        print(f"Other AWS error: {e}")
```

### Batch Processing
```python
aws_provider = AWSProvider()
batch_inputs = [
    {"messages": [{"role": "user", "content": f"Process item {i}"}]}
    for i in range(100)
]

results = aws_provider.batch_process(
    model_name="claude-3.5-haiku",
    batch_inputs=batch_inputs,
    inference_params={"max_tokens": 100}
)

# Los resultados incluyen manejo automático de errores
for i, result in enumerate(results):
    if "error" in result:
        print(f"Error in item {i}: {result['error']}")
    else:
        print(f"Success for item {i}")
```

## Monitoreo

### Logs Importantes
Busca estos mensajes en los logs:
```
WARNING:root:ThrottlingException on attempt 2/6. Retrying in 3.45 seconds...
INFO:root:Processing batch item 15/100
```

### Métricas a Monitorear
- Frecuencia de throttling exceptions
- Tiempo promedio de procesamiento
- Tasa de éxito en batch processing

## Troubleshooting

### Si sigues recibiendo throttling errors:
1. Aumenta `AWS_BASE_DELAY` y `AWS_REQUEST_DELAY`
2. Considera reducir el tamaño de los batches
3. Verifica los límites de tu cuenta AWS
4. Considera usar múltiples regiones si es apropiado

### Si el procesamiento es muy lento:
1. Reduce `AWS_REQUEST_DELAY` gradualmente
2. Usa `AWS_MAX_RETRIES` más bajo para fallar más rápido
3. Implementa procesamiento paralelo con múltiples workers
