# TODO - Carmina Healthcare Text Anonymization System

## ✅ **COMPLETADO - Funcionalidades Implementadas**

### **Sistema de Conteo de Tokens y Costos** 🎉
- [x] **Token Counter Utilities** (`src/carmina/llm/utils/token_counter.py`)
  - Implementación completa con soporte para OpenAI (tiktoken), Anthropic, Generic y Mock
  - Factory pattern para selección automática de contador por proveedor
  - Conteo de tokens para prompts sistema y usuario por separado
- [x] **Implementación en Estrategias LLM**
  - `MockLLMStrategy.count_tokens()` - Completamente implementado y testeado
  - `OpenAIStrategy.count_tokens()` - Implementado con soporte tiktoken
  - `AnthropicStrategy.count_tokens()` - Implementado con aproximación por caracteres
  - Método `count_prompt_tokens()` en base strategy para todas las estrategias
- [x] **Métricas de Tokens** (`src/carmina/metrics/recorder.py`)
  - `record_token_metrics()` - Tracking de tokens por modo de anonimización
  - `record_cost_metrics()` - Tracking de costos entrada/salida/total
  - Integración completa con sistema de métricas existente
- [x] **Tests Completos**
  - 9 tests nuevos para `test_token_counter.py` - Todos pasando ✅
  - 8 tests nuevos para `test_metrics_recorder.py` - Todos pasando ✅
  - Tests actualizados en `test_llm_integration.py` - 18/18 pasando ✅
- [x] **Ejemplo de Uso**
  - `examples/token_counting_example.py` - Demostración completa de funcionalidad
  - Casos de uso: conteo básico, prompts completos, tracking costos, procesamiento batch

**Resultado:** ✅ **95 tests pasando** (+39 desde inicio), **68% éxito** (+22% mejora)

---

## 🔴 Crítico - Implementaciones Faltantes

### 1. **Estrategias LLM Incompletas**
#### **OpenAIStrategy** (`src/carmina/llm/strategies/openai_strategy.py`)
- [ ] Implementar `batch_identify()` - actualmente tiene `pass`
- [ ] Implementar `get_context_window()` - actualmente tiene `pass`  
- [x] ~~Implementar `count_tokens()` - actualmente tiene `pass`~~ ✅ **COMPLETADO**

#### **MockLLMStrategy** (`src/carmina/llm/strategies/mock_strategy.py`)
- [ ] Implementar `batch_identify()` - actualmente tiene `pass`
- [x] ~~Implementar `count_tokens()` - actualmente tiene `pass`~~ ✅ **COMPLETADO**
- [ ] Corregir `run_inference()` - signatura inconsistente con base class
- [ ] Añadir más ejemplos mock para `__mock_identify__` y `__mock_label__`

#### **Otras Estrategias**
- [x] ~~Implementar `count_tokens()` en `AnthropicStrategy`~~ ✅ **COMPLETADO**
- [ ] Verificar e implementar métodos faltantes en todas las estrategias:
  - `DeepSeekStrategy`, `GeminiStrategy`, `LlamaStrategy`, `QwenStrategy`, `MistralStrategy`
- [ ] Implementar `batch_identify()` en todas las estrategias

### 2. **Factory Pattern Incompleto**
#### **LLMFactory** (`src/carmina/llm/factory.py`)
- [ ] Implementar estrategia para modelos DeepSeek
- [ ] Mejorar manejo de modelos desconocidos
- [ ] Añadir validación de compatibilidad modelo-proveedor

### 3. **Pipeline de Anonimización**
#### **AnonymizationPipeline** (`src/carmina/pipeline/anon_pipeline.py`)
- [ ] Mejorar manejo de errores en procesamiento
- [ ] Implementar validación de entrada de datos
- [ ] Añadir soporte para procesamiento por lotes
- [ ] Optimizar configuración de límites de documentos

#### **Procesadores** (`src/carmina/pipeline/processors/`)
- [ ] Verificar implementación completa de todos los procesadores
- [ ] Añadir validación de datos entre etapas
- [ ] Implementar recuperación de errores

### 4. **Model Executor**
#### **ModelExecutor** (`src/carmina/model_executor.py`)
- [ ] Mejorar manejo de directorios de salida inexistentes
- [ ] Añadir validación de extensiones de archivo
- [ ] Implementar logging más detallado de errores
- [ ] Añadir soporte para diferentes formatos de entrada

## 🟡 Importante - Mejoras de Testing

### 5. **Mocks y Fixtures**
- [ ] Crear fixtures completos para datos de entrada CARMEN
- [ ] Implementar mocks para todos los proveedores cloud
- [ ] Añadir datos de test realistas para pipeline
- [ ] Crear mocks para archivos y directorios

### 6. **Tests E2E**
#### **CompleteWorkflow** (`tests/e2e/test_complete_workflow.py`)
- [ ] Crear datos mock para workflow completo
- [ ] Implementar setup/teardown de archivos temporales
- [ ] Añadir mocks para dependencias externas
- [ ] Simplificar tests para ser menos dependientes de archivos reales

### 7. **Tests de Integración**
#### **Pipeline Tests** (`tests/integration/test_anonymization_pipeline.py`)
- [ ] Crear datos mock estructurados
- [ ] Implementar fixtures para configuración de entorno
- [ ] Añadir tests para diferentes formatos de datos
- [ ] Mejorar aislamiento de tests

#### **Benchmark Runner Tests** (`tests/integration/test_benchmark_runner.py`)
- [ ] Mockear sistema de archivos
- [ ] Crear fixtures para configuración de benchmark
- [ ] Implementar mocks para ModelExecutor
- [ ] Añadir tests para diferentes escenarios de error

### 8. **Tests Unitarios**
#### **Base Strategy** (`tests/unit/test_base_strategy.py`)
- [ ] Completar TestStrategy con todos los métodos requeridos
- [ ] Añadir tests para manejo de errores
- [ ] Implementar tests para validación de parámetros
- [ ] Mejorar cobertura de métodos abstractos

## 🟢 Mejoras - Funcionalidades Adicionales

### 9. **Logging y Monitoreo**
- [ ] Implementar logging estructurado en todas las clases
- [x] ~~Añadir métricas de rendimiento~~ ✅ **COMPLETADO** (token counting y cost tracking)
- [ ] Implementar sistema de alertas para errores
- [ ] Añadir logging de progreso para operaciones largas

### 10. **Configuración**
- [ ] Validar todas las variables de entorno requeridas
- [ ] Implementar configuración por defecto más robusta
- [ ] Añadir validación de configuración al inicio
- [ ] Documentar todas las variables de configuración

### 11. **Manejo de Errores**
- [ ] Implementar excepciones customizadas más específicas
- [ ] Añadir recuperación automática para errores transitorios
- [ ] Implementar reintentos con backoff exponencial
- [ ] Mejorar mensajes de error para usuarios

### 12. **Optimización**
- [ ] Implementar cache para respuestas de LLM
- [ ] Optimizar procesamiento por lotes
- [ ] Añadir paralelización donde sea apropiado
- [ ] Implementar streaming para archivos grandes

## 🔵 Documentación

### 13. **Documentación Técnica**
- [ ] Documentar todas las interfaces y contratos
- [ ] Crear diagramas de arquitectura actualizados
- [ ] Documentar proceso de adición de nuevos proveedores
- [ ] Crear guía de troubleshooting

### 14. **Documentación de Usuario**
- [ ] Actualizar README con ejemplos de uso
- [ ] Crear guía de configuración paso a paso
- [ ] Documentar formatos de datos soportados
- [ ] Añadir ejemplos de configuración para diferentes casos de uso

## 📊 Prioridades de Implementación

### **Fase 1 - Crítico (1-2 semanas)**
1. ~~Completar métodos faltantes en estrategias principales~~ ✅ **COMPLETADO** (count_tokens)
2. ~~Arreglar MockLLMStrategy para tests~~ ✅ **PARCIALMENTE COMPLETADO** (count_tokens implementado)
3. Implementar manejo básico de errores en pipeline

### **Fase 2 - Importante (2-4 semanas)**
1. Crear mocks completos para tests
2. Implementar validación de configuración
3. Mejorar manejo de archivos en ModelExecutor

### **Fase 3 - Mejoras (1-2 meses)**
1. Optimizaciones de rendimiento
2. Logging y monitoreo avanzado
3. Documentación completa

## 🎯 Objetivos de Calidad

### **Cobertura de Tests**
- [ ] Alcanzar 90%+ de cobertura de código
- [ ] Todos los tests unitarios deben pasar
- [ ] Al menos 80% de tests de integración funcionando

### **Robustez**
- [ ] Manejo graceful de todos los errores conocidos
- [ ] Validación completa de entrada de datos
- [ ] Recuperación automática donde sea posible

### **Rendimiento**
- [ ] Procesamiento de al menos 1000 documentos/hora
- [ ] Uso de memoria optimizado para archivos grandes
- [ ] Tiempo de respuesta < 10s para documentos individuales

---

## 📈 **Progreso Actual del Proyecto**

### **Estado de Tests (Última actualización)**
- ✅ **95 tests pasando** (vs 56 inicial)
- ❌ **45 tests fallando** (vs 66 inicial)  
- 📊 **68% tasa de éxito** (+22% mejora)
- 🆕 **+18 tests nuevos** para funcionalidad de tokens

### **Funcionalidades Principales Completadas**
- ✅ **Sistema completo de conteo de tokens y costos**
- ✅ **Métricas de rendimiento y cost tracking**
- ✅ **Tests unitarios e integración para tokens**
- ✅ **Soporte multi-proveedor (OpenAI, Anthropic, Mock)**

### **Próximos Hitos Prioritarios**
1. **Completar batch_identify()** en estrategias principales
2. **Implementar context_window()** en estrategias
3. **Mejorar pipeline de anonimización** (manejo de errores)
4. **Crear mocks completos** para tests E2E

---

**Nota**: Este TODO está basado en el análisis de tests fallidos y gaps identificados en la implementación actual. Las prioridades pueden ajustarse según las necesidades del proyecto.