"""
Script rápido para probar Vertex AI con input/output directo.

Requisitos:
- VERTEX_PROJECT_ID configurado
- VERTEX_LOCATION configurado (opcional, default: us-central1)
- Credenciales GCP activas (gcloud auth application-default login)

Uso:
    python test_vertex_quick.py
"""

import os
from dotenv import load_dotenv
from src.carmina.llm.cloud_providers.vertex_ai_provider import VertexAIProvider

# Cargar variables del .env
load_dotenv()


def main():
    """Ejecuta una consulta simple a Vertex AI."""
    
    # Verificar credenciales
    if not os.getenv("VERTEX_PROJECT_ID"):
        print("❌ ERROR: VERTEX_PROJECT_ID no está configurado")
        print("Configura: $env:VERTEX_PROJECT_ID='tu-project-id'")
        return
    
    print("=" * 60)
    print("🚀 TEST RÁPIDO DE VERTEX AI")
    print("=" * 60)
    print(f"📍 Project: {os.getenv('VERTEX_PROJECT_ID')}")
    print(f"📍 Location: {os.getenv('VERTEX_LOCATION', 'us-central1')}")
    print()
    
    # Inicializar provider
    try:
        provider = VertexAIProvider()
        print("✅ Provider inicializado correctamente")
        print()
    except Exception as e:
        print(f"❌ Error al inicializar provider: {e}")
        return
    
    # Definir modelo y parámetros
    model_id = "gemini-2.0-flash"  # Cambiar si quieres otro modelo
    temperature = 0.7
    max_tokens = 500
    
    print(f"🤖 Modelo: {model_id}")
    print(f"🌡️  Temperature: {temperature}")
    print(f"📝 Max tokens: {max_tokens}")
    print()
    
    # Pregunta de ejemplo (puedes cambiarla)
    user_question = "¿Qué es la inteligencia artificial? Responde en español en máximo 3 líneas."
    
    print("=" * 60)
    print("📥 INPUT (Usuario):")
    print("=" * 60)
    print(user_question)
    print()
    
    # Preparar mensajes
    messages = [
        {"role": "system", "content": "Eres un asistente útil que responde en español de forma clara y concisa."},
        {"role": "user", "content": user_question}
    ]
    
    # Ejecutar inferencia
    try:
        print("⏳ Enviando petición a Vertex AI...")
        result = provider.run_inference(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        print()
        print("=" * 60)
        print("📤 OUTPUT (Modelo):")
        print("=" * 60)
        print(result)
        print()
        print("=" * 60)
        print("✅ ¡Éxito!")
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ ERROR:")
        print("=" * 60)
        print(f"{type(e).__name__}: {e}")
        print()


def interactive_mode():
    """Modo interactivo para hacer múltiples preguntas."""
    
    if not os.getenv("VERTEX_PROJECT_ID"):
        print("❌ ERROR: VERTEX_PROJECT_ID no está configurado")
        return
    
    provider = VertexAIProvider()
    model_id = "gemini-2.0-flash"
    
    print("=" * 60)
    print("💬 MODO INTERACTIVO - VERTEX AI")
    print("=" * 60)
    print(f"Modelo: {model_id}")
    print("Escribe 'salir' o 'exit' para terminar")
    print("=" * 60)
    print()
    
    while True:
        try:
            # Leer pregunta del usuario
            question = input("Tu pregunta: ").strip()
            
            if question.lower() in ['salir', 'exit', 'quit', '']:
                print("\n👋 ¡Hasta luego!")
                break
            
            # Enviar a Vertex AI
            messages = [
                {"role": "system", "content": "Eres un asistente útil."},
                {"role": "user", "content": question}
            ]
            
            print("\n⏳ Pensando...")
            result = provider.run_inference(
                model_id=model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            print(f"\n🤖 Respuesta:\n{result}\n")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    import sys
    
    # Si se pasa --interactive, usar modo interactivo
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
