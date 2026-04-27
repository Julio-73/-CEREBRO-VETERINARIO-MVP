# 🧠 Sistema RAG - MVP para Entrevista Técnica

> **Stack**: Python · FAISS · sentence-transformers · Ollama (phi) · Streamlit

---

## ⚡ Inicio Rápido (5 pasos)

```bash
# 1. Instalar Ollama y descargar modelo
curl -fsSL https://ollama.com/install.sh | sh    # Linux/Mac
ollama pull phi                                    # Modelo ligero ~1.6GB

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows PowerShell

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Agregar documentos (carpeta /docs)
mkdir -p docs
cp tu_documento.pdf docs/     # o archivos .txt

# 5. Ejecutar (en terminales separadas)
ollama serve                   # Terminal 1: servidor LLM
streamlit run app.py           # Terminal 2: aplicación web
```

Abrir en el navegador: **http://localhost:8501**

---

## 🏗️ Arquitectura del Sistema

```
PIPELINE DE INGESTA (una vez):
┌──────────┐    ┌──────────┐    ┌────────────┐    ┌──────────────┐
│  /docs   │───▶│ Chunking │───▶│ Embeddings │───▶│ FAISS Index  │
│ PDF, TXT │    │ 500 chars│    │ MiniLM-L6  │    │  /data/*.bin │
└──────────┘    └──────────┘    └────────────┘    └──────────────┘

PIPELINE DE CONSULTA (cada pregunta):
┌────────────┐    ┌────────────┐    ┌──────────┐    ┌──────────────┐
│  Pregunta  │───▶│ Embedding  │───▶│ FAISS    │───▶│    Ollama    │
│  usuario   │    │  (query)   │    │ top-k=3  │    │  (phi model) │
└────────────┘    └────────────┘    └──────────┘    └──────────────┘
                                                            │
                                                     ┌──────▼──────┐
                                                     │  Respuesta  │
                                                     │  al usuario │
                                                     └─────────────┘
```

---

## 📁 Estructura del Proyecto

```
rag-mvp/
├── rag.py              # Motor RAG (lógica completa)
├── app.py              # Interfaz Streamlit
├── requirements.txt    # Dependencias Python
├── README.md
├── docs/               # ← Coloca tus documentos aquí
│   ├── mi_doc.pdf
│   └── mi_texto.txt
└── data/               # Generado automáticamente
    ├── faiss.index     # Índice vectorial
    └── chunks.pkl      # Fragmentos de texto
```

---

## 🔧 Comandos Útiles

```bash
# Probar solo el motor RAG (sin UI)
python rag.py

# Reconstruir el índice (si cambias documentos)
rm -rf data/ && streamlit run app.py

# Ver modelos Ollama disponibles
ollama list

# Cambiar modelo (editar en rag.py)
OLLAMA_MODEL = "phi"       # Ligero, rápido
# OLLAMA_MODEL = "llama3"  # Más capaz, más lento
# OLLAMA_MODEL = "mistral" # Buen balance
```

---

## 🎯 Cómo Defender Este Proyecto en Entrevista

### Pregunta: "¿Por qué RAG en lugar de fine-tuning?"
**Respuesta**: RAG es más apropiado cuando el conocimiento cambia frecuentemente (documentos nuevos, actualizaciones). El fine-tuning es costoso y requiere reentrenar el modelo completo. RAG permite actualizar la base de conocimiento sin tocar el LLM.

### Pregunta: "¿Por qué FAISS?"
**Respuesta**: FAISS es ideal para MVPs: sin servidor externo, funciona en memoria, ultra-rápido para búsquedas aproximadas (ANN). Para producción escalaría a Pinecone o pgvector según las necesidades de persistencia y distribución.

### Pregunta: "¿Por qué overlap en el chunking?"
**Respuesta**: El overlap de 50 chars evita que el contexto se pierda en los bordes de los chunks. Si una entidad importante (nombre, fecha, definición) cae justo en el corte entre dos chunks, el overlap asegura que al menos uno la capture completa.

### Pregunta: "¿Cómo evaluarías la calidad del RAG?"
**Respuesta**: Con el framework RAGAS que mide: Faithfulness (la respuesta solo usa el contexto), Answer Relevancy (la respuesta responde la pregunta), Context Precision y Context Recall. También haría A/B testing manual con un conjunto de preguntas con respuestas conocidas.

### Pregunta: "¿Cómo lo escalarías a producción?"
**Respuesta**: Reemplazaría componentes manteniendo la interfaz: FAISS→pgvector/Pinecone, Ollama→OpenAI API, Streamlit→FastAPI+React, agregando Redis para caché, Docker para containerización, y un pipeline de ingesta asíncrono con Celery.

---

## ⚠️ Solución de Problemas Comunes

| Error | Causa | Solución |
|---|---|---|
| `Connection refused` | Ollama no corre | Ejecutar `ollama serve` |
| `No module named faiss` | Dependencia faltante | `pip install faiss-cpu` |
| `No documents found` | /docs vacía | Agregar archivos .txt/.pdf |
| Respuestas en inglés | Modelo phi | Agregar "Responde en español" al prompt |
| Índice desactualizado | Docs nuevos | `rm -rf data/` y reiniciar |
