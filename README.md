# 🧠 CEREBRO VETERINARIO MVP

MVP de Rag Inteligente que contesta preguntas basado en sus contextos.  

> **Stack**: Python · FAISS · sentence-transformers · Groq · Streamlit

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


----------------------------------------------------------------------

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

---------------------------------------------------------------------

CONTACTO:

CCorreo: julioquispe.dev@gmail.com
limkedim: https://www.linkedin.com/in/julio-cesar-quispe-garrido/
live:





