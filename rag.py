"""
rag.py - Motor RAG (Retrieval-Augmented Generation) - VERSIÓN CORREGIDA
====================================================
CORRECCIONES APLICADAS:
1. Mejor logging para diagnosticar problemas
2. Umbral de similitud mínimo
3. Mostrar contenido de chunks para debugging
4. Manejo de chunks vacíos
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ─── CONFIGURACIÓN DE LOGGING ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─── CONFIGURACIÓN GLOBAL ────────────────────────────────────────────────────
DOCS_PATH     = Path("docs")           
INDEX_PATH    = Path("data/faiss.index")  
CHUNKS_PATH   = Path("data/chunks.pkl")   

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  

# ⭐ GROQ (gratis) - Alternativa a Ollama para Streamlit Cloud
# Obtén tu API key gratis en: https://console.groq.com/
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"  # Modelo gratis y rápido

# ⭐ CAMBIOS CLAVE AQUÍ
CHUNK_SIZE    = 500    
CHUNK_OVERLAP = 100    
TOP_K         = 5      
MIN_SIMILARITY = 0.2


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 1: CARGA DE DOCUMENTOS - MEJORADA CON MÁS LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def load_documents(path: Path = DOCS_PATH) -> List[str]:
    """
    Carga y lee todos los archivos .txt y .pdf de la carpeta especificada.
    ⭐ MEJORADA: Más logging para diagnosticar problemas
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Carpeta '{path}' no encontrada.\n"
            f"Crea la carpeta y agrega tus archivos .txt o .pdf"
        )

    documents = []
    
    # ⭐ NUEVO: Listar todos los archivos encontrados
    all_files = list(path.iterdir())
    logger.info(f"📂 Archivos encontrados en {path}: {len(all_files)}")
    for f in all_files:
        logger.info(f"   - {f.name} ({f.suffix})")

    for file_path in sorted(path.iterdir()):
        try:
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8")
                
                # ⭐ NUEVO: Verificar que el texto no esté vacío
                if len(text.strip()) < 10:
                    logger.warning(f"⚠ TXT casi vacío: {file_path.name} | Solo {len(text)} chars")
                    continue
                    
                documents.append(text)
                logger.info(f"✓ TXT cargado: {file_path.name} | {len(text):,} chars")
                
                # ⭐ NUEVO: Mostrar preview del contenido
                preview = text[:200].replace('\n', ' ')
                logger.info(f"   Preview: {preview}...")

            elif file_path.suffix.lower() == ".pdf":
                text = _extract_pdf(file_path)
                if text and len(text.strip()) > 50:
                    documents.append(text)
                    logger.info(f"✓ PDF cargado: {file_path.name} | {len(text):,} chars")
                else:
                    logger.warning(f"⚠ PDF sin texto extraíble: {file_path.name}")

        except Exception as e:
            logger.warning(f"✗ Error al cargar {file_path.name}: {e}")

    if not documents:
        raise ValueError(
            "No se encontraron documentos válidos en /docs.\n"
            "Agrega archivos .txt o .pdf y vuelve a intentar."
        )

    # ⭐ NUEVO: Resumen de lo cargado
    total_chars = sum(len(d) for d in documents)
    logger.info(f"📁 Total documentos cargados: {len(documents)}")
    logger.info(f"📝 Total caracteres: {total_chars:,}")
    return documents


def _extract_pdf(path: Path) -> Optional[str]:
    """Extrae texto de un archivo PDF."""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(pages_text)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"pdfplumber falló en {path.name}: {e}")

    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages_text = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages_text)
    except ImportError:
        logger.error("No se pudo leer el PDF. Instala: pip install pdfplumber")
        return None
    except Exception as e:
        logger.error(f"Error leyendo PDF {path.name}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 2: CHUNKING - MEJORADO
# ══════════════════════════════════════════════════════════════════════════════

def split_into_chunks(
    documents: List[str],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Divide los documentos en fragmentos de tamaño fijo con overlap.
    ⭐ MEJORADA: Más logging y manejo de documentos cortos
    """
    all_chunks = []

    for doc_idx, doc in enumerate(documents):
        doc = doc.strip()
        if not doc:
            continue
            
        # ⭐ NUEVO: Logging del documento
        logger.info(f"📄 Procesando documento {doc_idx + 1}: {len(doc)} caracteres")

        chunks_for_doc = []
        start = 0

        while start < len(doc):
            end = min(start + chunk_size, len(doc))
            chunk = doc[start:end].strip()

            # Reducido el mínimo de 30 a 20
            if len(chunk) > 20:
                chunks_for_doc.append(chunk)

            start += chunk_size - overlap

        # ⭐ NUEVO: Logging de chunks generados
        logger.info(f"   → {len(chunks_for_doc)} chunks generados")
        
        # ⭐ NUEVO: Mostrar primer chunk de cada documento
        if chunks_for_doc:
            preview = chunks_for_doc[0][:150].replace('\n', ' ')
            logger.info(f"   Primer chunk: {preview}...")
        
        all_chunks.extend(chunks_for_doc)

    logger.info(f"✂️ Total chunks generados: {len(all_chunks)}")
    
    # ⭐ NUEVO: Advertencia si hay muy pocos chunks
    if len(all_chunks) < 5:
        logger.warning(f"⚠️ SOLO {len(all_chunks)} chunks! Esto puede causar problemas de búsqueda.")
        logger.warning("   Considera agregar más documentos o reducir CHUNK_SIZE")
    
    return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 3: EMBEDDINGS Y CONSTRUCCIÓN DEL ÍNDICE FAISS
# ══════════════════════════════════════════════════════════════════════════════

def build_index(
    chunks: List[str],
    model_name: str = EMBEDDING_MODEL
) -> Tuple[faiss.Index, SentenceTransformer]:
    """
    Genera embeddings para todos los chunks y construye el índice FAISS.
    """
    logger.info(f"🤗 Cargando modelo: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"⚡ Generando embeddings para {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32
    )

    # Normalización L2
    faiss.normalize_L2(embeddings)

    # Crear índice
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    logger.info(f"✅ Índice FAISS creado: {index.ntotal} vectores | {dimension} dimensiones")
    return index, model


def save_index(index: faiss.Index, chunks: List[str]) -> None:
    """Persiste el índice FAISS y los chunks en disco."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    logger.info(f"💾 Índice guardado: {INDEX_PATH}")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"💾 Chunks guardados: {CHUNKS_PATH}")


def load_index() -> Tuple[faiss.Index, List[str]]:
    """Carga el índice FAISS y los chunks desde disco."""
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "No se encontró índice guardado. "
            "Elimina /data si existe y reinicia la aplicación."
        )

    index = faiss.read_index(str(INDEX_PATH))
    logger.info(f"📂 Índice cargado desde disco: {index.ntotal} vectores")

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 4: BÚSQUEDA SEMÁNTICA - ⭐ CORREGIDA
# ══════════════════════════════════════════════════════════════════════════════

def search_context(
    query: str,
    index: faiss.Index,
    chunks: List[str],
    model: SentenceTransformer,
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY  # ⭐ NUEVO PARÁMETRO
) -> List[Tuple[str, float]]:
    """
    Busca los chunks más semánticamente similares a la consulta.
    
    ⭐ CORREGIDA: 
    - Agrega umbral mínimo de similitud
    - Mejor logging
    - Siempre devuelve al menos 1 resultado (el mejor)
    """
    logger.info(f"🔍 Buscando: '{query}'")
    logger.info(f"   Parámetros: top_k={top_k}, min_similarity={min_similarity}")

    # Generar y normalizar el embedding de la query
    query_vector = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)

    # ⭐ NUEVO: Buscar más resultados para filtrar después
    search_k = min(top_k * 3, index.ntotal)  # Buscar 3x más para tener margen
    scores, indices = index.search(query_vector.astype(np.float32), search_k)

    # ⭐ NUEVO: Logging detallado de resultados
    logger.info(f"   Resultados crudos (top {search_k}):")
    
    all_results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(chunks):
            all_results.append((chunks[idx], float(score), idx))
            logger.info(f"      Chunk {idx}: score={score:.4f} | preview: {chunks[idx][:80].replace(chr(10), ' ')}...")

    # ⭐ NUEVO: Filtrar por umbral de similitud
    filtered_results = [(chunk, score) for chunk, score, _ in all_results if score >= min_similarity]
    
    logger.info(f"   Resultados filtrados (>= {min_similarity}): {len(filtered_results)}")
    
    # ⭐ NUEVO: Si no hay resultados que pasen el umbral, devolver el mejor de todos modos
    if not filtered_results and all_results:
        best_chunk, best_score, best_idx = all_results[0]
        logger.warning(f"   ⚠️ Ningún resultado superó el umbral {min_similarity}")
        logger.warning(f"   → Devolviendo el mejor resultado: score={best_score:.4f}")
        filtered_results = [(best_chunk, best_score)]
    
    # ⭐ NUEVO: Limitar a top_k
    final_results = filtered_results[:top_k]
    
    logger.info(f"   Resultados finales: {len(final_results)}")
    
    return final_results


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 5: GENERACIÓN DE RESPUESTA - MEJORADA
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(
    query: str,
    context_results: List[Tuple[str, float]]
) -> str:
    """
    Construye el prompt con contexto + pregunta y obtiene respuesta de Ollama.
    ⭐ MEJORADA: Better prompt y manejo de errores
    """
    if not context_results:
        return "No encontré información relevante en los documentos para responder tu pregunta."

    # ⭐ NUEVO: Mostrar scores en el contexto para debugging
    context_parts = []
    for i, (chunk, score) in enumerate(context_results, 1):
        context_parts.append(f"[Fragmento {i} - Relevancia: {score:.2f}]\n{chunk}")
    
    context = "\n\n---\n\n".join(context_parts)

    # ⭐ MEJORADO: Prompt más claro
    prompt = f"""Eres un asistente médico veterinario experto. 

INSTRUCCIONES IMPORTANTES:
1. Responde la pregunta usando SOLO la información de los documentos proporcionados abajo
2. Si la información NO está en los documentos, di: "No tengo información sobre ese tema en los documentos disponibles"
3. Responde en español
4. Sé conciso pero completo
5. Si mencionas algo, cita de qué fragmento lo sacaste

DOCUMENTOS DISPONIBLES:
{context}

PREGUNTA DEL USUARIO: {query}

RESPUESTA:"""

    # ⭐ LÓGICA DUAL: Groq (nube) o Ollama (local)
    if GROQ_API_KEY:
        # === MODO GROQ (Streamlit Cloud) ===
        return _generate_groq(prompt, context)
    else:
        # === MODO OLLAMA (local) ===
        return _generate_ollama(prompt, context)


def _generate_groq(prompt: str, context: str) -> str:
    """Genera respuesta usando Groq (para Streamlit Cloud)"""
    try:
        logger.info(f"🤖 Enviando a Groq ({GROQ_MODEL})...")
        logger.info(f"   Contexto enviado: {len(context)} caracteres")
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Eres un asistente útil que responde preguntas sobre documentos veterinarios. Responde en español."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 512
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=90
        )
        response.raise_for_status()
        
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        
        logger.info(f"   Respuesta recibida: {len(answer)} caracteres")
        
        if not answer:
            return "El modelo retornó una respuesta vacía. Intenta reformular la pregunta."

        return answer

    except requests.exceptions.ConnectionError:
        return "❌ **Error de conexión**. Verifica tu conexión a internet."
    except requests.exceptions.HTTPError as e:
        error_msg = e.response.text if e.response else str(e)
        return f"❌ **Error de Groq**: {error_msg}"
    except Exception as e:
        logger.error(f"Error inesperado en generate_response: {e}", exc_info=True)
        return f"❌ **Error inesperado**: {str(e)}"


def _generate_ollama(prompt: str, context: str) -> str:
    """Genera respuesta usando Ollama (para desarrollo local)"""
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "llama3.2:1b"
    
    try:
        logger.info(f"🦙 Enviando a Ollama ({OLLAMA_MODEL})...")
        logger.info(f"   Contexto enviado: {len(context)} caracteres")
        
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512,
                    "top_p": 0.9,
                }
            },
            timeout=90
        )
        response.raise_for_status()

        answer = response.json().get("response", "").strip()
        
        logger.info(f"   Respuesta recibida: {len(answer)} caracteres")
        
        if not answer:
            return "El modelo retornó una respuesta vacía. Intenta reformular la pregunta."

        return answer

    except requests.exceptions.ConnectionError:
        return (
            "❌ **Error de conexión con Ollama**\n\n"
            "Asegúrate de que Ollama está ejecutándose:\n"
            "```\nollama serve\n```\n"
            "Y que el modelo está descargado:\n"
            "```\nollama pull llama3.2:1b\n```"
        )
    except requests.exceptions.Timeout:
        return "⏱️ **Timeout**: El modelo tardó más de 90 segundos."
    except requests.exceptions.HTTPError as e:
        return f"❌ **Error HTTP de Ollama**: {e.response.status_code}"
    except Exception as e:
        logger.error(f"Error inesperado en generate_response: {e}", exc_info=True)
        return f"❌ **Error inesperado**: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def initialize_rag() -> Tuple[faiss.Index, List[str], SentenceTransformer]:
    """
    Inicializa el sistema RAG completo.
    ⭐ MEJORADA: Más logging
    """
    logger.info("🚀 Inicializando sistema RAG...")

    logger.info(f"🤗 Cargando modelo de embeddings: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    if INDEX_PATH.exists() and CHUNKS_PATH.exists():
        logger.info("📂 Índice existente encontrado, cargando...")
        index, chunks = load_index()
        
        # ⭐ NUEVO: Advertencia si hay pocos chunks
        if len(chunks) < 5:
            logger.warning(f"⚠️ ADVERTENCIA: Solo {len(chunks)} chunks en el índice")
            logger.warning("   Esto puede causar mala calidad en las respuestas")
            logger.warning("   Considera: 1) Agregar más documentos, 2) Reducir CHUNK_SIZE")
            logger.warning("   Para reconstruir: elimina la carpeta /data y reinicia")
    else:
        logger.info("🔨 Construyendo nuevo índice desde documentos...")
        documents = load_documents()
        chunks = split_into_chunks(documents)
        index, model = build_index(chunks)
        save_index(index, chunks)

    logger.info(
        f"✅ Sistema RAG listo: {index.ntotal} vectores indexados | "
        f"Modelo: {EMBEDDING_MODEL} | LLM: {'Groq' if GROQ_API_KEY else 'Ollama'}"
    )
    return index, chunks, model


def query_rag(
    question: str,
    index: faiss.Index,
    chunks: List[str],
    model: SentenceTransformer,
    top_k: int = TOP_K
) -> dict:
    """
    Función principal de consulta.
    ⭐ MEJORADA: Mejor manejo y logging
    """
    if not question or not question.strip():
        return {
            "answer": "Por favor, escribe una pregunta.",
            "sources": []
        }

    # Retrieval
    context_results = search_context(question, index, chunks, model, top_k)

    # Generation
    answer = generate_response(question, context_results)

    return {
        "answer": answer,
        "sources": context_results
    }


# ─── EJECUCIÓN DIRECTA (para testing) ────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SISTEMA RAG - Modo CLI - VERSIÓN DEBUG")
    print("="*60 + "\n")

    index, chunks, model = initialize_rag()

    # ⭐ NUEVO: Mostrar resumen de chunks
    print(f"\n📊 RESUMEN DEL ÍNDICE:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Total vectores: {index.ntotal}")
    
    print(f"\n📋 CONTENIDO DE LOS CHUNKS:")
    for i, chunk in enumerate(chunks):
        preview = chunk[:200].replace('\n', ' ')
        print(f"\n   Chunk {i+1} ({len(chunk)} chars):")
        print(f"   {preview}...")

    print("\n✅ Sistema listo. Escribe 'salir' para terminar.\n")

    while True:
        question = input("\nTu pregunta: ").strip()

        if question.lower() in ["salir", "exit", "quit"]:
            print("¡Hasta luego!")
            break

        if not question:
            continue

        result = query_rag(question, index, chunks, model)
        print(f"\n🤖 Respuesta:\n{result['answer']}")
        
        print(f"\n📚 Fuentes usadas:")
        for i, (chunk, score) in enumerate(result['sources'], 1):
            preview = chunk[:100].replace('\n', ' ')
            print(f"   {i}. [Score: {score:.4f}] {preview}...")
        
        print("-" * 50)

        