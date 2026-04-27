"""
app.py - Interfaz Web con Streamlit - VERSIÓN CORREGIDA
========================================================
Cambios:
1. Mostrar advertencia si hay pocos chunks
2. Forzar reconstrucción del índice
"""

import streamlit as st
import time
from pathlib import Path

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""

  <style>
    /* 1. ELIMINAR LA MANCHA BLANCA/CREMA */
    [data-testid="stMain"] {
        background-color: transparent !important;
    }
    [data-testid="stMainBlockContainer"] {
        background-color: transparent !important;
        padding-top: 2rem;
    }

    /* 2. ARREGLAR EL COLOR DE LAS LETRAS */
    .stChatMessage p, 
    .stChatMessage li, 
    .stChatMessage span {
        color: #EAEAEA !important; 
    }
    .stChatMessage h1, 
    .stChatMessage h2, 
    .stChatMessage h3 {
        color: #FFFFFF !important;
    }

    /* 3. FONDO OSCURO EN EL PANEL IZQUIERDO */
    section[data-testid="stSidebar"] {
        background-color: #0E1117 !important;
    }

    /* 4. OCULTAR MENÚ Y CENTRAR (Lo que ya tenías) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stChat { max-width: 750px; margin: 0 auto; }
    .stChatInputContainer { max-width: 750px; margin: 0 auto; }
    </style>

""", unsafe_allow_html=True)

try:
    from rag import initialize_rag, query_rag, EMBEDDING_MODEL, OLLAMA_MODEL, TOP_K, MIN_SIMILARITY
except ImportError as e:
    st.error(f"❌ Error importando rag.py: {e}")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_rag_system():
    try:
        return initialize_rag()
    except Exception as e:
        return None, None, None, str(e)


with st.spinner("⚙️ Inicializando sistema RAG..."):
    result = load_rag_system()

if len(result) == 4:
    index, chunks, model, error_msg = result
else:
    index, chunks, model = result
    error_msg = None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🧠 RAG Assistant")
    st.caption("Powered by FAISS + HuggingFace + Ollama")
    st.divider()

    st.subheader("📊 Estado del Sistema")

    if index is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vectores", f"{index.ntotal:,}")
        with col2:
            st.metric("Chunks", f"{len(chunks):,}" if chunks else "0")

        # ⭐ NUEVO: Advertencia si hay pocos chunks
        if len(chunks) < 10:
            st.warning(f"⚠️ Solo {len(chunks)} chunks. Agrega más documentos o reduce CHUNK_SIZE")
        
        st.success("✅ Sistema operativo")
    else:
        st.error("❌ Error al inicializar")
        if error_msg:
            st.code(error_msg, language="text")
        st.info(
            "**Posibles soluciones:**\n\n"
            "1. Agrega archivos .txt o .pdf en la carpeta `/docs`\n"
            "2. Ejecuta `ollama serve` en otra terminal\n"
            "3. Verifica que instalaste todas las dependencias"
        )

    st.divider()

    st.subheader("⚙️ Configuración")

    top_k = st.slider(
        "Chunks a recuperar (top-k)",
        min_value=1, max_value=10, value=TOP_K,
        help="Cuántos fragmentos de contexto usar para generar la respuesta."
    )

    show_sources = st.checkbox("Mostrar fuentes", value=True)
    show_scores = st.checkbox("Mostrar scores de similitud", value=True)  # ⭐ Cambiado a True

    st.divider()

    st.subheader("🛠️ Stack Técnico")
    st.markdown(f"""
    | Componente | Tecnología |
    |---|---|
    | 🗄️ Vector DB | FAISS |
    | 🤗 Embeddings | `{EMBEDDING_MODEL}` |
    | 🦙 LLM | Ollama · `{OLLAMA_MODEL}` |
    | 🚀 Frontend | Streamlit |
    """)

    st.divider()

    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.stats = {"queries": 0, "total_time": 0.0}
        st.rerun()

    # ⭐ MEJORADO: Botón de reconstruir índice más claro
    if st.button("🔨 RECONSTRUIR ÍNDICE (forzar)", use_container_width=True, type="primary"):
        data_path = Path("data")
        if data_path.exists():
            import shutil
            shutil.rmtree(data_path)
            st.success("✅ Carpeta /data eliminada. Reconstruyendo...")
        st.cache_resource.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# CONTENIDO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

st.title("🩺 CEREBRO VETERINARIO MVP") 
st.caption("Haz preguntas sobre los documentos cargados en `/docs`")

if index is None:
    st.warning("⚠️ El sistema no está listo.")
    st.stop()

# ⭐ NUEVO: Advertencia prominente si hay pocos chunks
if len(chunks) < 10:
    st.markdown(
        f'<div class="warning-box">'
        f'⚠️ <strong>ATENCIÓN:</strong> Solo tienes <strong>{len(chunks)} chunks</strong> indexados. '
        f'Esto puede causar que no encuentre información relevante.<br><br>'
        f'<strong>Soluciones:</strong><br>'
        f'1. Agrega más documentos en la carpeta <code>/docs</code><br>'
        f'2. Haz clic en <strong>"RECONSTRUIR ÍNDICE"</strong> en el panel lateral'
        f'</div>',
        unsafe_allow_html=True
    )

if "stats" not in st.session_state:
    st.session_state.stats = {"queries": 0, "total_time": 0.0}

if st.session_state.stats["queries"] > 0:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Consultas en sesión", st.session_state.stats["queries"])
    with col2:
        avg_time = st.session_state.stats["total_time"] / st.session_state.stats["queries"]
        st.metric("Tiempo promedio", f"{avg_time:.1f}s")
    with col3:
        st.metric("Chunks indexados", f"{index.ntotal:,}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 **Soy el **CEREBRO VETERINARIO MVP.**\n\n"
            f"He indexado **{index.ntotal:,} fragmentos** de tus documentos. "
            "Puedo responder preguntas basándome exclusivamente en su contenido.\n\n"
            "¿En qué puedo ayudarte?"
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if (
            message["role"] == "assistant"
            and "sources" in message
            and show_sources
            and message["sources"]
        ):
            with st.expander(f"📚 Ver fuentes ({len(message['sources'])} fragmentos)"):
                for i, (chunk_text, score) in enumerate(message["sources"], 1):
                    score_color = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
                    header = f"**Fragmento {i}**"
                    if show_scores:
                        header += f" {score_color} `{score:.3f}`"
                    st.markdown(header)
                    preview = chunk_text[:400] + "..." if len(chunk_text) > 400 else chunk_text
                    st.markdown(f'<div class="source-card">{preview}</div>', unsafe_allow_html=True)


if question := st.chat_input("Escribe tu pregunta sobre los documentos..."):

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        start_time = time.time()
        status_placeholder = st.empty()

        status_placeholder.markdown("🔍 *Buscando contexto relevante...*")
        time.sleep(0.3)
        status_placeholder.markdown("🧠 *Generando respuesta con el LLM...*")

        try:
            result = query_rag(question, index, chunks, model, top_k)
            elapsed = time.time() - start_time

            status_placeholder.empty()
            st.markdown(result["answer"])
            st.caption(f"⏱️ Respuesta generada en {elapsed:.1f}s")

            # ⭐ NUEVO: Siempre mostrar fuentes con scores
            if result.get("sources"):
                sources = result["sources"]
                with st.expander(f"📚 Fuentes utilizadas ({len(sources)} fragmentos)"):
                    for i, (chunk_text, score) in enumerate(sources, 1):
                        score_color = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
                        header = f"**Fragmento {i}** {score_color} `Score: {score:.4f}`"
                        st.markdown(header)
                        preview = chunk_text[:400] + "..." if len(chunk_text) > 400 else chunk_text
                        st.markdown(f'<div class="source-card">{preview}</div>', unsafe_allow_html=True)

            st.session_state.stats["queries"] += 1
            st.session_state.stats["total_time"] += elapsed

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", [])
            })

        except Exception as e:
            status_placeholder.empty()
            error_text = f"❌ Error al procesar la consulta: {str(e)}"
            st.error(error_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_text,
                "sources": []
            })  

            

            