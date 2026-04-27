import pickle

try:
    with open('data/chunks.pkl', 'rb') as f:
        chunks = pickle.load(f)
    
    print(f"✅ Total chunks cargados: {len(chunks)}\n")
    
    for i, chunk in enumerate(chunks):
        print(f"=== Chunk {i+1} (longitud: {len(chunk)} caracteres) ===")
        # Muestra los primeros 300 caracteres del chunk
        preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
        print(preview)
        print("-" * 80)
        
except FileNotFoundError:
    print("❌ Error: No se encontró 'data/chunks.pkl'. Asegúrate de haber reconstruido el índice.")
except Exception as e:
    print(f"❌ Error al cargar chunks: {e}")

    