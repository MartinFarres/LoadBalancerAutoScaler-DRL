from flask import Flask, jsonify

app = Flask(__name__)

# Variable global para simular carga en RAM
memory_hog = []

@app.route("/")
def health_check():
    return jsonify({"status": "ok", "message": "Nodo activo"})

@app.route("/cpu")
def stress_cpu():
    # Bucle intensivo para estresar la CPU artificialmente
    resultado = 0
    for i in range(500_000):
        resultado += i ** 2
    return jsonify({"status": "warning", "message": "CPU estresada"})

@app.route("/ram")
def stress_ram():
    global memory_hog
    
    # Añade un bloque de ~10MB a la memoria
    bloque = "A" * (10 * 1024 * 1024)
    memory_hog.append(bloque)
    
    # Límite de seguridad: vaciar a los ~200MB para evitar que Docker mate el contenedor (OOM Kill)
    if len(memory_hog) > 20:
        memory_hog.clear()
        return jsonify({"status": "reset", "message": "Memoria liberada por seguridad"})
        
    return jsonify({"status": "warning", "message": f"RAM estresada. Bloques en memoria: {len(memory_hog)}"})