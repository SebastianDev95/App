import cv2
import os
from flask import Flask, Response, render_template, request, jsonify, redirect, url_for
from ultralytics import YOLO

# --- 1. CONFIGURACIÓN Y VARIABLES GLOBALES ---

app = Flask(__name__)

# Configuración de la carpeta de subida
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Crea la carpeta si no existe

print("Cargando modelo YOLOv8...")
model = YOLO('yolov8n.pt')
clases_a_detectar = [0, 2, 3, 5, 7] 

# Variable global para guardar las estadísticas
# Usamos un diccionario para poder añadir más datos en el futuro
current_stats = {
    "personas": 0,
    "vehiculos": 0
}

# --- 2. GENERADORES DE VIDEO ---

def process_frame(frame):
    """
    Función auxiliar para procesar un frame y actualizar las estadísticas.
    Esto evita repetir código.
    """
    global current_stats # Usamos la variable global
    
    results = model(frame, conf=0.4, classes=clases_a_detectar)
    
    # DIBUJAMOS LAS CAJAS, PERO NO EL TEXTO
    frame_con_detecciones = results[0].plot(labels=False, conf=False) # Solo cajas
    
    # Lógica de conteo
    conteo_personas = 0
    conteo_vehiculos = 0
    for box in results[0].boxes:
        clase = int(box.cls[0])
        if clase == 0:
            conteo_personas += 1
        elif clase in clases_a_detectar:
            conteo_vehiculos += 1
            
    # Actualizamos la variable global
    current_stats = {"personas": conteo_personas, "vehiculos": conteo_vehiculos}

    return frame_con_detecciones


def generar_frames_webcam():
    """Generador para el stream de la WEBCAM."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        return

    print("Webcam abierta. Empezando streaming...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error de captura de la webcam.")
            break 
        
        # Procesar el frame y actualizar stats
        frame_procesado = process_frame(frame)
        
        # Codificar y 'servir' el frame
        ret, buffer = cv2.imencode('.jpg', frame_procesado)
        if not ret: continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generar_frames_file(filename):
    """Generador para un ARCHIVO DE VIDEO."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {filepath}")
        return
        
    print(f"Procesando video {filename}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video.")
            break 
        
        # Procesar el frame y actualizar stats
        frame_procesado = process_frame(frame)
        
        # Codificar y 'servir' el frame
        ret, buffer = cv2.imencode('.jpg', frame_procesado)
        if not ret: continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- 3. RUTAS DE API Y SUBIDA ---

@app.route('/api/stats')
def api_stats():
    """
    Esta es la API. JavaScript le preguntará a esta ruta
    por los últimos conteos.
    """
    return jsonify(current_stats)

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Maneja la subida del archivo de video.
    """
    if 'video' not in request.files:
        return "No se encontró el archivo", 400
    
    file = request.files['video']
    
    if file.filename == '':
        return "No se seleccionó ningún archivo", 400
        
    if file:
        # Guardamos el archivo de forma segura
        filename = file.filename 
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Redirigimos a la nueva página de demo
        return redirect(url_for('demo', filename=filename))

# --- 4. RUTAS DE PÁGINA (HTML) ---

@app.route('/')
def index():
    """Página principal (usa la webcam)."""
    return render_template('index.html') 

@app.route('/demo/<filename>')
def demo(filename):
    """Página de demo (usa el video subido)."""
    return render_template('demo.html', video_filename=filename)

@app.route('/video_feed_webcam')
def video_feed_webcam():
    """Stream para la webcam."""
    return Response(generar_frames_webcam(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_file/<filename>')
def video_feed_file(filename):
    """Stream para el archivo de video."""
    return Response(generar_frames_file(filename), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# ... (todas tus otras rutas e importaciones) ...

@app.route('/dashboard')
def dashboard():
    """
    Renderiza la nueva página de dashboard dedicada
    exclusivamente a los gráficos.
    """
    return render_template('dashboard.html')



# --- Punto de entrada ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)