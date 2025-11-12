import cv2
from ultralytics import YOLO

# --- Carga del Modelo ---
print("Cargando modelo YOLOv8...")
model = YOLO('yolov8n.pt')


clases_a_detectar = [0, 2, 3, 5, 7] 

# --- NUEVO: Menú de selección de fuente ---
print("--- Prototipo de Detección de Tráfico ---")
print("¿Qué deseas procesar?")
print("  1: Webcam en vivo")
print("  2: Archivo de video")
opcion = input("Escribe 1 o 2: ")

source = None
is_webcam = False

if opcion == '1':
    source = 0  # El '0' es la webcam por defecto
    is_webcam = True
    print("Iniciando webcam...")
elif opcion == '2':
    video_path = input("Escribe el nombre del archivo (ej. trafico.mp4): ")
    source = video_path
    print(f"Cargando video '{video_path}'...")
else:
    print("Opción no válida. Saliendo.")
    exit()

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print(f"Error: No se pudo abrir la fuente: {source}")
    exit()

print("Procesando... Presiona 'q' en la ventana para salir.")

# --- Bucle Principal de Procesamiento (funciona para ambos) ---
while True:
    ret, frame = cap.read()
    
    # Si 'ret' es False, el video terminó o la webcam falló
    if not ret:
        if is_webcam:
            print("Error al leer frame de la webcam.")
        else:
            print("Fin del video.")
        break # Salir del bucle

    # 1. Detección de Objetos
    results = model(frame, conf=0.4, classes=clases_a_detectar)

    # 2. Dibujar las cajas y etiquetas
    frame_con_detecciones = results[0].plot()
    
    # 3. Conteo separado
    conteo_personas = 0
    conteo_vehiculos = 0
    for box in results[0].boxes:
        clase = int(box.cls[0])
        if clase == 0:
            conteo_personas += 1
        elif clase in [2, 3, 5, 7]:
            conteo_vehiculos += 1

    # 4. Añadir AMBOS conteos al frame
    cv2.putText(
        frame_con_detecciones, f'Personas: {conteo_personas}',
        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    cv2.putText(
        frame_con_detecciones, f'Vehiculos: {conteo_vehiculos}',
        (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )

    # 5. Mostrar el Video en una ventana
    cv2.imshow('Detector Flexible (Personas y Vehiculos) - Presiona q para salir', frame_con_detecciones)

    # 6. Condición de Salida
    # Para la webcam, (1) es espera en vivo.
    # Para el video, (1) lo reproduce a velocidad (casi) normal.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Limpieza Final ---
print("Cerrando detector...")
cap.release()
cv2.destroyAllWindows()