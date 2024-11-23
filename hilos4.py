import threading
import time
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from sort import Sort
import yolov5
import pytesseract
import easyocr
import os
import re
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from yolov5.models. common import DetectMultiBackend
import warnings
import queue
import mysql.connector
from mysql.connector import Error

host = "localhost"
user = "javier"
password = "javier"
database = "sys" 
""" 
host = "database.c1ic2qca45m5.us-east-2.rds.amazonaws.com"
user = "admin"
password = "pfjessiryjavier"
database = "capstone_project" 
"""

# Dimensiones de la placa en milímetros
PLATE_WIDTH_MM = 303.6
PLATE_WIDTH_MM -= 80
PLATE_HEIGHT_MM = 151.2

pixelametro = (20/1000)/137 #metros

#PLATE_HEIGHT_MM -= 20
initial_time = datetime.now()

# Definir múltiples rangos de colores para el amarillo y el blanco en el espacio HSV
color_ranges = [
    (np.array([20, 100, 100]), np.array([30, 255, 255])),  # Amarillo Rango 1
    (np.array([15, 100, 100]), np.array([25, 255, 255])),  # Amarillo Rango 2
    (np.array([25, 100, 100]), np.array([35, 255, 255])),  # Amarillo Rango 3
    (np.array([20, 100, 50]), np.array([30, 255, 200])),
    (np.array([15, 100, 50]), np.array([25, 255, 180])),
    (np.array([20, 150, 40]), np.array([30, 255, 150])),
    (np.array([0, 0, 200]), np.array([180, 25, 255])),      # Blanco
    (np.array([0, 0, 220]), np.array([180, 30, 255])),
    (np.array([0, 0, 180]), np.array([180, 40, 255])),
    (np.array([0, 0, 160]), np.array([180, 50, 255])),

    (np.array([0, 0, 0]), np.array([180, 255, 50])),
    (np.array([0, 0, 0]), np.array([180, 255, 70])),
    (np.array([0, 0, 0]), np.array([180, 50, 100]))
]


plate_speeds = {}

DISTANCE_VECTOR = np.linspace(45, 75, num=2160)  # Suponiendo un ancho de imagen de 2160 píxeles

# Diccionario para almacenar el frame inicial de detección de la placa para cada track_id
plate_initial_frames = {}


# Lista global para almacenar los registros de los track_id desaparecidos
disappeared_track_ids = []

# Cola para pasar los datos del hilo secundario al hilo principal
frame_queue = queue.Queue()

# Variables globales para compartir datos entre hilos#
# frame_queue = []
results_vehicle_queue = []
results_plate_queue = []
lock = threading.Lock()

warnings.filterwarnings("ignore", category=FutureWarning)

initial_frame_number = 0

# Definir el frame final deseado
final_frame_number = 30

# Ruta video
video_path = "/home/javier/ProyectoFinal/Archivos finales/videos/Prueba2.MP4"

# Inicializar el lector de EasyOCR
reader = easyocr.Reader(['en'])

# Cargar el modelo preentrenado de YOLOv5 para la detección de matrículas
model = yolov5.load('/home/javier/ProyectoFinal/best.pt')


vehicle_speeds = {}
vehicle_plates = {}
vehicle_lanes = {}
vehicle_positions = {}

# Buffer para almacenar los últimos 5 frames
image_buffer = []

global track_id

global terminate_flag1 
terminate_flag1 = False

PRINTS = False  

def calculate_average_speed(speeds):
    if speeds:
        return abs(sum(speeds) / len(speeds),2)
    return 0


# Función para calcular la velocidad basada en el tamaño de la placa
def calPlateSpeed(plate_width_pixels, video_fps, frame_count,relacion,plate_y_position):
    # Supongamos una distancia promedio entre la cámara y los vehículos de 60
    DISTANCE_TO_CAMERA_M = 60
    DISTANCE_TO_CAMERA_M = DISTANCE_VECTOR[plate_y_position]

    # Calcular el tamaño de la placa en metros
    plate_width_meters = (PLATE_WIDTH_MM / 1000) / DISTANCE_TO_CAMERA_M * plate_width_pixels

    #print(plate_width_meters)

    #Velocidad = Distancia / Tiempo
    # Calcular la velocidad como un cambio en el tamaño de la placa entre frames consecutivos
    speed = 600 / video_fps * plate_width_meters * 3.6   # Convertir a km/h
    speed = round(abs(speed), 2)
    print(f"Speed: {speed}")
    return speed

cap = cv2.VideoCapture(video_path)

# Establecer el frame inicial
cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_number)

VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)


# Adquisición de video y reconocimiento de placas
def Hilo1():
    global terminate_flag1
    active_track_ids = set()


    #if __name__ == "__main__":
        #cap = cv2.VideoCapture("/home/javier/ProyectoFinal/videosfinales/Videos/MVI_7857.MP4")
        #initial_frame_number = 600 # Cambia este valor al frame deseado 2850

        
        #initial_frame_number = 570

        #cap = cv2.VideoCapture("/home/javier/ProyectoFinal/videosfinales/Videos/MVI_7862.MP4")
        #initial_frame_number = 529


    # Modelos YOLO
    model_vehicle = YOLO("yolov8n.pt")
    model_plate = YOLO("license_plate_detector.pt")

    tracker = Sort()
    prev_frame = None
    prev_boxes = {}

    # Leer el primer frame
    ret, initial_frame = cap.read()

    prev_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    initial_frame = prev_frame


    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        if frame is None or frame.size == 0:
            print("Frame vacío, saltando...")
            continue


        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if current_frame >= final_frame_number:
            print("Frame final alcanzado hilo 1")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype="uint8")
        for lower, upper in color_ranges:
            mask += cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        results_vehicle = model_vehicle(frame, stream=True, verbose=False)
        results_plate = model_plate(frame, stream=True, verbose=False)

        #hgerer

        for res in results_vehicle:
            filtered_indices = np.where((np.isin(res.boxes.cls.cpu().numpy(), [2, 5, 7])) & (res.boxes.conf.cpu().numpy() > 0.3))[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)

            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)

            for xmin, ymin, xmax, ymax, track_id in tracks:
                xc, yc = int((xmin + xmax) / 2), ymax
                vehicle_positions[track_id] = (xmin, ymin, xmax, ymax)
                if track_id not in vehicle_speeds:
                    vehicle_speeds[track_id] = []
                prev_boxes[track_id] = [xmin, ymin, xmax, ymax]
            
            current_track_ids = set(track_id for xmin, ymin, xmax, ymax, track_id in tracks)

            # Detectar track_ids desaparecidos
            disappeared_track_ids_set = active_track_ids - current_track_ids

            for track_id in disappeared_track_ids_set:
                if track_id in vehicle_speeds:
                    average_speed = calculate_average_speed(vehicle_speeds[track_id])
                    #print(f"Track ID {track_id} desapareció. Velocidad promedio: {average_speed:.2f} Km/h")
                    disappeared_track_ids.append((track_id,average_speed))

            # Actualizar track_ids activos
            active_track_ids = current_track_ids
        
        if track_id is not None:
            print(f"Track Ids: {track_id}")
            for plate_res in results_plate:
                plate_boxes = plate_res.boxes.xyxy.cpu().numpy().astype(int)
                for plate_box in plate_boxes:
                    p_xmin, p_ymin, p_xmax, p_ymax = plate_box
                    plate_width_pixels = p_xmax - p_xmin
                    plate_heigth_pixels = p_ymax - p_ymin
                    plate_x_position = (p_ymin + p_ymax) // 2  # Calcular la posición Y de la placa

                    pxc, pyc = int((p_xmin + p_xmax) / 2), p_ymax
                    
                    # Verificar si la placa está dentro de la región de algún vehículo en prev_boxes
                    for vehicle_id, (v_xmin, v_ymin, v_xmax, v_ymax) in prev_boxes.items():
                        print(f"Vehicle Id: {vehicle_id}")
                        #print(f"Vehicle: {v_xmin, v_ymin, v_xmax, v_ymax}")
                        #print(f"Plate: {p_xmin, p_ymin, p_xmax, p_ymax}")
                        if p_xmin >= v_xmin and p_ymin >= v_ymin and p_xmax <= v_xmax and p_ymax <= v_ymax:
                            # Almacenar el frame inicial de detección de la placa
                            if vehicle_id not in plate_initial_frames:
                                plate_initial_frames[vehicle_id] = current_frame

                            # Calcular la diferencia de frames
                            frame_count = current_frame - plate_initial_frames[vehicle_id]

                            relacion = (v_xmax - v_xmin) / (p_xmax - p_xmin)
                            
                            # Calcular velocidad basada en la placa
                            plate_speed = calPlateSpeed(plate_width_pixels, VIDEO_FPS, frame_count, relacion, plate_x_position)

                            if plate_speed > 50 and len(image_buffer)>3:
                                print("Velocidad maxima alcanzada")
                                try:
                                    connection = mysql.connector.connect(
                                        host=host,
                                        user=user,
                                        password=password,
                                        database=database
                                    )

                                    # Check for high speed and create a GIF if necessary
                                    gif_path = f'alert_{track_id}.gif'
                                    pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in image_buffer]
                                    pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], duration=200, loop=0)
                                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    with lock:
                                        plate = vehicle_plates.get(vehicle_id, "UNKNOWN")
                                    
                                    if len(plate) != 6:
                                        break

                                    # Convert GIF to blob
                                    with open(gif_path, 'rb') as f:
                                        gif_blob = f.read()

                                    if connection.is_connected():
                                        #cursor = connection.cursor()
                                        #insert_query = "INSERT INTO registro (placa, velocidad, timestamp, video) VALUES (%s, %s, %s, %s)"
                                        #record = (plate, str(average_speed), timestamp, gif_blob)

                                        # Debug prints to check types
                                        print(f"plate: {plate}, type: {type(plate)}")
                                        #print(f"average_speed: {str(average_speed)}, type: {type(str(average_speed))}")
                                        print(f"timestamp: {timestamp}, type: {type(timestamp)}")
                                        print(f"gif_blob: {type(gif_blob)}")

                                        #cursor.execute(insert_query, record)
                                        #connection.commit()
                                        print("Registro insertado exitosamente")

                                except Error as e:
                                    print("Error al conectar con MySQL", e)

                                finally:
                                    if connection.is_connected():
                                        #cursor.close()
                                        #connection.close()
                                        print("Conexión cerrada")

            frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
            result = cv2.resize(result, (0, 0), fx=0.3, fy=0.3)

        #cv2.imshow("Result", frame)
        # Add frame to queue
        frame_queue.put(frame)
        #frame_reduced = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
        image_buffer.append(frame)
        if PRINTS:
            print(f"buffer: {len(image_buffer)}")
        if len(image_buffer) > 46:
            image_buffer.pop(0) 
        

    cap.release()
    terminate_flag1 = True
    print("---------------")
    print("Hilo 1 terminado")
    print("---------------")
        

# Tracking
def Hilo2():
    print("Hilo 2 Iniciado ...")
   
    #cv2.waitKey(5000)
    print("Modelo cargado")
    # Configuración personalizada para Tesseract
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    print("Configuración personalizada de Tesseract cargada")

    # Expresión regular para validar el patrón LLLNNN
    pattern = re.compile(r'^[A-Z]{3}[0-9]{3}$')

    print("Expresión regular cargada")

    # Cargar el video
    cap = cv2.VideoCapture(video_path)

    print(f"Procesando video desde el frame {initial_frame_number} hasta el frame {final_frame_number}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_number)

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el archivo de video")
        exit()

    # Obtener las dimensiones del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Definir la ROI (por ejemplo, la mitad inferior del video)
    roi_top = 0
    roi_bottom = height #// 2
    roi_left = 0
    roi_right = width

    # Establecer la tasa de fotogramas a 24 fps
    fps = 24
    frame_interval = 1.0 / fps  # Intervalo de tiempo entre fotogramas en segundos
    print(f"Procesando video a {fps} FPS y resolución {width}x{height}")

    # Procesar cada fotograma del video
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if PRINTS:
            print(f"Procesando frame {frame_number}")

        if frame_number >= final_frame_number:
            break

        roi_frame = frame[roi_top:roi_bottom, roi_left:roi_right]
        results = model(roi_frame)

        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1 += roi_left
            y1 += roi_top
            x2 += roi_left
            y2 += roi_top
            plate_region = frame[y1:y2, x1:x2]
            margin_x = int((x2 - x1) * 0.1)
            margin_y = int((y2 - y1) * 0.1)
            x1 = max(x1 + margin_x, 0)
            y1 = max(y1 + margin_y, 0)
            x2 = min(x2 - margin_x, frame.shape[1])
            y2 = min(y2 - margin_y, frame.shape[0])
            plate_region = frame[y1:y2, x1:x2]
            # Buscar el track_id correspondiente
            track_id = None
            for tid, (vx1, vy1, vx2, vy2) in vehicle_positions.items():
                if vx1 <= x1 <= vx2 and vy1 <= y1 <= vy2:
                    track_id = tid
                    break

            if track_id is not None:
                # Procesar la placa
                plate_region = cv2.resize(plate_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                brightness = 50
                bright_image = cv2.convertScaleAbs(plate_region, alpha=1, beta=brightness)
                contrast = 1.5
                contrast_image = cv2.convertScaleAbs(bright_image, alpha=contrast, beta=0)
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                sharp_image = cv2.filter2D(contrast_image, -1, kernel)
                text_tesseract = pytesseract.image_to_string(sharp_image, config=custom_config).strip()
                result_easyocr = reader.readtext(sharp_image, detail=0)
                text_easyocr = result_easyocr[0] if result_easyocr else ""
                text_easyocr = ''.join(filter(lambda x: x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', text_easyocr))


                #vehicle_plates[track_id] = text_tesseract  # o text_easyocr
                with lock:
                    vehicle_plates[track_id] = text_easyocr  # o text_easyocr
                #vehicle_plates[track_id] = easyocr  # o text_easyocr
            

                if pattern.match(text_tesseract) and len(text_tesseract) == 6:
                    if PRINTS:
                        print(f'Tesseract detectó: {text_tesseract}')
                if pattern.match(text_easyocr) and len(text_easyocr) == 6:
                    if PRINTS:
                        print(f'EasyOCR detectó: {text_easyocr}')

                contains_similar_chars = 'Q' in text_tesseract or 'O' in text_tesseract or 'Q' in text_easyocr or 'O' in text_easyocr
                box_color = (0, 255, 0)
                if contains_similar_chars:
                    box_color = (0, 0, 255)
                    if PRINTS:
                        print("Caracteres alfanuméricos similares - Posibilidad alta de error")

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, text_tesseract, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
                cv2.putText(frame, text_easyocr, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
        # Add frame to queue
        #frame_queue.put(frame)

        

    cap.release()
    print("---------------")
    print("Hilo 2 terminado")
    print("---------------")

def Hilo3():
    print("Hilo 3 Iniciado ...")

    def nothing(x):
        pass

    def calculate_histograms(image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        
        h_peak = np.argmax(h_hist)
        s_peak = np.argmax(s_hist)
        v_peak = np.argmax(v_hist)
        
        return h_hist, s_hist, v_hist, h_peak, s_peak, v_peak

    def adjust_trackbar_values(h_peak, s_peak, v_peak):
        desired_h_peak = 20
        desired_s_peak = 64
        desired_v_peak = 30

        h_diff = desired_h_peak - h_peak
        s_diff = desired_s_peak - s_peak
        v_diff = desired_v_peak - v_peak

        l_h = max(0, h_diff)
        u_h = min(255, 255 + h_diff)
        l_s = max(0, s_diff)
        u_s = min(255, 255 + s_diff)
        l_v = max(0, v_diff)
        u_v = min(255, 255 + v_diff)

        return l_h, l_s, l_v, u_h, u_s, u_v

    # Transformar las coordenadas reales a la vista de pájaro
    def transform_real_positions(real_positions, matrix):
        transformed_positions = {"left": [], "middle": [], "right": []}
        for line_name, points in real_positions.items():
            for point in points:
                transformed_point = cv2.perspectiveTransform(np.array([[point]], dtype='float32'), matrix)
                transformed_positions[line_name].append((int(transformed_point[0][0][0]), int(transformed_point[0][0][1])))
        return transformed_positions

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()

    #cv2.namedWindow("Trackbars")
    #cv2.namedWindow("Bird's Eye View")
    #cv2.namedWindow("Sliding Windows")

    h_hist, s_hist, v_hist, h_peak, s_peak, v_peak = calculate_histograms(image)


    print(f'Hue peak: {h_peak}')
    print(f'Saturation peak: {s_peak}')
    print(f'Value peak: {v_peak}')

    l_h, l_s, l_v, u_h, u_s, u_v = adjust_trackbar_values(h_peak, s_peak, v_peak)

    """ cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 49, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 56, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", u_h, 255, nothing)
    cv2.createTrackbar("U - S", "Trackbars", u_s, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing) """

    # Loop para procesar el video y calcular la desviación estándar
    repetitions = 1
    for i in range(repetitions):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_number)  # Reset video to the beginning
        while True:
            success, image = vidcap.read()
            cv2
            if not success:
                break
            frame = cv2.resize(image, (640,480))

            frame_number = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
          
            if frame_number >= final_frame_number:
                break

            tl = (130, 0)
            bl = (0, 480)
            tr = (520, 0)
            br = (640, 480)

            pts1 = np.float32([tl, bl, tr, br]) 
            pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 
            
            matrix = cv2.getPerspectiveTransform(pts1, pts2) 
            inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
            transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))

            hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
            
            """ l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars") """
            
            lower = np.array([l_h,l_s,l_v])
            upper = np.array([u_h,u_s,u_v])
            mask = cv2.inRange(hsv_transformed_frame, lower, upper)

            histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
            midpoint = int(histogram.shape[0] / 2)
            left_base = np.argmax(histogram[:midpoint])
            right_base = np.argmax(histogram[midpoint:]) + midpoint

            y = 472
            lx = []
            mx = []
            rx = []

            msk = mask.copy()

            while y > 0:
                img = mask[y-20:y, left_base-25:left_base+25]
                contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"]/M["m00"])
                        cy = int(M["m01"]/M["m00"])
                        lx.append(left_base-25 + cx)
                        left_base = left_base-25 + cx

                middle_base = (left_base + right_base) // 2
                img = mask[y-20:y, middle_base-25:middle_base+25]
                contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"]/M["m00"])
                        cy = int(M["m01"]/M["m00"])
                        mx.append(middle_base-25 + cx)
                        middle_base = middle_base-25 + cx

                img = mask[y-20:y, right_base-25:right_base+25]
                contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"]/M["m00"])
                        cy = int(M["m01"]/M["m00"])
                        rx.append(right_base-25 + cx)
                        right_base = right_base-25 + cx

                cv2.rectangle(msk, (left_base-25, y), (left_base+25, y-20), (255,255,255), 2)
                cv2.rectangle(msk, (middle_base-25, y), (middle_base+25, y-20), (255,255,255), 2)
                cv2.rectangle(msk, (right_base-25, y), (right_base+25, y-20), (255,255,255), 2)
                y -= 20

            lane1_mask = np.zeros_like(transformed_frame)
            lane2_mask = np.zeros_like(transformed_frame)
            cv2.rectangle(lane1_mask, (left_base-25, 0), (middle_base+25, 480), (0, 255, 0), -1)
            cv2.rectangle(lane2_mask, (middle_base-25, 0), (right_base+25, 480), (255, 0, 0), -1)

            lane1_original = cv2.warpPerspective(lane1_mask, inv_matrix, (640, 480))
            lane2_original = cv2.warpPerspective(lane2_mask, inv_matrix, (640, 480))
            lanes_original = cv2.addWeighted(frame, 1, lane1_original, 0.3, 0)
            lanes_original = cv2.addWeighted(lanes_original, 1, lane2_original, 0.3, 0)

            cv2.putText(lanes_original, f"Repetition: {i+1}/{repetitions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            #cv2.imshow("Sliding Windows", msk)
            #cv2.imshow("Bird's Eye View", lanes_birdseye)
            #cv2.imshow("Lanes", lanes_original)

            # Asegurarse de que los trackbars sean editables
            #cv2.imshow("Trackbars", np.zeros((1, 400), np.uint8))
        
            # Add frame to queue
            #frame_queue.put(lanes_original)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    vidcap.release()
    cv2.destroyAllWindows()

    # Dibujar las líneas de desviación media en el frame final
    final_frame = lanes_original.copy()

    # Dibujar la línea de desviación media para el carril izquierdo
    for y, x in enumerate(lx):
        cv2.circle(final_frame, (x, y * 20), 5, (0, 0, 255), -1)

    # Dibujar la línea de desviación media para el carril medio
    for y, x in enumerate(mx):
        cv2.circle(final_frame, (x, y * 20), 5, (0, 255, 0), -1)

    # Dibujar la línea de desviación media para el carril derecho
    for y, x in enumerate(rx):
        cv2.circle(final_frame, (x, y * 20), 5, (255, 0, 0), -1)

    #frame_queue.put(final_frame)
    # Guardar el carril dependiendo del track_id
    for track_id in vehicle_speeds.keys():
        if track_id in lx:
            vehicle_lanes[track_id] = "left"
        elif track_id in mx:
            vehicle_lanes[track_id] = "middle"
        elif track_id in rx:
            vehicle_lanes[track_id] = "right"
    print("---------------")
    print("Hilo 3 terminado")
    print("---------------")
    

def display_frames():
    global terminate_flag1
    while not (terminate_flag1):
        #print (terminate_flag1, terminate_flag2, terminate_flag3)
        if not frame_queue.empty():
            frame = frame_queue.get()
            resized_frame = cv2.resize(frame, (1280, 720))  # Resize the frame to 640x480
            cv2.imshow("Result", resized_frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                terminate_flag1 = True 
                break
        else:
            cv2.waitKey(5)  # Añadir un pequeño tiempo de espera para evitar un bucle ocupado
    cv2.destroyAllWindows()
    print("display_frames terminado")

# Crear los hilos para cada tarea
hilo1 = threading.Thread(target=Hilo1)
hilo2 = threading.Thread(target=Hilo2)
hilo3 = threading.Thread(target=Hilo3)
display_thread = threading.Thread(target=display_frames)

# Iniciar los hilos
hilo1.start()
hilo2.start()
hilo3.start()
display_thread.start()

# Esperar a que todos los hilos terminen
hilo1.join()
hilo2.join()
hilo3.join()
display_thread.join()

print("Todas las tareas han terminado")