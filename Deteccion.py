# Importamos las librerias
import cv2
import mediapipe as mp
#import pywavefront import Wavefront #para generar modelo 3d

# Declaramos la deteccion de rostros
detrostro = mp.solutions.face_detection
rostros = detrostro.FaceDetection(min_detection_confidence= 0.5, model_selection=0)
# Dibujo
imagenrostro = mp.solutions.drawing_utils

# Declaramos la deteccion de malla facial
detmalla =mp.solutions.face_mesh
malla =detmalla.FaceMesh(max_num_faces= 1, min_detection_confidence= 0.5)
# Dibujo
imagenmalla = mp.solutions.drawing_utils
modelorostro =dibmalla.DrawingSpec(thickness= 1, circle_radius=1)

# Crear el objeto de captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3,1280)
cap.set(4,720)


# Procesamiento en tiempo real
while True:
    #Leer la camara
    ret, frame = cap.read()
    # Conversion de color
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    #DETECCION DE ROSTRO   
    # Mediante Mediapipe buscamos los rostros
    resrostros = rostros.process(rgb)

    if resrostros.detections is not None: #Si existen resultados
        # Registramos
        for rostro in resrostros.detections:
            # Dibujamos rostro
            imagenrostro.draw_detection(frame, rostro, imagenrostro.DrawingSpec(color=(0, 255, 0)))
            
            
    #MALLA DE MODELO FACIL        
    # Mediante Mediapipe buscamos la malla del rostro
    resmalla = malla.process(rgb)

    if resmalla.multi_face_landmarks:#Si existen resultados
        for mesh in resmalla.multi_face_landmarks:
            # Dibujamos malla
            imagenmalla.draw_landmarks(frame, mesh, detmalla.FACEMESH_TESSELATION, modelorostro, modelorostro)
            """PARA GENERAR UN MODELO 35 (NO FINZALIZADO)
            for face_landmarks in resmalla.multi_face_landmarks:
                # Crear una lista de vértices para el modelo Wavefront (.obj)
                vertices = []
                for landmark in face_landmarks.landmark:
                    # Obtener las coordenadas 3D de cada punto de referencia
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

            TERMINA AQUI"""
    
     # Mostrar la imagen
    cv2.imshow("Modelado cara", frame)
            
    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
