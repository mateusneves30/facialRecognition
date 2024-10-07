import cv2
import numpy as np
import face_recognition as fr

# Lista para armazenar as codificações e os nomes das pessoas
known_face_encodings = []
known_face_names = []

# Carregar e codificar a imagem de referência (Mateus)
imgMateus = fr.load_image_file('Mateus.jpg')
imgMateus = cv2.cvtColor(imgMateus, cv2.COLOR_BGR2RGB)
encodeMateus = fr.face_encodings(imgMateus)[0]

# Adicionar a codificação e o nome à lista
known_face_encodings.append(encodeMateus)
known_face_names.append("Mateus")

# Carregar e codificar a imagem de outra pessoa (exemplo: João)
imgJoao = fr.load_image_file('Elon.jpg')
imgJoao = cv2.cvtColor(imgJoao, cv2.COLOR_BGR2RGB)
encodeJoao = fr.face_encodings(imgJoao)[0]

# Adicionar a codificação e o nome à lista
known_face_encodings.append(encodeJoao)
known_face_names.append("Elon")

# Acessar a webcam (normalmente, '0' indica a webcam padrão)
cap = cv2.VideoCapture(0)

# Variável para controlar a frequência de detecção
process_frame = True

while True:
    # Ler o frame da câmera
    ret, frame = cap.read()

    # Obter as dimensões do frame
    height, width, _ = frame.shape

    # Criar uma imagem branca com o mesmo tamanho do frame
    overlay = np.ones_like(frame, dtype=np.uint8) * 255

    # Definir os parâmetros do oval
    center = (int(width / 2), int(height / 2))
    axes = (int(width * 0.2), int(height * 0.4))  # largura e altura do oval

    # Desenhar o oval na imagem de sobreposição (overlay)
    cv2.ellipse(overlay, center, axes, 0, 0, 360, (0, 0, 0), -1)

    # Aplicar a máscara inversa na área fora do oval para que fique branca
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
    frame = cv2.bitwise_and(frame, mask)
    frame = cv2.add(frame, cv2.bitwise_and(overlay, cv2.bitwise_not(mask)))

    # Redimensionar o frame para acelerar o processamento
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Converter a imagem de BGR para RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_frame:
        # Detectar as faces e codificar
        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)

    # Alternar entre processar o frame e pular o próximo
    process_frame = not process_frame

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Comparar com todas as faces conhecidas
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        print(face_distances)

        # Inicializar o nome da pessoa detectada como "Desconhecido"
        name = "Desconhecido"
        cor = (0, 0, 255)

        # Verificar se há alguma correspondência
        if True in matches:
            # Obter o índice da face mais próxima
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                cor = (0, 255, 0)

        # Escalar a localização da face de volta ao tamanho original
        top, right, bottom, left = [v * 4 for v in face_location]

        # Desenhar um retângulo ao redor da face detectada
        cv2.rectangle(frame, (left, top), (right, bottom), cor, 2)

        # Adicionar o nome da pessoa detectada
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

    # Mostrar o frame resultante
    cv2.imshow('Reconhecimento Facial ao Vivo', frame)

    # Pressionar 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar as janelas
cap.release()
cv2.destroyAllWindows()
