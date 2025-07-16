import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a webcam.")
    exit()

# Inicializa o frame de referência (background)
# Capturamos o primeiro frame e o tratamos como o background "sem intrusos"
ret, frame_ref = cap.read()
if not ret:
    print("Não foi possível capturar o frame de referência.")
    exit()

# Converte para escala de cinza e aplica borramento no frame de referência
frame_ref_gray = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
frame_ref_blur = cv2.GaussianBlur(frame_ref_gray, (21, 21), 0)

print("Sistema de detecção de intrusos iniciado. Pressione 'q' para sair.")

while True:
    ret, frame_atual = cap.read()
    if not ret:
        print("Falha ao receber frame. Saindo...")
        break

    # Converte o frame atual para escala de cinza e aplica borramento
    frame_atual_gray = cv2.cvtColor(frame_atual, cv2.COLOR_BGR2GRAY)
    frame_atual_blur = cv2.GaussianBlur(frame_atual_gray, (21, 21), 0)

    # Calcula a diferença absoluta entre o frame de referência e o frame atual
    diferenca = cv2.absdiff(frame_ref_blur, frame_atual_blur)

    # Aplica limiarização para destacar as áreas de diferença
    # O valor 25 define o limiar: pixels com diferença acima de 25 se tornam brancos
    # cv2.THRESH_BINARY é o tipo de limiarização
    # 255 é o valor máximo (branco) para pixels que excedem o limiar
    _, thresh = cv2.threshold(diferenca, 50, 255, cv2.THRESH_BINARY)

    # Dilata a imagem para preencher buracos e agrupar áreas de movimento
    # Isso ajuda a ter contornos mais contínuos
    dilatada = cv2.dilate(thresh, None, iterations=2)

    # Encontra os contornos na imagem dilatada
    # cv2.RETR_EXTERNAL recupera apenas os contornos externos
    # cv2.CHAIN_APPROX_SIMPLE compacta os pontos dos contornos
    contornos, _ = cv2.findContours(dilatada.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Itera sobre os contornos encontrados
    for contorno in contornos:
        # Se o contorno for muito pequeno, ignora (provavelmente ruído)
        if cv2.contourArea(contorno) < 2000: # Ajuste este valor conforme a necessidade
            continue

        # Calcula o retângulo delimitador para o contorno
        (x, y, w, h) = cv2.boundingRect(contorno)
        # Desenha o retângulo no frame original para indicar o movimento
        cv2.rectangle(frame_atual, (x, y), (x + w, y + h), (0, 255, 0), 2) # Cor verde

        # Opcional: Você pode adicionar texto para indicar "Intruso Detectado!"
        cv2.putText(frame_atual, "Intruso Detectado!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Exibe os frames processados
    cv2.imshow('Diferenca', diferenca)
    cv2.imshow('Threshold', thresh)
    cv2.imshow('Dilatada', dilatada)
    cv2.imshow('Detector de Intrusos', frame_atual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()