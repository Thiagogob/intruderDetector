import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a webcam.")
    exit()


# history: Número de frames para considerar ao construir o modelo do background.
#          Um valor maior (ex: 500) significa que ele leva mais tempo para se adaptar,
#          mas é mais robusto contra ruídos.
# varThreshold: Limiar de variância. Quanto maior, menos sensível a mudanças.
#               Similar ao '25' ou '40' que usamos no threshold anterior.
# detectShadows: Se deve ou não detectar sombras (geralmente False para simplificar).
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)

print("Sistema de detecção de intrusos iniciado (MOG2). Pressione 'q' para sair.")

while True:
    ret, frame_atual = cap.read()
    if not ret:
        print("Falha ao receber frame. Saindo...")
        break

    
    # Isso retorna uma máscara de primeiro plano (foreground mask)
    # onde os objetos em movimento são brancos e o background é preto.
    fgmask = fgbg.apply(frame_atual)

   

    # Dilata a máscara para preencher buracos e agrupar áreas de movimento
    # 'iterations' pode ser ajustado para agrupar mais ou menos.
    dilatada = cv2.dilate(fgmask, None, iterations=3) # Aumentei as iterações para garantir agrupamento

    # Encontra os contornos na imagem dilatada
    contornos, _ = cv2.findContours(dilatada.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    intruso_detectado = False # Flag para saber se algum intruso foi detectado neste frame

    for contorno in contornos:
       
      
        if cv2.contourArea(contorno) < 10000: 
            continue

        (x, y, w, h) = cv2.boundingRect(contorno)
        cv2.rectangle(frame_atual, (x, y), (x + w, y + h), (0, 255, 0), 2) # Cor verde

        intruso_detectado = True # Um contorno grande o suficiente foi encontrado

    # Se intruso_detectado for True, exibe a mensagem
    if intruso_detectado:
        cv2.putText(frame_atual, "Intruso Detectado!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      

    # Exibe os frames processados
    cv2.imshow('Foreground Mask (MOG2)', fgmask) # Mostra a máscara gerada pelo MOG2
    cv2.imshow('Dilatada', dilatada)
    cv2.imshow('Detector de Intrusos', frame_atual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()