from imutils import paths
import face_recognition
import pickle
from cv2 import cv2
import os

# cria lista com as imagens do dataset
imagePaths = list(paths.list_images('C:\\repository\\imagens\\rostos'))

# caminho do arquivo de encodings
encoder = 'C:\\Users\\Claudio\\Desktop\\encodings.pickle'

# listas que vão receber os encodings e os nomes de cada imagem
encodingsEncontrados = []
nomesEncontrados = []

for (i, imagePath) in enumerate(imagePaths):
    # extrai o nome da imagem
    print("[INFO] processando imagem {}/{}".format(i + 1, len(imagePaths)))
    nome = imagePath.split(os.path.sep)[-2]

    # lê a imagem e converte do formato BGR para RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # encontra as coordenadas (x,y) da caixa ao redor da face localizada
    rosto = face_recognition.face_locations(rgb, model='hog')

    # extrai o vetor com os pontos da face encontrada na imagem
    encodings = face_recognition.face_encodings(rgb, rosto)

    for e in encodings:
        encodingsEncontrados.append(e)
        nomesEncontrados.append(nome)

    # salva uma lista de encodings + nomes no disco
    print("[INFO] salvando encodings...")
    data = {"encodings": encodingsEncontrados, "nomes": nomesEncontrados}
    f = open('encoder', "wb")
    f.write(pickle.dumps(data))
    f.close()
