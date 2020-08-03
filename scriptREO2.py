print('a) Apresente a imagem e as informações de número de linhas e colunas; número de canais e número total de pixels;')
import cv2
import numpy as np
nome_arquivo = "arroz.png"
imgbgr = cv2.imread(nome_arquivo,1) # Carrega imagem (0 - Binária e Escala de Cinza; 1 - Colorida (BGR))
img_arroz = cv2.cvtColor(imgbgr,cv2.COLOR_BGR2RGB) #transformar em rgb
lin, col, canais = np.shape(img_arroz)
print('Tipo: ',img_arroz.dtype)
print('Número de linhas: ' + str(lin))
print('Número de colunas: ' + str(col))
print('Número de canais: ' + str(canais))
print('Dimensão:' + str (lin) + 'x' + str (col))
print('Portanto, número de pixels é: ' + str(lin*col))
print(img_arroz) #matriz da imagem #cada posição representa um pixel e recebe um valor
#Imagem
from matplotlib import pyplot as plt
import os
plt.figure('Imagem arroz')
plt.imshow(img_arroz,cmap="gray") # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
plt.title("Imagem arroz")
plt.show()
print('-'*100)
print('b)Faça um recorte da imagem para obter somente a área de interesse. Utilize esta imagem para a solução das próximas alternativas;')
# Recortar uma imagem
img_recorte = img_arroz[109:370,160:433]

print('INFORMAÇÕES IMAGEM RECORTE')
lin_r, col_r, canais_r = np.shape(img_recorte)
print('Dimensão: ' + str(lin_r) +' x '+ str(col_r))
print('Número de canais :' + str(canais_r))
plt.figure('Imagem Recortada')
fig = plt.figure('Imagem Recortada')
plt.imshow(img_recorte,cmap="gray") # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
plt.title("Imagem recortada")
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.show()
#nome = 'folha_recortada'
#fig.savefig((nome+'.png'), bbox_inches="tight")
#os.startfile(nome+'.png')
plt.figure('Comparação sem e com recorte')
plt.subplot(1,2,1)
plt.imshow(img_arroz) # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
plt.title("Sem recorte")
plt.colorbar(orientation = 'horizontal')


plt.subplot(1,2,2)
plt.imshow(img_recorte) # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
plt.title("Com recorte")
plt.colorbar(orientation = 'horizontal')
plt.show()
#nome = 'Comparação sem e com recorte'
#fig.savefig((nome+'.png'), bbox_inches="tight")
#os.startfile(nome+'.png')
print('-'*100)
print('c)Converta a imagem colorida para uma de escala de cinza (intensidade) e a apresente utilizando os mapas de cores “Escala de Cinza” e “JET”;')
# Apresentar imagens no matplotlib
img_arrozcinza = cv2.cvtColor(img_recorte,cv2.COLOR_BGR2GRAY)
print(img_arrozcinza)
#nome = 'img_arrozcinza'
#fig.savefig((nome+'.png'), bbox_inches="tight")
#os.startfile(nome+'.png')
plt.figure('Imagens')
plt.subplot(1,2,1)
plt.imshow(img_arrozcinza,cmap="gray") # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
#plt.xticks([]) # Eliminar o eixo X
#plt.yticks([]) # Eliminar o eixo Y
plt.title("Escala de Cinza")
plt.colorbar(orientation = 'horizontal')


plt.subplot(1,2,2)
plt.imshow(img_arrozcinza,cmap="jet") # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
#plt.xticks([]) # Eliminar o eixo X
#plt.yticks([]) # Eliminar o eixo Y
plt.title("JET")
plt.colorbar(orientation = 'horizontal')
plt.show()

print('-'*100)
print('d) Apresente a imagem em escala de cinza e o seu respectivo histograma; Relacione o histograma e a imagem')
histograma = cv2.calcHist([img_arrozcinza],[0],None,[256],[0,256])
dim = len(histograma)
print ('Dimensão do histograma: ' + str(dim))

plt.figure('Imagens')
plt.subplot(1,2,1)
plt.imshow(img_arrozcinza,cmap="gray")
plt.title("Escala de Cinza")

plt.subplot(1,2,2)
plt.plot(histograma,color = 'black')
plt.title("Histograma")
plt.xlabel("Valores de pixels")
plt.ylabel("Número de pixels")
plt.show()

dimen = len(histograma)
print('-'*100)
print('e) Utilizando a imagem em escala de cinza (intensidade) realize a segmentação da imagem de modo a remover o fundo da imagem utilizando um limiar manual e o limiar obtido pela técnica de Otsu. Nesta questão apresente o histograma com marcação dos limiares utilizados, a imagem limiarizada (binarizada) e a imagem colorida final obtida da segmentação. Explique os resultados')
#imagens utilizadas
imgbgr = cv2.imread(nome_arquivo,1) # Carrega imagem (0 - Binária e Escala de Cinza; 1 - Colorida (BGR))
img_arroz = cv2.cvtColor(imgbgr,cv2.COLOR_BGR2RGB)
histograma = cv2.calcHist([img_arrozcinza],[0],None,[256],[0,256])

# Histograma da imagem em escala de cinza
histo_cinza = cv2.calcHist([img_arrozcinza],[0], None, [256], [0, 256])

# Limiarização manual (Thresholding)
valor_limiar = 130  # Valor do limiar
(L, img_limiar) = cv2.threshold(img_arrozcinza, valor_limiar, 255, cv2.THRESH_BINARY)
(LI, img_limiar_invertida) = cv2.threshold(img_arrozcinza, valor_limiar, 255, cv2.THRESH_BINARY_INV)
#invertida: abaixo do limiar recebe 255
# Limiarização (Thresholding) da imagem e escala de cinza pela técnica de Otsu
(LO, img_otsu) = cv2.threshold(img_arrozcinza, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Apresentando as imagens
plt.figure('Limiarização e')
plt.subplot(3,3,1)
plt.imshow(img_recorte)  # Plota a imagem em RGB
plt.xticks([])
plt.yticks([])
plt.title('Imagem em RGB')

plt.subplot(3,3,2)
plt.imshow(img_arrozcinza, cmap="gray")
plt.xticks([])
plt.yticks([])
plt.title('Imagem em escala de cinza')

plt.subplot(3,3,3)
plt.plot(histo_cinza, color="black")
plt.axvline(x=valor_limiar, color="red")  # Coloca uma reta vertical do limiar
plt.title('Histograma em escala de cinza',)
plt.xlim([0, 256])  # O eixo X vai variar de 0 a 256
plt.xlabel('Valores de pixels')
plt.ylabel('Número de pixels')

# Imagem limiarizada manualmente e seu histograma
plt.subplot(3, 3, 4)
plt.imshow(img_limiar)  # Plota a imagem binária
plt.title('Imagem binária')

plt.subplot(3, 3, 5)
plt.imshow(img_limiar_invertida)  # Plota a imagem binária invertida
plt.title ('Imagem binária invertida')

# Imagem obtida pela tecnica de OTSU e seu histograma
plt.subplot(3, 3, 6)
plt.imshow(img_otsu)
plt.title('Imagem OTSU')

plt.subplot(3, 3, 7)
plt.plot(histo_cinza, color="black")
plt.axvline(x=LO, color="red")  # Coloca uma linha vermelha no histograma para marcar o limiar
plt.title('Histograma - OTSU')
plt.xlim([0, 256])
plt.xlabel('Valores de pixels')
plt.ylabel('Número de pixels')

plt.show()

print('-' * 100)
print('f)Apresente uma figura contendo a imagem selecionada nos sistemas RGB, Lab, HSV e YCrCb.')
imgLAB = cv2.cvtColor(img_recorte, cv2.COLOR_BGR2Lab)  # Para Lab
imgHSV = cv2.cvtColor(img_recorte, cv2.COLOR_BGR2HSV)  # Para HSV
imgYCR = cv2.cvtColor(img_recorte, cv2.COLOR_BGR2YCrCb) #Para YCrCb

plt.figure('Imagens em RGB, Lab, HSV, YcrCb')
plt.subplot(2, 2, 1)
plt.imshow(img_recorte)
plt.xticks([])
plt.yticks([])
plt.title('Imagem do arroz em RGB')

plt.subplot(2, 2, 2)
plt.imshow(imgLAB)
plt.xticks([])
plt.yticks([])
plt.title('Imagem do arroz em Lab')

plt.subplot(2, 2, 3)
plt.imshow(imgHSV)
plt.xticks([])
plt.yticks([])
plt.title('Imagem do arroz em HSV')

plt.subplot(2, 2, 4)
plt.imshow(imgYCR)
plt.xticks([])
plt.yticks([])
plt.title('Imagem do arroz em YCrCb')
plt.show()
print('g)Apresente uma figura para cada um dos sistemas de cores (RGB, HSV, Lab e YCrCb) contendo a imagem de cada um dos canais e seus respectivos histogramas.')

hist_red = cv2.calcHist([img_recorte], [0], None, [256], [0,256])
hist_green = cv2.calcHist([img_recorte], [1], None, [256], [0,256])
hist_blue = cv2.calcHist([img_recorte], [2], None, [256], [0,256])


plt.figure('Sistemas de cores RGB')
plt.subplot(3, 4, 1)
plt.imshow(img_recorte)  # Plotando a imagem em RGB
plt.xticks([])  # Eliminar o eixo X
plt.yticks([])  # Eliminar o eixo Y
plt.title('Imagem RGB')

plt.subplot(3, 4, 2)
plt.imshow(img_recorte[:, :, 0])
plt.xticks([])
plt.yticks([])
plt.title('Canal Red - Vermelho')

plt.subplot(3, 4, 3)
plt.imshow(img_recorte[:, :, 1])
plt.xticks([])
plt.yticks([])
plt.title('Canal Green - Verde')

plt.subplot(3, 4, 4)
plt.imshow(img_recorte[:, :, 2])  # Obtendo a imagem do canal B "azul"
plt.xticks([])
plt.yticks([])
plt.title('Canal Blue - Azul')

plt.subplot(3, 4, 5)
plt.plot(hist_red, color="red")  # Obtendo o histograma do canal R "vermelho"
plt.title("Histograma - Red")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 4, 6)
plt.plot(hist_green, color="green")  # Obtendo o histograma do canal Green "verde"
plt.title("Histograma - Green")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 4, 7)
plt.plot(hist_blue, color="blue")  # Obtendo o histograma do canal Blue "azul"
plt.title("Histograma - Blue")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()

####Lab
hist_cL = cv2.calcHist([imgLAB], [0], None, [256], [0,256])
hist_cA = cv2.calcHist([imgLAB], [1], None, [256], [0,256])
hist_cB = cv2.calcHist([imgLAB], [2], None, [256], [0,256])

plt.figure('Lab')
plt.subplot(3, 4, 1)
plt.imshow(imgLAB)  # Plotando a imagem em Lab
plt.xticks([])
plt.yticks([])
plt.title('Imagem em Lab')

plt.subplot(3, 4, 2)
plt.imshow(imgLAB[:, :, 0])  # Obtendo a imagem do canal L
plt.xticks([])
plt.yticks([])
plt.title('Canal L')

plt.subplot(3, 4, 3)
plt.imshow(imgLAB[:, :, 1])  # Obtendo a imagem do canal a
plt.xticks([])
plt.yticks([])
plt.title('Canal A')

plt.subplot(3, 4, 4)
plt.imshow(imgLAB[:, :, 2])  # Obtendo a imagem do canal b
plt.xticks([])  # Eliminar o eixo X
plt.yticks([])  # Eliminar o eixo Y
plt.title('Canal B')

plt.subplot(3, 4, 5)
plt.plot(hist_cL, color="black")  # Obtendo o histograma do canal L
plt.title("Histograma do canal L")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")


plt.subplot(3, 4, 6)
plt.plot(hist_cA, color="black")  # Obtendo o histograma do canal a
plt.title("Histograma - a")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")


plt.subplot(3, 4, 7)
plt.plot(hist_cB, color="black")  # Obtendo o histograma do canal b
plt.title("Histograma - b")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()
#HSV
hist_H = cv2.calcHist([imgHSV], [0], None, [256], [0,256])
hist_S = cv2.calcHist([imgHSV], [1], None, [256], [0,256])
hist_V = cv2.calcHist([imgHSV], [2], None, [256], [0,256])

plt.figure('HSV')
plt.subplot(3, 4, 1)
plt.imshow(imgHSV)
plt.xticks([])
plt.yticks([])
plt.title('Imagem HSV')

plt.subplot(3, 4, 2)
plt.imshow(imgHSV[:, :, 0])
plt.xticks([])
plt.yticks([])
plt.title('Canal H')

plt.subplot(3, 4, 3)
plt.imshow(imgHSV[:, :, 1])  # Obtendo a imagem do canal S
plt.xticks([])
plt.yticks([])
plt.title('Canal S')

plt.subplot(3, 4, 4)
plt.imshow(imgHSV[:, :, 2])
plt.xticks([])
plt.yticks([])
plt.title('Canal V')

plt.subplot(3, 4, 5)
plt.plot(hist_H, color="black")
plt.title("Histograma - H")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")


plt.subplot(3, 4, 6)
plt.plot(hist_S, color="black")
plt.title("Histograma - S")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 4, 7)
plt.plot(hist_V, color="black")
plt.title("Histograma - V")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()

#YCrCb
hist_Y = cv2.calcHist([imgYCR], [0], None, [256], [0, 256])
hist_Cr = cv2.calcHist([imgYCR], [1], None, [256], [0, 256])
hist_Cb = cv2.calcHist([imgYCR], [2], None, [256], [0, 256])

plt.figure('YCrCb')
plt.subplot(3, 4, 1)
plt.imshow(imgYCR)  # Plotando a imagem em YCrCb
plt.xticks([])
plt.yticks([])
plt.title('Imagem YCrCb')

plt.subplot(3, 4, 2)
plt.imshow(imgYCR[:, :, 0])  # Obtendo a imagem do canal Y
plt.xticks([])
plt.yticks([])
plt.title('Canal Y')

plt.subplot(3, 4, 3)
plt.imshow(imgYCR[:, :, 1])  # Obtendo a imagem do canal Cr
plt.xticks([])
plt.yticks([])
plt.title('Canal Cr')

plt.subplot(3, 4, 4)
plt.imshow(imgYCR[:, :, 2])  # Obtendo a imagem do canal Cb
plt.xticks([])
plt.yticks([])
plt.title('Canal Cb')

plt.subplot(3, 4, 5)
plt.plot(hist_Y, color="black")  # Obtendo o histograma do canal Y
plt.title("Histograma - Y")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 4, 6)
plt.plot(hist_Cr, color="black")  # Obtendo o histograma do canal Cr
plt.title("Histograma - Cr")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")


plt.subplot(3, 4, 7)
plt.plot(hist_Cb, color="black")
plt.title("Histograma - Cb")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()
print('-'*100)
print('h) Encontre o sistema de cor e o respectivo canal que propicie melhor segmentação da imagem de modo a remover o fundo a imagem utilizando limiar manual e limiar obtido pela técnica de Otsu. Nesta questão apresente o histograma com marcação dos limiares utilizados, a imagem limiarizada (binarizada) e a imagem colorida final obtida da segmentação.Explique os resultados e sua escolha pelo sistema de cor e canal utilizado na segmentação. Nesta questão apresente a imagem limiarizada (binarizada) e a imagem colorida final obtida da segmentação.')
r,g,b = cv2.split(img_recorte)
hist_r = cv2.calcHist([r],[0], None, [256],[0,256])

# Limiarização manual (Thresholding)
vl = 140  # Valor do limiar
(LL, img_limiar2) = cv2.threshold(r, vl, 255, cv2.THRESH_BINARY)
# Limiarização (Thresholding) da imagem pela técnica de Otsu
(LOO, img_otsu2) = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img_segmentada_r = cv2.bitwise_and(img_recorte,img_recorte,mask=img_limiar2)
img_segmentada_rmanual = cv2.bitwise_and(img_recorte,img_recorte,mask=img_otsu2)


plt.figure('Letra g')
plt.subplot(4,4,1)
plt.imshow(img_recorte)
plt.title('RGB')
plt.xticks([])
plt.yticks([])

plt.subplot(4,4,2)
plt.imshow(r,cmap='gray')
plt.title('RGB - r')
plt.xticks([])
plt.yticks([])

plt.subplot(4,4,3)
plt.plot(hist_r,color = 'black')
plt.axvline(x=LL,color = 'r')
plt.title("Histograma - Manual")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(4,4,4)
plt.plot(hist_r,color = 'black')
plt.axvline(x=LOO,color = 'r')
plt.title("Histograma - Otsu")
plt.xlim([0,256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(4,4,5)
plt.imshow(img_limiar,cmap='gray')
plt.title('Limiar: ' + str(L))
plt.xticks([])
plt.yticks([])

plt.subplot(4,4,6)
plt.imshow(img_segmentada_r)
plt.title('Imagem segmentada (OTSU- canal r)')
plt.xticks([])
plt.yticks([])

plt.subplot(4,4,7)
plt.imshow(img_segmentada_rmanual)
plt.title('Imagem segmentada (MANUAL- canal r)')
plt.xticks([])
plt.yticks([])
plt.show()

print('i) Obtenha o histograma de cada um dos canais da imagem em RGB, utilizando como mascara a imagem limiarizada (binarizada) da letra h.')
hist_red = cv2.calcHist([img_segmentada_r], [0], img_limiar2, [256], [0,256])
hist_green = cv2.calcHist([img_segmentada_r], [1], img_limiar2, [256], [0,256])
hist_blue = cv2.calcHist([img_segmentada_r], [2], img_limiar2, [256], [0,256])


plt.figure('Sistemas de cores RGB')
plt.subplot(3, 4, 5)
plt.plot(hist_red, color="red")  # Obtendo o histograma do canal R "vermelho"
plt.title("Histograma - Red")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 4, 6)
plt.plot(hist_green, color="green")  # Obtendo o histograma do canal Green "verde"
plt.title("Histograma - Green")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")

plt.subplot(3, 4, 7)
plt.plot(hist_blue, color="blue")  # Obtendo o histograma do canal Blue "azul"
plt.title("Histograma - Blue")
plt.xlim([0, 256])
plt.xlabel("Valores Pixels")
plt.ylabel("Número de Pixels")
plt.show()
print('j) Realize operações aritméticas na imagem em RGB de modo a realçar os aspectos de seu interesse. Exemplo (2*R-0.5*G')
# Operações nos canais da imagem
img_j = 1.7*img_arroz[:, :, 0] - 1.2 * img_arroz[:, :, 1]

# Converção da variavel para inteiro de 8 bits
img_j2 = img_j.astype(np.uint8)

# Histograma
histj = cv2.calcHist([img_j2], [0], None, [256], [0, 256])

# Limiarização de Otsu
(LLL, img_otsu3) = cv2.threshold(img_j2, 0, 256, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Segmentação da imagem com mascara
img_segmentadaj = cv2.bitwise_and(img_arroz, img_arroz, mask=img_otsu3)

# Apresentando a imagem
plt.figure('Imagens letra J')
plt.subplot(2, 3, 1)
plt.imshow(img_arroz, cmap='gray')
plt.title('RGB')

plt.subplot(2, 3, 2)
plt.imshow(img_j2, cmap='gray')
plt.title('B - 1,2*G')

plt.subplot(2, 3, 3)
plt.plot(histj, color='black')
# plt.axline(x=LLL color='black')
plt.xlim([0, 256])
plt.xlabel('Valores de pixels')
plt.xlabel('Número de pixels')

plt.subplot(2, 3, 4)
plt.imshow(img_otsu3, cmap='gray')
plt.title('Imagem binária')

plt.subplot(2, 3, 5)
plt.imshow(img_segmentadaj, cmap='gray')
plt.title('Imagem segmentada com mascara')
plt.xticks([])
plt.yticks([])

plt.show()
