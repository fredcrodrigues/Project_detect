## Detecção da Iris e Pupila em imagens

Este projeto apresenta a detetcção da iris e pupila em imagens, baseado em aprendizado profundo. São usadas para treinamento a base de dados pública UTIRIS [link](https://utiris.wordpress.com/2014/03/04/university-of-tehran-iris-image-repository/), que contém imagens de pupilas diltadas ou contraidas. A proposta e baseada no processo de seguementação dessas regiões usando o conjunto de daos adquiridos.

![Screenshot](/data/test/image/01.JPG)
![Screenshot](/data/test/mask_iris/01.png)
![Screenshot](/data/test/mask_pupil/01.png)

## Execução do Projeto

Para testar o projeto é necessario configurar o ambiente da seguinte forma:

```bash
    tensorflow = 2.4
    opencv-python = 4.6.0
    numpy = 1.21
    python = 3.8.8
``` 

O Coeficiente de Dice é calculado para cada imagem durante o processo de segmentação. Está disponível apenas uma pequena porção do projeto. Para ter acesso aos dados da base UTIRIS rotulados entre em contato atraves do email: rodrigues.fredson@discente.ufma.br. 


## Detetcção 
O algoritmo realiza a detecção da iris e pupila em imagens, e pode ser aplicada em imagens de videos. 

![Screenshot](/results/detect/bbox_0.png)
![Screenshot](/results/detect/bbox_13.png)
