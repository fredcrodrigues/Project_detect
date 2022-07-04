## Detecção da Iris e Pupila em imagens

Este projeto é baseado em aprendizado profundo(IA) para detectar a íris e pupila em imagens da região dos olhos. A base de dados pública UTIRIS [link](https://utiris.wordpress.com/2014/03/04/university-of-tehran-iris-image-repository/) é selecionada para testar o projeto desenvolvido. A proposta consiste no processo de segmentação dessas regiões.

![Screenshot](/01.png)

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
