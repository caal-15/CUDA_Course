Convolución en dos dimensiones en CUDA
======================================
##Introducción
En este repositorio se presentan tres implementaciones básicas de la convolución en 2 Dimensiones (específicamente usando el filtro de Sobel) sobre imágenes (limitado a imagenes en escala de grises) en CUDA:
-Convolución ingenua usando memoria Global.
-Convolución ingenua usando memoria Constante.
-Convolución Tiled usando además memoria constante.

##Dependencias
En este ejemplo particular se aprovechan algunas de las funcionalidades de la librería OpenCV como implementación secuencial del filtro de Sobel que esta provee, además de funcionalidades de lectura y escritura de imágenes. Para instalar en Debian:

    sudo aptitude update
    sudo aptitude install libopencv-dev
  
CMake para la creación del makefile:

    sudo aptitude install cmake
  
##Compilación y uso
Para compilar, estando en la carpeta Convolution_2D:

    mkdir build && cd build
    cmake ../
    make

Para correr el programa

    ./convolution2d
  
El comando correrá cada una de las implementaciones de la convolución usando como máscara el filtro de Sobel sobre las 6 imágenes que contiene la carpeta images, y dejará los resultados de cada implementación en la carpeta outputs. Se generarán además varios archivos en la carpeta build con propósito de realizar gráficas con ellos.
Para generar las gráficas:

    cd ..
    python plotter.py && python barplotter.py
    
##Resultados
Como era de esperarse, la versión secuencial se queda atrás incluso en imágenes pequeñas (580 * 580 pixeles), a continuación se muestra una imagen comparando el tiempo de ejecución de cada algoritmo para cada una de las imágenes de entrada:

![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Convolution_2D/doc/Bar_All.png)

A su vez, como era de esperarse, dentro de las implementaciones concurrentes la mas rápida es la tiled, y esto se va viendo reflejado a medida que van subiendo el número de datos:

![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Convolution_2D/doc/Bar_Concurrent.png)

A continuación se muestra una tabla con los índices de aceleración obtenidos para cada imagen, de la implementación secuencial respecto a la concurrente con memoria global, de la concurrente con memoria global respecto a la concurrente constante, etc:

|Imágenes/Implementación|Secuencial|Memoria Global|Memoria Constante|Tiled + Mem. Constante|
|-----------------------|---------:|-------------:|----------------:|---------------------:|
|IMG1  (580 * 580)      |3.43x     |   |   |   |
|   |3.70x   |   |   |   |
|   |4.41x   |   |   |   |
|   |6.52x   |   |   |   |
|   |8.18x   |   |   |   |
|   |8.33x   |   |   |   |
