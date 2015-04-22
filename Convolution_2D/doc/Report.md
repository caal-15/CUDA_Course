Convolución en dos dimensiones en CUDA
======================================
##Introducción
En este repositorio se presentan tres implementaciones básicas de la convolución en 2 Dimensiones (específicamente usando el filtro de Sobel) sobre imágenes (limitado a imagenes en escala de grises) en CUDA:
-Convolución ingenua usando memoria Global.
-Convolución ingenua usando memoria Constante.
-Convolución Tiled usando además memoria constante.

##Dependencias
En este ejemplo particular se aprovechan algunas de las funcionalidades de la librería OpenCV como implementación secuencial del filtro de Sobel que esta provee, además de funcionalidades de lectura y escritura de imágenes. Para instalar en Debian:
  $sudo aptitude update
  $sudo aptitude install libopencv-dev
  
CMake para la creación del makefile:
  $sudo aptitude install cmake
  
##Compilación y uso
Para compilar, estando en la carpeta Convolution_2D:
  $mkdir build && cd build
  $cmake ../
  $make
Para correr el programa
  $./convolution2d
El comando correrá cada una de las implementaciones de la convolución usando como máscara el filtro de Sobel sobre las 6 imágenes que contiene la carpeta images, y dejará los resultados de cada implementación en la carpeta outputs. Se generarán además varios archivos en la carpeta build con propósito de realizar gráficas con ellos.
Para generar las gráficas:
  $cd ..
  $python plotter.py && python barplotter.py
