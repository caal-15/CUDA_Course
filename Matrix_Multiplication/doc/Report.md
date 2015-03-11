Multiplicación de Matrices en Cuda
====================================
##Introducción
En este repositorio se encuentran dos archivos fuente de multiplicación de matrices en C/CUDA, uno para multiplicación de matrices enteras y otro para matrices de punto flotante, se incluyen 3 tipos de multiplicación:

- Multiplicación Secuencial (en procesador).
- Multiplicación Paralela (Device).
- Multiplicación Paralela Usando Tiles (Device).

Actualmente el programa main realiza una serie de multiplicaciones de acuerdo a los datos de entrada, como se muestra en la sgte sección

##Uso 
./mul max_number step offset_A offset_B

./mul_float max_number step offset_A offset_B

Donde:
- max_number es el tamaño máximo de la matriz
- step es la diferencia de tamaños entre cada una de las matrices a multiplicar (se inicia multiplicando matrices llenadas aleatoriamente del tamaño del step, y se finaliza multiplicando matrices del tamaño max_number, con pasos de tamaño step).
- offset_A es un número que se le sumará al número de filas de A (es decir si se ingresa un max_number = 128, con step = 32, y un offset = 3, la primera matriz A será de tamaño 35 * 32, y la última de tamaño 131 * 128) para generar una matriz rectangular en vez de cuadrada.
- offset_B es un número que se le sumará al número de columnas de B para generar una matriz rectangular en vez de cuadrada.

##Consideraciones
Todas las multiplicaciones soportan multiplicación entre matrices de cualquier tamaño, excepto la versión tiled, en la cual las matrices deben ser simñetricas y múltiplos del tamaño del tile. Por este motivo todas las pruebas se realizaron así: ./mul max_number 32 64 64, donde los valores de max_number fueron: 128, 256, 512, 1024, 2048.

##Resultados Obtenidos

###Para enteros

####Secuencial vs Paralelo (Sin Tiling)
- Max A 128 * 192, B 192 * 128
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Int/seq_vs_con_128.png)
- Max A 320 * 256, B 256 * 320
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Int/seq_vs_con_256.png)
- Max A 576 * 512, B 512 * 576
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Int/seq_vs_con_512.png)
- Max A 1088 * 1024, B 1024 * 1088
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Int/seq_vs_con_1024.png)
- Max A 2112 * 2048, B 2048 * 2112
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Int/seq_vs_con_2048.png)

####Paralelo (Sin Tiling) vs Paralelo (Con Tiling)
- Max A 128 * 192, B 192 * 128
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Int/non_tiled_vs_tiled_128.png)
- Max A 320 * 256, B 256 * 320
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Int/non_tiled_vs_tiled_256.png)
- Max A 576 * 512, B 512 * 576
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Int/non_tiled_vs_tiled_512.png)
- Max A 1088 * 1024, B 1024 * 1088
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Int/non_tiled_vs_tiled_1024.png)
- Max A 2112 * 2048, B 2048 * 2112
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Int/non_tiled_vs_tiled_2048.png)

###Para Flotantes

####Secuencial vs Paralelo (Sin Tiling)
- Max A 128 * 192, B 192 * 128
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Float/seq_vs_con_128.png)
- Max A 320 * 256, B 256 * 320
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Float/seq_vs_con_256.png)
- Max A 576 * 512, B 512 * 576
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Float/seq_vs_con_512.png)
- Max A 1088 * 1024, B 1024 * 1088
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Float/seq_vs_con_1024.png)
- Max A 2112 * 2048, B 2048 * 2112
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Float/seq_vs_con_2048.png)

####Paralelo (Sin Tiling) vs Paralelo (Con Tiling)
- Max A 128 * 192, B 192 * 128
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Float/non_tiled_vs_tiled_128.png)
- Max A 320 * 256, B 256 * 320
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Float/non_tiled_vs_tiled_256.png)
- Max A 576 * 512, B 512 * 576
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Float/non_tiled_vs_tiled_512.png)
- Max A 1088 * 1024, B 1024 * 1088
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Float/non_tiled_vs_tiled_1024.png)
- Max A 2112 * 2048, B 2048 * 2112
![](https://raw.githubusercontent.com/caal-15/CUDA_Course/master/Matrix_Multiplication/doc/Plots_Float/non_tiled_vs_tiled_2048.png)

###Coeficientes de acelereción Obtenidos 

- Matrices 128 + offset 64 :
  - Enteros 
    - Secuencial / Paralelo
    - Paralelo No Tiled / Paralelo Tiled
  - Flotantes 
    - Secuencial / Paralelo
    - Paralelo No Tiled / Paralelo Tiled
- Matrices 256 + offset 64 :
  - Enteros 
    - Secuencial / Paralelo = 0.039388 / 0.001923 = 20.48x
    - Paralelo No Tiled / Paralelo Tiled = 0.001923 / 0.001069 = 1.8x
  - Flotantes 
    - Secuencial / Paralelo 
    - Paralelo No Tiled / Paralelo Tiled
- Matrices 512 + offset 64 :
  - Enteros 
    - Secuencial / Paralelo = 0.148064 / 0.007128 = 20.77x
    - Paralelo No Tiled / Paralelo Tiled = 0.007128 / 0.002922 = 2.45x
  - Flotantes 
    - Secuencial / Paralelo 
    - Paralelo No Tiled / Paralelo Tiled
- Matrices 1024 + offset 64 :
  - Enteros 
    - Secuencial / Paralelo = 7.67902 / 0.044369 = 173.07x
    - Paralelo No Tiled / Paralelo Tiled = 0.044369 / 0.015525 = 2.86x
  - Flotantes 
    - Secuencial / Paralelo 
    - Paralelo No Tiled / Paralelo Tiled
- Matrices 2048 + offset 64 :
  - Enteros 
    - Secuencial / Paralelo = 65.7892 / 0.302649 = 217.38x
    - Paralelo No Tiled / Paralelo Tiled = 0.302649 / 0.098106 = 3.08x
  - Flotantes 
    - Secuencial / Paralelo 
    - Paralelo No Tiled / Paralelo Tiled

##Conclusiones

- Debido a como funciona la GPU se ve un aumento casi lineal en el tiempo que le toma a la GPU multiplicar las matrices de acuerdo a su tamaño, mientras se ve uno exponencial en el de la CPU.
- El aumento de desempeño es inmediatamente evidente para tareas que contengan paralelismo de datos hasta 217x).
- Si bien el aumento no parece tan sustancial entre el algoritmo paralelo no tiled, y el tiled (3x), hay que considerar que esto significaría un algoritmo 3 veces mas rápido respecto al no tiled, y comparandolo con el secuencial significaría una mejora de alrededor de 600x.
- Mas aún los coeficientes de mejora de desempeño entre el no tiled y el tiled tienden a crecer con la talla de las matrices.
- Esta mejora viene con un costo en complejidad del algoritmo para el programador.
