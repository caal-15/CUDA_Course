
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
int main(){
  int scale = 1;
  int delta = 0;
  int ddepth = CV_8UC1;
  Mat image;
  image = imread("img1.jpg",0); // Read the file
  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  /// Gradient X
  // ( src , grad_x, ddepth,dx,dy,scale,delta, BORDER_DEFAULT );
  Sobel( image, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  /// Gradient Y
  //Sobel( image, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  imwrite("gato.png",grad_x);
  return 0;
}
