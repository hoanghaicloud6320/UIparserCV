#include <opencv2/imgcodecs.hpp>

#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: opencv_probe <image-path>\n";
    return 2;
  }

  const cv::Mat image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
  if (image.empty()) {
    std::cerr << "failed to load image: " << argv[1] << '\n';
    return 1;
  }

  std::cout << "path: " << argv[1] << '\n'
            << "width: " << image.cols << '\n'
            << "height: " << image.rows << '\n'
            << "channels: " << image.channels() << '\n'
            << "depth: " << image.depth() << '\n';

  return 0;
}

