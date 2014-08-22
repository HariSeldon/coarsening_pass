#include "FileUtils.h"

#include <fstream>
#include <stdexcept>
#include <sstream>

// No copy constructor is called, I checked.
std::string readFile(const std::string& filePath) {
  std::ifstream fileStream(filePath.c_str());
  verifyFileStreamOpen(filePath, fileStream);
  return readWholeFileStream(fileStream);
}

inline std::string readWholeFileStream(std::ifstream& fileStream) {
  std::stringstream stringStream;
  stringStream << fileStream.rdbuf();
  fileStream.close();
  return stringStream.str();
}

inline void verifyFileStreamOpen(const std::string& filePath,
                                 const std::ifstream& fileStream) {
  if(!fileStream.is_open())
    throw std::runtime_error("Error opening file: " + filePath);
}

std::string getHomeDirectory() {
  const char* homeString = getenv("HOME");
  return std::string(homeString);
}
