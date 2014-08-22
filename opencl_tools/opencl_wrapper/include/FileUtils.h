#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <string>

std::string readFile(const std::string& filePath);

inline std::string readWholeFileStream(std::ifstream& fileStream);

inline void verifyFileStreamOpen(const std::string& filePath,
                                 const std::ifstream& fileStream);

std::string getHomeDirectory();

#endif
