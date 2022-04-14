#pragma once

#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>
#include <iostream>

namespace paddle {
namespace benchmark {

inline int64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

template <typename T>
inline void get_value_from_sstream(std::stringstream *ss, T *value) {
  (*ss) >> (*value);
}

template <>
inline void get_value_from_sstream<std::string>(std::stringstream *ss,
                                         std::string *value) {
  *value = ss->str();
}

template <typename T>
std::vector<T> split_string(const std::string &str, char sep) {
  std::stringstream ss;
  std::vector<T> values;
  T value;
  values.clear();
  for (auto c : str) {
    if (c != sep) {
      ss << c;
    } else {
      get_value_from_sstream<T>(&ss, &value);
      values.push_back(std::move(value));
      ss.str({});
      ss.clear();
    }
  }
  if (!ss.str().empty()) {
    get_value_from_sstream<T>(&ss, &value);
    values.push_back(std::move(value));
    ss.str({});
    ss.clear();
  }
  return values;
}

template <typename T = std::string>
inline T parse_string(const std::string& v) {
  return v;
}

template <>
inline int64_t parse_string<int64_t>(const std::string& v) {
  return std::stoll(v);
}

template <>
inline int parse_string<int>(const std::string& v) {
  return std::stoi(v);
}

template <>
inline float parse_string<float>(const std::string& v) {
  return std::stof(v);
}

template <class T = std::string>
std::vector<T> Split(const std::string& original,
                            const std::string& separator) {
  std::vector<T> results;
  std::string::size_type pos1, pos2;
  pos2 = original.find(separator);
  pos1 = 0;
  while (std::string::npos != pos2) {
    if (pos1 != pos2) {
      results.push_back(parse_string<T>(original.substr(pos1, pos2 - pos1)));
    }
    pos1 = pos2 + separator.size();
    pos2 = original.find(separator, pos1);
  }
  if (pos1 != original.length()) {
    results.push_back(parse_string<T>(original.substr(pos1)));
  }
  return results;
}

bool read_file(const std::string &filename,
               std::vector<char> *contents,
               bool binary = true);

bool write_file(const std::string &filename,
                const std::vector<char> &contents,
                bool binary = true);

std::vector<std::string> ReadLines(const std::string& filename);

std::vector<float> ReadInputData(
    const std::string& input_data_dir,
    const int64_t input_size);

template <class T>
std::string Vec2Str(std::vector<T> vec) {
  std::stringstream ss;
  ss << "{" ;
  for (auto v : vec) {
    ss << v << ",";
  }
  ss << "}";
  return ss.str();
}
} // namespace benchmark
} // namespace paddle