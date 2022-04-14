#include "utils.hpp"

namespace paddle {
namespace benchmark {

bool read_file(const std::string &filename,
               std::vector<char> *contents,
               bool binary) {
  FILE *fp = fopen(filename.c_str(), binary ? "rb" : "r");
  if (!fp) return false;
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  contents->clear();
  contents->resize(size);
  size_t offset = 0;
  char *ptr = reinterpret_cast<char *>(&(contents->at(0)));
  while (offset < size) {
    size_t already_read = fread(ptr, 1, size - offset, fp);
    offset += already_read;
    ptr += already_read;
  }
  fclose(fp);
  return true;
}

bool write_file(const std::string &filename,
                const std::vector<char> &contents,
                bool binary) {
  FILE *fp = fopen(filename.c_str(), binary ? "wb" : "w");
  if (!fp) return false;
  size_t size = contents.size();
  size_t offset = 0;
  const char *ptr = reinterpret_cast<const char *>(&(contents.at(0)));
  while (offset < size) {
    size_t already_written = fwrite(ptr, 1, size - offset, fp);
    offset += already_written;
    ptr += already_written;
  }
  fclose(fp);
  return true;
}

std::vector<std::string> ReadLines(const std::string& filename) {
  std::ifstream ifile(filename.c_str());
  if (!ifile.is_open()) {
    std::cout << "Open file: [" << filename << "] failed." << std::endl;
  }
  std::vector<std::string> res;
  std::string tmp;
  while (getline(ifile, tmp)) res.push_back(tmp);
  ifile.close();
  return res;
}

std::vector<float> ReadInputData(
    const std::string& input_data_dir,
    const int64_t input_size) {
  std::vector<float> input_data(input_size);
  float* data = &(input_data.at(0));
  std::ifstream fin(input_data_dir, std::ios::in | std::ios::binary);
  fin.seekg(0, std::ios::end);
  int file_size = fin.tellg();
  fin.seekg(0, std::ios::beg);
  fin.read(reinterpret_cast<char*>(data), file_size);
  fin.close();
  return input_data;
}

} // namespace benchmark
} // namespace paddle