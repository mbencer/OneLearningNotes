#include <circle_schema_generated.h>

#include <vector>

#include <fcntl.h>    // O_RDONLY
#include <sys/mman.h> // mmap, munmap
#include <sys/stat.h> // fstat
#include <unistd.h>   // close

#include <iostream>
#include <string>
#include <stdexcept>
#include <map>

namespace {

struct Tensor {
  std::string name;
  std::vector<float> weights;
  uint32_t op_consumer_idx;
};

bool compare_vec(const std::vector<float>& vec1, const std::vector<float>& vec2, float epsilon = 0.00000001) {
    if (vec1.size()!= vec2.size()) {
        return false;
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::abs(vec1[i] - vec2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

  void compare_weights(const std::vector<Tensor>& before, const std::vector<Tensor>& after) {
    for(int i=0;i<before.size();i++) {
      assert(before[i].op_consumer_idx == after[i].op_consumer_idx);
      assert(before[i].name == after[i].name);
      if(compare_vec(before[i].weights, after[i].weights)) {
        std::cout << before[i].op_consumer_idx << " " << "unchanged" << std::endl;
      } else {
        std::cout << before[i].op_consumer_idx << " " << "changed" << std::endl;
      }
    }
  }
  class MMappedFile
  {
  public:
    MMappedFile(const char *filename) { _fd = open(filename, O_RDWR); }

    bool ensure_mmap()
    {
      struct stat file_stat;
      if (fstat(_fd, &file_stat) != 0 || file_stat.st_size < 0 ||
          static_cast<uint64_t>(file_stat.st_size) > SIZE_MAX)
        return false;

      _buf_sz = static_cast<size_t>(file_stat.st_size);
      _buf = mmap(NULL, _buf_sz, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
      return _buf != MAP_FAILED;
    }

    bool sync() { return msync(_buf, _buf_sz, MS_SYNC) == 0; }

    bool close()
    {
      bool ret = false;
      if (_buf != MAP_FAILED)
      {
        ret = munmap(_buf, _buf_sz) == 0;
        _buf = MAP_FAILED; // mark as cleaned up
      }
      if (_fd != -1)
      {
        ::close(_fd);
        _fd = -1; // mark as cleaned up
      }
      return ret;
    }

    ~MMappedFile() { close(); }

    uint8_t *buf() const { return static_cast<uint8_t *>(_buf); }
    size_t buf_size() const { return _buf_sz; }

  private:
    int _fd;
    void *_buf = MAP_FAILED;
    size_t _buf_sz = 0;
  };

}

std::vector<Tensor> extract_weights(std::string path) {
  std::vector<Tensor> tensors;
  MMappedFile mmapfile{path.c_str()};
  if (!mmapfile.ensure_mmap())
      throw std::runtime_error("mmap failed");
  auto model = ::circle::GetModel(mmapfile.buf());
  if (!model)
      throw std::runtime_error("reading model failed");
  auto subgs = model->subgraphs();
  if (!subgs || subgs->size() != 1)
      throw std::runtime_error("many subgraphs not supported");
  const auto subg = subgs->Get(0);
  for(uint32_t i=0;i<subg->operators()->size();++i) {
    std::cout << "size: " << subg->operators()->size() << "\n";
    const auto op = subg->operators()->Get(i);
    for(uint32_t j=0;j<op->inputs()->size();++j) {
      auto idx = op->inputs()->Get(j);
      if(-1 == idx) {
        continue;
      }
      auto buf_idx = subg->tensors()->Get(idx)->buffer();
      auto tensor_name = subg->tensors()->Get(buf_idx)->name()->c_str();
      tensors.push_back(Tensor{tensor_name, {}, i});

      const ::circle::Buffer *buffer = (*model->buffers())[buf_idx];
      const flatbuffers::Vector<uint8_t> *tensor_array = buffer->data();
      if (!tensor_array) {
          continue;
      }
      const auto float_array = reinterpret_cast<const float*>(tensor_array->data());
      const auto float_array_size = tensor_array->size()/sizeof(float);
      for(uint32_t k=0;k<float_array_size;k++) {
          tensors.back().weights.push_back(float_array[k]);
      }
    }
  }

  if (mmapfile.sync() == false)
    throw std::runtime_error("mmap sync failed");

  if (mmapfile.close() == false)
    throw std::runtime_error("mmap closing failed");

  return tensors;
}

int main(int argc, char *argv[]) {
  assert(argc == 3);
  auto weight_before = extract_weights(argv[1]);
  auto weight_after = extract_weights(argv[2]);
  compare_weights(weight_before, weight_after);
  return 0;
}
