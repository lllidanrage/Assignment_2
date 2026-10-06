#ifndef BPE_MD5_H_
#define BPE_MD5_H_
#include "bpe.h"
#include <string>
namespace bpe {

std::string md5_hex(const std::vector<Byte>& data);
}

#endif
