#ifndef DEMO_SIMPLE_DICT_H
#define DEMO_SIMPLE_DICT_H
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <stdlib.h>
namespace baidu {
    class SimpleDict {
        public:
            SimpleDict();
            ~SimpleDict();
            void init();
            int load_dict(std::string vec_file);
            int search_dict(std::string key);
            std::string search_invert_dict(unsigned int index);
        private:
            std::unordered_map<std::string, int > _dict;
            std::vector<std::string> _invert_vec;
    };
}
#endif
