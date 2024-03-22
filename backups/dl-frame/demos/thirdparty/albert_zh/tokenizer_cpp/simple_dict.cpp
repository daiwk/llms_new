#include "simple_dict.h"
#include <iostream>

namespace baidu {

    SimpleDict::SimpleDict() {
        this->init();
    }

    SimpleDict::~SimpleDict() {
    }

    void SimpleDict::init() {
        _dict.clear();
        _invert_vec.clear();
    }

    int SimpleDict::load_dict(std::string dict_file) {
        std::ifstream infile(dict_file.c_str());
        if (!infile) {
            std::cout << "SimpleDict::load_dict() failure " << dict_file << " not exist err!" << std::endl;
            return -1;
        }

        std::string token = "\t";
        std::string line;
        int count = 0;
        while (std::getline(infile, line)) {
            this->_dict[line] = count;
            _invert_vec.push_back(line);
            std::cout << "word:" << line << " id:" << count << std::endl;
            count++;
        }
        infile.close();

        return 0;
    }

    int SimpleDict::search_dict(std::string key) {
        if (this->_dict.find(key) != this->_dict.end()) {
            return _dict[key];
        }
        else {
            return -1;
        }
    }

    std::string SimpleDict::search_invert_dict(unsigned int idx) {
        if (idx + 1 < _invert_vec.size()) {
            return _invert_vec[idx];
        }
        else {
            return "";
        }

    }

}
