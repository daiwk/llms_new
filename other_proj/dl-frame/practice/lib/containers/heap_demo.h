/*Copyright 2018 The grace_t Authors. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef GRACE_T_LIB_CONTAINERS_HEAP_DEMO_H
#define GRACE_T_LIB_CONTAINERS_HEAP_DEMO_H

#include <vector>
#include <string>
#include <stdint.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <map>

namespace grace_t {
namespace lib {
namespace containers {

class Result {
public:
    Result() : _index(0), _score(0.0) {}
    Result(uint32_t index, float score) : _index(index), _score(score) {}

    bool operator<(const Result& other) const {
        return _score < other._score;
    }

    bool operator>(const Result& other) const {
        return _score > other._score;
    }

    uint32_t _index;
    float _score;
};

class FixedMinHeap {
public:
    FixedMinHeap() : _in_heap(false) {
    }

    int add(const Result& result, int k) {
        if (_vec.size() < k) {
            _vec.emplace_back(result);
        } else {
            if (!_in_heap) {
                ::std::make_heap(_vec.begin(), _vec.end(), ::std::greater<Result>());
                _in_heap = true;
            }

            const Result& top = _vec[0];
            if (result._score > top._score) {
                _vec.emplace_back(result);
                std::pop_heap(_vec.begin(), _vec.end(), std::greater<Result>());
                _vec.pop_back();
            }
        }
        return 0;
    }

    // after sort, 0...size-1 are from large to small
    int sort(int k) {
        if (k >= _vec.size()) {
            std::sort(_vec.begin(), _vec.end(), std::greater<Result>());
        }
        else {
            std::sort_heap(_vec.begin(), _vec.end(), std::greater<Result>());
        }
        return 0;
    }

    int vec_size() {
        return _vec.size();
    }

    Result& get_result(int i) {
        return _vec[i];
    }

    bool _in_heap;
    std::vector<Result> _vec;
};

}
}
}

#endif
