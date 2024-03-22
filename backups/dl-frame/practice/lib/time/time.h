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

#ifndef GRACE_T_LIB_TIME_TIME_H
#define GRACE_T_LIB_TIME_TIME_H

#include <chrono>
#include <iostream>

namespace grace_t {
namespace lib {
namespace time {

double elapsed_s(std::chrono::time_point<std::chrono::high_resolution_clock> m_begin)
{
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - m_begin).count();
}

int64_t elapsed_ms(std::chrono::time_point<std::chrono::high_resolution_clock> m_begin)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - m_begin).count();
}

}
}
}

#endif
