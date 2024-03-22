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


#include <iostream>

#include "lib/time/time.h"

int func_time() {

    std::chrono::time_point<std::chrono::high_resolution_clock> m_begin;
    m_begin = std::chrono::high_resolution_clock::now();
    std::cout << "elapsed seconds: " << grace_t::lib::time::elapsed_s(m_begin) << std::endl;
    m_begin = std::chrono::high_resolution_clock::now();
    std::cout << "elapsed mseconds: " << grace_t::lib::time::elapsed_ms(m_begin) << std::endl;

    return 0;
}

int main()
{
    func_time();
    return 0;
}

