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

// refers to https://blog.csdn.net/qq_33266987/article/details/78784286

#ifndef GRACE_T_LIB_PTRS_UNIQUE_PTR_UTIL_H
#define GRACE_T_LIB_PTRS_UNIQUE_PTR_UTIL_H

#include <memory>
#include <iostream>

namespace grace_t {
namespace lib {
namespace ptrs {

//从函数返回一个unique_ptr
std::unique_ptr<int> func1(int a);
 
//返回一个局部对象的拷贝
std::unique_ptr<int> func2(int a);

void func3(std::unique_ptr<int> &up);

std::unique_ptr<int> func4(std::unique_ptr<int> up);

}
}
}

#endif
