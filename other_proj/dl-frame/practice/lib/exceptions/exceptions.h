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

// refers to https://blog.csdn.net/yuxing55555/article/details/80855274 

#ifndef GRACE_T_LIB_EXCEPTIONS_EXCEPTIONS_H
#define GRACE_T_LIB_EXCEPTIONS_EXCEPTIONS_H

#include <iostream>
#include <exception>
#include <stdexcept>

namespace grace_t {
namespace lib {
namespace exceptions {

struct MyException : public std::exception {
    const char* what () const throw ()
    {
        return "my C++ Exception";
    }
};

void handle_eptr(std::exception_ptr eptr);

void Throw();

void NoBlockThrow();

//表示函数不会抛出异常，如果抛出，会回调用std::terminate中断程序执行
//noexcept为修饰符
void BlockThrow() noexcept;

struct A {
    ~A() {
        throw MyException();
    }
};
struct B {
    // noexcept(false)表示可以抛出异常
    ~B() noexcept(false) {
        throw MyException();
    }
};
struct C {
    B b;
};

int funA();
int funB();
int funC();

}
}
}

#endif
