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
#include <exception>
#include <stdexcept>

#include "lib/exceptions/exceptions.h"

int func_exceptions_destruct() {

    std::exception_ptr eptr;
    // noexcept(false)可以抛出异常
    try {
        grace_t::lib::exceptions::funB();
    } catch(...) {
        std::cout << "caught funB" << std::endl;
        eptr = std::current_exception(); // capture
    }
    grace_t::lib::exceptions::handle_eptr(eptr);

    // 如果一个类的成员有noexcept(false)，它也可以抛异常
    try {
        grace_t::lib::exceptions::funC();
    } catch(...) {
        std::cout << "caught funC" << std::endl;
        eptr = std::current_exception(); // capture
    }
    grace_t::lib::exceptions::handle_eptr(eptr);

    // C++11中析构函数默认为noexcept(true), 所以如果我强制抛异常，是不行的，程序会直接terminate
    try {
        grace_t::lib::exceptions::funA();
    } catch(grace_t::lib::exceptions::MyException & e) {
        std::cout << "caught funA" << std::endl;
        eptr = std::current_exception(); // capture
    }
    grace_t::lib::exceptions::handle_eptr(eptr);

    // 不会输出这个，因为上个函数不该抛异常却抛异常，就terminate了
    std::cout << "enter here.." << std::endl;
    return 0;
}

int func_exceptions() {

    std::exception_ptr eptr;
    try {
        grace_t::lib::exceptions::Throw();
    } catch(...) {
        std::cout << "caught Throw" << std::endl;
        eptr = std::current_exception(); // capture
    }
    grace_t::lib::exceptions::handle_eptr(eptr);

    try {
        grace_t::lib::exceptions::NoBlockThrow();
    } catch(...) {
        std::cout << "Throw is no block" << std::endl;
        eptr = std::current_exception(); // capture
    }
    grace_t::lib::exceptions::handle_eptr(eptr);

//    try {
//        grace_t::lib::exceptions::BlockThrow();
//    } catch(...) {
//        std::cout << "caught Throw.." << std::endl;
//        eptr = std::current_exception(); // capture
//    }
//    grace_t::lib::exceptions::handle_eptr(eptr);
//    
//    // 不会输出这个，因为上个函数不该抛异常却抛异常，就terminate了
//    std::cout << "enter here.." << std::endl;

    return 0;
}


int main()
{
    func_exceptions();
    func_exceptions_destruct();
    return 0;
}

