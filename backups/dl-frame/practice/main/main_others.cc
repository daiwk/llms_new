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

#include <cassert>
#include <iostream>

class Base {
public:
    inline virtual void who () {
        std::cout << "I am base\n";
    }
    virtual ~Base () {} 
    int val;
};

class Derived: public Base {
public:
    inline void who () {
        std::cout << "I am Derived\n";
    }
};

int main() {
    Base b;
    b.who();

    Base* ptr = new Derived();
    ptr->who();
    int a = 3;
    int * ptr_int = &a;
    int a_array[] = {1, 2, 3 , 5, 4};
    std::cout << sizeof(b) << std::endl;
    std::cout << sizeof(ptr) << std::endl;
    std::cout << sizeof(ptr_int) << std::endl;
    std::cout << sizeof(a_array) << std::endl;
    std::cout << sizeof(a) << std::endl;
    assert(sizeof(a) == sizeof(a_array) / 5);
    std::cout << (sizeof(a) == sizeof(a_array) / 5) << std::endl;

    delete ptr;
    ptr = nullptr;
    return 0;
}
