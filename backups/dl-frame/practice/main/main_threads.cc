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
#include <vector>
#include <unistd.h>

#include "lib/threads/thread_util.h"

int func_thread() {
    std::cout << "****** using simple func:" << std::endl;
    for (uint8_t i = 0; i < 4; i++)
    {
        std::thread t(grace_t::lib::threads::output, i);
        t.detach(); 
    }
    usleep(1000); // 1000us

    std::cout << "****** using lambda func:" << std::endl;
    for (int i = 0; i < 4; i++)
    {
        std::thread t([i]{
                std::cout << "lambda func" << i + 9 << std::endl;
                });
        t.detach();
    }
    usleep(1000);

    std::cout << "****** using simple task class operator:" << std::endl;
    for (uint8_t i = 0; i < 4; i++)
    {
        // 如下是不行的。向std::thread的构造函数中传入的是一个临时变量，而不是命名变量就会出现语法解析错误。相当于声明了一个函数t，其返回类型为thread，而不是启动了一个新的线程。
        // std::thread t(grace_t::lib::threads::TaskSimple()); 

        // 如下理论上可以，因为用了新的初始化语法。。但试了会报attempt to use a deleted function
//        std::thread t{grace_t::lib::threads::TaskSimple2()};
        grace_t::lib::threads::TaskSimple task;
        std::thread t(task, i);
        t.detach(); 
    }


    return 0;
}


int main()
{
    func_thread();
    return 0;
}

