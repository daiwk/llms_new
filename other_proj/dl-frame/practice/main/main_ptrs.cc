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

#include "lib/ptrs/unique_ptr_util.h"
#include "lib/ptrs/shared_ptr_util.h"


int func_unique_ptr() {
    //// unique_ptr
    std::unique_ptr<int> up0(new int());
    //std::unique_ptr<int> up2 = new int();   //error! 构造函数是explicit
    //std::unique_ptr<int> up3(up1); ////error! 不允许拷贝

    std::unique_ptr<int> up(new int(10));
    // 传引用，不拷贝，不涉及所有权的转移
    grace_t::lib::ptrs::func3(up);
    // 暂时转移所有权，函数结束时返回拷贝，重新收回所有权
    // 如果不用up重新接受func4的返回值，这块内存就泄漏了
    up = grace_t::lib::ptrs::func4(std::unique_ptr<int> (up.release()));
    // up放弃对它所指对象的控制权，并返回保存的指针，将up置为空，不会释放内存
    up.release();

    //释放up指向的对象，将up置为空== up.reset();
    up = nullptr;

    int *x(new int());
    std::unique_ptr<int> up1,up2;
    // up.reset(…) 参数可以为 空、内置指针，先将up所指对象释放，然后重置up的值.
    up1.reset(x);
    // 不能再下面这么做了，因为会报：pointer being freed was not allocated
    // 因为这样会使up1 up2指向同一个内存，但unique_ptr不允许两个独占指针指向同一个对象
    //up2.reset(x);

    std::unique_ptr<int> sp(new int(88));
    std::vector<std::unique_ptr<int> > vec;
    vec.push_back(std::move(sp)); // 可以通过std::move作为容器的元素。std::move让调用者明确知道拷贝构造、赋值后会导致之前的unique_ptr失效。
    // vec.push_back(sp); // 直接编译失败
    // std::cout << *sp << std::endl; // 会core，因为已经std::move
    return 0;
}

int func_shared_ptr() {

    std::shared_ptr<int> spx1 = std::make_shared<int>(10);
    std::cout << "spx1: " << *spx1 << std::endl;
    std::shared_ptr<std::string> spx2 = std::make_shared<std::string>("Hello c++");
    std::cout << "spx2: " << *spx2 << std::endl;

    auto spx3 = std::make_shared<int>(11);
    std::cout << "spx3: " << *spx3 << std::endl;
    auto spx4 = std::make_shared<std::string>("C++11");
    std::cout << "spx4: " << *spx4 << std::endl;

    std::cout << "******************* test swap:" << std::endl; 
    std::shared_ptr<int> sp0(new int(2));
    std::shared_ptr<int> sp1(new int(11));
    std::shared_ptr<int> sp2 = sp1;
    std::cout << "before sp0 sp1 swap:" << std::endl;
    std::cout << "sp0: " << *sp0 << std::endl;
    std::cout << "sp1: " << *sp1 << std::endl;
    std::cout << "sp2: " << *sp2 << std::endl;
    std::cout << "sp0 use_count: " << sp0.use_count() << std::endl;
    std::cout << "sp1 use_count: " << sp1.use_count() << std::endl;
    std::cout << "sp2 use_count: " << sp2.use_count() << std::endl;
    std::cout << "address sp0: " << sp0 << std::endl;
    std::cout << "address sp1: " << sp1 << std::endl;
    std::cout << "address sp2: " << sp2 << std::endl;
    sp1.swap(sp0);
    std::cout << "after sp0 sp1 swap:" << std::endl; 
    std::cout << "sp0: " << *sp0 << std::endl;
    std::cout << "sp1: " << *sp1 << std::endl;
    std::cout << "sp2: " << *sp2 << std::endl; // sp0和1互换，不影响2。
    std::cout << "sp0 use_count: " << sp0.use_count() << std::endl;
    std::cout << "sp1 use_count: " << sp1.use_count() << std::endl;
    std::cout << "sp2 use_count: " << sp2.use_count() << std::endl;
    std::cout << "address sp0: " << sp0 << std::endl;
    std::cout << "address sp1: " << sp1 << std::endl;
    std::cout << "address sp2: " << sp2 << std::endl;

    std::cout << "******************* test reset nullptr:" << std::endl; 
    std::shared_ptr<int> sp3(new int(22));
    std::shared_ptr<int> sp4 = sp3;
    std::cout << "before sp3 reset:" << std::endl; 
    std::cout << "sp3: " << *sp3 << std::endl;
    std::cout << "sp4: " << *sp4 << std::endl;
    std::cout << "sp3 use_count: " << sp3.use_count() << std::endl;
    std::cout << "sp4 use_count: " << sp4.use_count() << std::endl;
    std::cout << "address sp3: " << sp3 << std::endl;
    std::cout << "address sp4: " << sp4 << std::endl;
    sp3.reset();
    std::cout << "after sp3 reset:" << std::endl; 
    //std::cout << "sp3: " << *sp3 << std::endl; // sp3已经reset了，是空指针，取*会core
    std::cout << "sp3 is nullptr: " << (sp3 == nullptr) << std::endl; // 记得加()
    std::cout << "sp4: " << *sp4 << std::endl;
    std::cout << "sp3 use_count: " << sp3.use_count() << std::endl;
    std::cout << "sp4 use_count: " << sp4.use_count() << std::endl;
    std::cout << "address sp3: " << sp3 << std::endl;
    std::cout << "address sp4: " << sp4 << std::endl;

    std::cout << "******************* test reset new int:" << std::endl; 
    std::shared_ptr<int> sp5(new int(22));
    std::shared_ptr<int> sp6 = sp5;
    std::shared_ptr<int> sp7 = sp5;
    std::cout << "before sp5 reset new int:" << std::endl; 
    std::cout << "sp5: " << *sp5 << std::endl;
    std::cout << "sp6: " << *sp6 << std::endl;
    std::cout << "sp7: " << *sp7 << std::endl;
    std::cout << "sp5 use_count: " << sp5.use_count() << std::endl;
    std::cout << "sp6 use_count: " << sp6.use_count() << std::endl;
    std::cout << "sp7 use_count: " << sp7.use_count() << std::endl;
    std::cout << "address sp5: " << sp5 << std::endl;
    std::cout << "address sp6: " << sp6 << std::endl;
    std::cout << "address sp7: " << sp7 << std::endl;
    sp5.reset(new int(33));
    std::cout << "after sp5 reset new int:" << std::endl; 
    std::cout << "sp5: " << *sp5 << std::endl;
    std::cout << "sp6: " << *sp6 << std::endl;
    std::cout << "sp7: " << *sp7 << std::endl;
    std::cout << "sp5 use_count: " << sp5.use_count() << std::endl;
    std::cout << "sp6 use_count: " << sp6.use_count() << std::endl;
    std::cout << "sp7 use_count: " << sp7.use_count() << std::endl;
    std::cout << "address sp5: " << sp5 << std::endl;
    std::cout << "address sp6: " << sp6 << std::endl;
    std::cout << "address sp7: " << sp7 << std::endl;

    return 0;
}


int main()
{
    func_unique_ptr();
    func_shared_ptr();
    return 0;
}

