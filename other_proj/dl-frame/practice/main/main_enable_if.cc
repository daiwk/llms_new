#include <type_traits> 


#define B_F_M_FEF_DEPEND_TYPE(index) \
    typename ::std::enable_if< \
        ::std::is_const< \
        typename ::std::remove_pointer< \
        typename ::std::remove_reference<D##index>::type>::type>::value && \
        (::std::is_lvalue_reference<D##index>::value || ::std::is_pointer<D##index>::value), \
        D##index>::type




template<typename V, typename D0, typename D1>
class cls {

    typedef B_F_M_FEF_DEPEND_TYPE(1) depend_1_type;
};

int main()
{
    cls<float, float, const float&> xx;
    // cls<float, float, float&> yy; // failed because:
    // ERROR: xxxxx/grace_t/c++/workspace/main/BUILD:54:1: C++ compilation of rule '//main:main_enable_if' failed (Exit 1)
    //             main/main_enable_if.cc:18:13: error: no type named 'type' in 'std::__1::enable_if<false, float &>'; 'enable_if' cannot be used to disable this declaration
    //             typedef B_F_M_FEF_DEPEND_TYPE(1) depend_1_type;
    //         ^~~~~~~~~~~~~~~~~~~~~~~~
    //             main/main_enable_if.cc:6:9: note: expanded from macro 'B_F_M_FEF_DEPEND_TYPE'
    //             ::std::is_const< \
    //             ^~~~~~~~~~~~~~~~~~
    //             main/main_enable_if.cc:24:31: note: in instantiation of template class 'cls<float, float, float &>' requested here
    //             cls<float, float, float&> yy;
    //         ^
    //             1 error generated.

    return 0;
}
