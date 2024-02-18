from ctypes import cdll
from ctypes import c_ulonglong
import ctypes

mydll = cdll.LoadLibrary('./x.so') 
s = "iiiabb".encode("utf8")
a = ctypes.c_int(2)
b = ctypes.c_int(2)

mydll.MurmurHash64B.restype = c_ulonglong
print mydll.MurmurHash64B(s, a, b)
###
####s = "iiiabb"#.encode("utf8")
###
###x = ctypes.c_char_p(s)
###x1 = ctypes.cast(x, ctypes.c_void_p).value
###x2 = ctypes.cast(x, ctypes.c_void_p)
###
####print mydll.MurmurHash64B(ctypes.addressof(x.contents), a,b)
####print mydll.MurmurHash64B(x, a,b)
###
####mydll.MurmurHash64B.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint]
####mydll.MurmurHash64B.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
###mydll.MurmurHash64B.restype = c_ulonglong
####print x
####print x1
####print x2
###print mydll.MurmurHash64B(s, a,b)
