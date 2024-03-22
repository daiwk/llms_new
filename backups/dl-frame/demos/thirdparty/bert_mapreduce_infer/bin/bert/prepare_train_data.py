#encoding=utf8
"""
prepare_train_data
"""

import sys

for line in sys.stdin:
    line = line.strip("\n").split("\t")
    vals = line[1].split("\1")
    for v in vals:
        print v
    print ""
