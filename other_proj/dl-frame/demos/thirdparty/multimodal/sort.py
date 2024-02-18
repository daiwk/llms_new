import sys
from operator import itemgetter, attrgetter

lst = []

for line in sys.stdin:
    xx = line.strip("\n").split("\t")
    a = [xx[0], int(xx[1]), int(xx[2])] + xx[3:]
    lst.append(a)

#xlst = sorted(lst, key=lambda x: itemgetter(0, 1, 2))
xlst = sorted(lst, key=lambda x: (x[0], x[1], x[2]))
for x in xlst:
    print "\t".join(str(q) for q in x)
