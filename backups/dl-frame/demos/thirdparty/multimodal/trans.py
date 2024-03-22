import sys

for line in sys.stdin:
    line = line.strip("\n").replace("\t", "")
    xx = line.split(" ")
    nid = xx[0]
    kk = " ".join(xx[1:])
    show, clk = kk.split("} ")[-1].split(" ")
    #info = "show:%s;clk%s" % (show, clk)
    info = show
    print "\t".join([nid, info])
