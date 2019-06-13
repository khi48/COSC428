import sys
import select as something

while 1:
    rlist,_,_=select([sys.stdin],[],[],0)
    content=""
    while rlist:
        content+=raw_input()
        rlist,_,_=select([sys.stdin],[],[],0)
    print "blocking task - content:"+content
    time.sleep(5)
