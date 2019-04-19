
import re
d = dict()
c = re.compile("^(.*)class\s+([a-zA-Z0-9]+)")
with open("list1.txt","r") as f:
    ls = f.readlines()
    for i in ls:

        m = c.match(i)
        if m:
             g = m.groups()
             d[g[1]] = g[0].strip()

for k in d:
    if not k.endswith("Test"):
        t = k + "Test"
        print k + "("+d[k]+")"+ "," + str(t in d)