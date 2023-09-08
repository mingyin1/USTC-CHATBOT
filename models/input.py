
n = int(input())
la = []
lb = []
lc = []
for i in range(n):
    a,b,c = input().split()
    la.append(a)
    lb.append(b)
    lc.append(c)
s = input().strip()
x,y,z = (0,0,0)
for i in range(n):
    if s == la[i]:
        x += 1
    if s == lb[i]:
        y += 1
    if s == lc[i]:
        z += 1
print(x,y,z)