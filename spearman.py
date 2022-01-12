import math
import sys
import string
from itertools import combinations


# sys.srgs="../src/outdir/did-0.TC.json"
# if len(sys.argv) > 1:
#     #data file given as arg
#     filename = sys.argv[1]
# else:
#     sys.exit("Usage: python " + sys.argv[0] + " [matrix filename]")
x = [0, 50, 150, 200, 250, 300, 350, 400, 450, 500]
y = [0, 10, 28, 42, 59, 51, 73, 85, 104, 96]
#x, y must be one-dimensional arrays of the same length
#Pearson algorithm
def pearson(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: len(n) * sum(map(lambda i: i ** 2, n)) - (sum(n) ** 2)
    return (len(x) * sum(map(lambda a: a[0] * a[1], zip(x, y))) - sum(x) * sum(y)) / math.sqrt(q(x) * q(y))

#Spearman algorithm
def spearman(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: map(lambda val: sorted(n).index(val) + 1, n)
    d = sum(map(lambda x, y: (x - y) ** 2, q(x), q(y)))
    return 1.0 - 6.0 * d / float(len(x) * (len(y) ** 2 - 1.0))

#Kendall algorithm
def kendall(x, y):
    assert len(x) == len(y) > 0
    c = 0 #concordant count
    d = 0 #discordant count
    t = 0 #tied count
    for (i, j) in combinations(range(len(x)), 2):
        s = (x[i] - x[j]) * (y[i] - y[j])
        if s:
            c += 1
            d += 1
            if s > 0:
                t += 1
            elif s < 0:
                t -= 1
        else:
            if x[i] - x[j]:
                c += 1
            elif y[i] - y[j]:
                d += 1
    return t / math.sqrt(c * d)

#read in file
# with open(filename) as f:
#     map(split_values, f.readlines())

print('Pearson Rho: %f' % pearson(x, y))

print('Spearman Rho: %f' % spearman(x, y))

print('Kendall Tau: %f' % kendall(x, y))


# spearman-rank.py

# set critical values
CRIT_VALUES = { 5:  1.000,
				6:  0.886,
				7:  0.786,
				8:  0.738,
				9:  0.700,
				10: 0.648,
				11: 0.618,
				12: 0.587,
				13: 0.560,
				14: 0.538,
				15: 0.521,
				16: 0.503,
				17: 0.488,
				18: 0.472,
				19: 0.460,
				20: 0.447,
				21: 0.436,
				22: 0.425,
				23: 0.416,
				24: 0.407,
				25: 0.398,
				26: 0.390,
				27: 0.383,
				28: 0.375,
				29: 0.368,
				30: 0.362 }

# the data sets to be ranked
set_1 = [0, 50, 150, 200, 250, 300, 350, 400, 450, 500]
set_2 = [0, 10, 28, 42, 59, 51, 73, 85, 104, 96]

# order the sets
set_1_ord = sorted(set_1)
set_2_ord = sorted(set_2)

# append relevant rank to each value in set
set_1_ranked = []
set_2_ranked = []
#写出在原来的数据每个数据在排序后的序列
for i in range(len(set_1)):
    set_1_ranked.append([set_1[i], set_1_ord.index(set_1[i])+1])

for i in range(len(set_2)):
    set_2_ranked.append([set_2[i], set_2_ord.index(set_2[i])+1])

print(set_1_ranked)
print(set_2_ranked)

# calculate d
d = []
for i in range(len(set_1_ranked)):
    d.append(set_1_ranked[i][1] - set_2_ranked[i][1])
print(d)

# calculate d^2
d_sq = [i**2 for i in d]
print(d_sq)

# sum d^2
sum_d_sq = sum(d_sq)
print(sum_d_sq)

# calculate n^3 - n
n_cu_min_n = len(set_1)**3 - len(set_1)
print(n_cu_min_n)

# calculate r
r = 1 - ((6.0*sum_d_sq)/n_cu_min_n)
print(r)

critical = CRIT_VALUES[len(set_1)]

# compare r to relevant critical value
if r > critical:
    print('significant correlation')


