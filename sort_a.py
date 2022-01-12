import json
import matplotlib.pyplot as plt
import numpy as np

list1 = []
def sort_a(text):#i biaoshi how many json? jbiaoshi how many if?
    list = []
    with open(text) as a:
        with open("sort_reslut.txt","a") as r:
            # for t in range(i):
            #     for k in range(j):
            #         list.append(a.readlines()[0][t]["ifs"][k]["if"])
            for i in a.readlines():
                if '"if":' in i:
                    list.append((float(i[14:-1])))
            s={}
            j=0
            for i in list:

                s[j]=i
                j+=1
            list1.append(sorted(s.items(),key=lambda x:x[1]))

            r.write(json.dumps(sorted(s.items(),key=lambda x:x[1]))+"\n")

sort_a("outdir2/did-38.TC.json")
sort_a("outdir2/did-38.TC.json")

list2=[[],[]]
for i in list1[0]:
    if len(list2[0])==10:
        break
    list2[0].append(i[0])
for j in range(10):
    t=0
    for i in list1[1]:

        if i[0]==list2[0][j]:

            list2[1].append(t)
            break
        t+=1
plt.figure(figsize=(20, 20),dpi=80)
list3=[0,1,2,3,4,5,6,7,8,9]
x=np.linspace(0,9,10)
y=np.linspace(0,19,20)
plt.xlabel("2 dim")
plt.ylabel("1 dim")
plt.ylim(0, 20)
plt.xlim(0, 10)
plt.xticks(x)
plt.yticks(y)
ax = plt.gca()
ax.spines['right'].set_color('none')#右坐标轴不显示
ax.spines['top'].set_color('none')#
ax.xaxis.set_ticks_position('bottom') #x坐标轴用底部轴代替
ax.yaxis.set_ticks_position('left')#y坐标轴用左部轴刻度代替
plt.plot(list3,list2[1],color='blue',linestyle='dashdot',linewidth=1,marker='o')
for a, b in zip(list3,list2[1]):
    plt.text(a,b,b,ha="center",va="bottom",fontsize=10)
plt.grid(color="#95a5a6",linestyle=":",linewidth=1,axis="y",alpha=0.4)
plt.show()
# with open("outdir1_1dim/did-38.TC.json") as a:
#     print(a.readlines()[1])