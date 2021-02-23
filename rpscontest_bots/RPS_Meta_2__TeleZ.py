import random

RNA={'RR':'1','RP':'2','RS':'3','PR':'4','PP':'5','PS':'6','SR':'7','SP':'8','SS':'9'}
rot={'R':'P','P':'S','S':'R'}

if not input:
   DNA=[""]*3
   prin=[random.choice("RPS")]*19
   meta=[random.choice("RPS")]*7
   skor1=[[0]*19,[0]*19,[0]*19,[0]*19,[0]*19,[0]*19]
   skor2=[0]*7
else:
   for j in range(19):
       for i in range(4):
           skor1[i][j]*=0.8
       for i in range(4,6):
           skor1[i][j]*=0.5
       for i in range(0,6,2):
           skor1[i][j]-=(input==rot[rot[prin[j]]])
           skor1[i+1][j]-=(output==rot[rot[prin[j]]])
       for i in range(2,6,2):
           skor1[i][j]+=(input==prin[j])
           skor1[i+1][j]+=(output==prin[j])
       skor1[0][j]+=1.3*(input==prin[j])-0.3*(input==rot[prin[j]])
       skor1[1][j]+=1.3*(output==prin[j])-0.3*(output==rot[prin[j]])
   for i in range(7):
       skor2[i]=0.9*skor2[i]+(input==meta[i])-(input==rot[rot[meta[i]]])
   DNA[0]+=input
   DNA[1]+=output
   DNA[2]+=RNA[input+output]
   prin[18]=meta[6]=random.choice("RPS")
   for i in range(3):
       j=min(11,len(DNA[2]))
       k=-1
       while j>1 and k<0:
             j-=1
             k=DNA[i].rfind(DNA[i][-j:],0,-1)
       prin[2*i]=DNA[0][j+k]
       prin[2*i+1]=rot[DNA[1][j+k]]
   for i in range(6,18):
       prin[i]=rot[prin[i-6]]
   for i in range(0,6,2):
       meta[i]=prin[skor1[i].index(max(skor1[i]))]
       meta[i+1]=rot[prin[skor1[i+1].index(max(skor1[i+1]))]]
output=rot[meta[skor2.index(max(skor2))]]