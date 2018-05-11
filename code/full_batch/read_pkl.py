import cPickle
import gzip
import csv

path = '../data/' + "pubmed" + '.pkl.gz'

f = gzip.open(path, 'rb')

# unpickler = cPickle.Unpickler(f)
# print(unpickler.load()[0])
entity_mul=[]
label=[]
word_list1=["urinari","insulin","kidney","genotyp","heart","glucose","insulindepend"]
word_list2=["obes","fat","genotyp","heart,glucose"]
word_list0=["genotyp","heart","glucose"]
with open("Pubmed-Diabetes.NODE.paper.tab") as tsv:
    for line in csv.reader(tsv, dialect="excel-tab"):
        for i in range(len(line)):
            for word in word_list1:
                count = 0
                count0=0
                if word in line[i]:
                    count += 1
            for word1 in word_list2:
                if word1 in line[i]:
                    count+=1
            if 'label=1' in line[i]:
                for word0 in word_list0:
                    if word0 in line:
                        count0+=1
                if count0 >=1:
                    label.append(1)
                else:
                    label.append(0)
            if 'label=2' in line[i]:
                label.append(1)
            if 'label=3' in line[i]:
                label.append(2)
        entity_mul.append(count)

entity_mul_new=entity_mul[2:]
for i in range(len(entity_mul_new)):
    if entity_mul_new[i]==0:
        entity_mul_new[i]=1
    if entity_mul_new[i]==1:
        entity_mul_new[i]=1.5
#print len(entity_mul_new),len(label)

# type 1: urinari,insulin, kidney,genotyp,heart,glucose,insulindepend
# type 2: obes, fat,genotyp,heart,glucose
# type 0: genotyp,heart,glucose: change label to 1 or 2

# feats, labels, rel_list, train_ids, valid_ids, test_ids = cPickle.load(f)
#
# print(labels[-1])