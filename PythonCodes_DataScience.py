# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 18:18:37 2018

@author: Dmob
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import turtle
#import time
import random
import os
import dbm
import pickle
import copy
from collections import namedtuple
import re

import nltk
from nltk.book import wordnet as wn
from nltk.corpus import wordnet_ic

import gensim

'''
########## import csv
#os.chdir('C:\Users\me\Documents')

#wd=os.getcwd()
#a = os.chdir(r'C:\Users\Dmob\Desktop\ANDROID DEVELOPMENT\KAGGLE\kaggle models\churn ng data science')
#wd2=os.getcwd()
#print(os.getcwd())

#df=pd.read_csv(R"C:\Users\Dmob\Desktop\ANDROID DEVELOPMENT\KAGGLE\kaggle models\churn ng data science\TRAIN.csv", sep=',')
#print(df.values)

DIR=r'C:\Users\Dmob\Desktop\ANDROID DEVELOPMENT\KAGGLE\kaggle models\churn ng data science'

train_data = pd.read_csv(DIR+'/train.csv', delimiter=',')
#train_inputs = train_data.ix[:,0]
#train_labels = train_data.drop(0, axis=1)
test_data = pd.read_csv(DIR+'/test.csv', delimiter=',')


#reader = csv.reader(open(R"C:\Users\Dmob\Desktop\ANDROID DEVELOPMENT\KAGGLE\kaggle models\churn ng data science\TRAIN.csv", "rb"), delimiter=",")
#x = list(reader)

print(train_data.head(4))
'''

deer = wn.synset('deer.n.01')
elk = wn.synset('elk.n.01')
horse = wn.synset('horse.n.01')

##find path similarity
deer.path_similarity(elk)

print(deer.path_similarity(horse))

brown_ic=wordnet_ic.ic('ic-brown.dat')

a=deer.lin_similarity(elk,brown_ic)
z=deer.lin_similarity(horse,brown_ic)

print(a,z)
print(type((a,z)))
'''
text11 = "Children shouldn't drink a sugary drink before bed."
a=text11.split(' ')

b=nltk.word_tokenize(text11) #good way to split, takes care of fullstop & apostrophe

c = nltk.pos_tag(b)
print(a)
print(b)
#print(c)

#POS TAGS - part of speech 
#nltk.help.upenn_tagset('VB')


'''

'''
print(text7)
print(len(text7))
print(text8)
print(len(text8))

dist = FreqDist(text7)
#print(len(dist))


vocab1 = dist.keys()
#vocab1[:10] 
# In Python 3 dict.keys() returns an iterable view instead of a list
#print(list(vocab1)[:10])

freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100] #hw may tiems word ocurs and if word is at least lenght 5
#print(freqwords)

text1 = "Ethics are built building buildings riiigM2ht Ethos Etheos into @EAE @74k63 @oioisd843 the ideals and objectives of the United Nations http.qwqw https.qwasasa   "
text2 = text1.lower().split(' ')


porter = nltk.PorterStemmer()
print([porter.stem(t) for t in text2 if len(t)>2])
'''

'''
q = "800-123-123, 943-123-434,123-133-3447, 553-187-564, 312+123;123"
q1 = q.split(",")
text1 = "Ethics are built riiigM2ht Ethos Etheos into @EAE @74k63 @oioisd843 the ideals and objectives of the United Nations http.qwqw https.qwasasa   "
text2 = text1.split(' ') # Return a list of the words in text2, separating by ' '

#print(len(text1))
#print ('\t \ntext\ntext\ntext')
#print([w for w in text2 if re.search('^\d\d',w)])

#print([w for w in q1 if re.search('[89][40][02][-]\d\d\d[-]\d\d\d',w)])
print([w for w in q1 if re.search('\d{3}[-]\d{3}[-]\d*4',w)])

#print([w for w in text2 if re.search('Eth(i|o)',w)])
s= [w for w in text2 if re.search(r'(https?)\.(\w+)',w)]
print(s)

#print(s.group(1))

'''

'''

import pandas as pd

time_sentences = ["Monday: The doctor's appointment is at 2:45pm.", 
                  "Tuesday: The dentist's appointment is at 11:30 am.",
                  "Wednesday: At 7:00pm, there is a basketball game!",
                  "Thursday: Be back home by 11:15 pm at the latest.",
                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]

df = pd.DataFrame(time_sentences, columns=['text'])
df
print(df)

'''



'''
def printall(*args):
    print(args)


text1 = "Ethics are built right into the ideals and objectives of the United Nations    "
text2 = text1.split(' ') # Return a list of the words in text2, separating by ' '

print(len(text1))
print(text1)
'''


'''
Point = namedtuple('Point', ['x', 'y'])

p = Point(1, 2)

print(p[0])
'''

'''
class Time:
    """Represents the time of day.""" 
    #def print_time(time):
        #print('%.2d:%.2d:%.2d' % (time.hour, time.minute, time.second))
    def __str__(self):
        return '%.2d:%.2d:%.2d' % (self.hour, self.minute, self.second)
    
    def __add__(self,other):
        seconds = self.time_to_int() + other.time_to_int()
        return int_to_time(seconds)
    
    def __lt__(self,other):
         a = self.hour, self.minute, self.second 
         b = other.hour, other.minute, other.second
         #if a>b: return True
         #if self.hour>other.hour: return True
         return a>b
    
time2 = Time()
time2.hour = 11
time2.minute = 59
time2.second = 30   

time32 = Time()
time32.hour = 12
time32.minute = 3
time32.second = 3

print(time2.__lt__(time32))  
    #print_time(time2)

#Time.print_time(time2)
#time2.print_time()
###print(time2)


class Point:
    def __init__(self,x=0,y=0):
        self.x=x
        self.y=y
    def __str__(self):
        #return '(%d,%d)' %(self.x,self.y)
        return '{},{}'.format(self.x,self.y)
    def __add__(self, other):
        if isinstance(other, Point):
            return self.x + other.x, self.y + other.y
        elif isinstance (other, tuple):
            return self.x +other[0],self.y + other[1]
    
    def __radd__(self,other):
        return self.__add__(other)
        
k = Point(3,5)
e = Point(3,5)

print(k+(4,7))
print(k+e)

print(type(k+e))
print(type(k+(4,7)))

def print_attributes(obj):
    for attr in vars(obj):
        print(attr, getattr(obj, attr))



'''



'''
class Time:
    """Represents the time of day.
    attributes: hour, minute, second
    """
    
time2 = Time()
time2.hour = 11
time2.minute = 59
time2.second = 30

#####pure fn
def add_time(t1, t2):
    sum = Time()
    sum.hour = t1.hour + t2.hour
    sum.minute = t1.minute + t2.minute
    sum.second = t1.second + t2.second
    return sum

def print_time(x):
    print(str(x.hour)+':'+str(x.minute)+':'+str(x.second))
 
#####modifier
def increment(time, seconds):
    time.second += seconds
    if time.second >= 60:
        time.second -= 60
        time.minute += 1
    if time.minute >= 60:
        time.minute -= 60
        time.hour += 1
    #return time.minute, time.hour,time.second
    return print_time(time)

print(increment(time2,90))

'''

'''
y = Time()
y.hour = 11
y.minute = 54
y.second = 10

s = Time()
s.hour = 13
s.minute = 59
s.second = 30

def print_time(x):
    print(str(x.hour)+':'+str(x.minute)+':'+str(x.second))
    
#print_time(time)

def is_after(x,y):
    a= x.hour
    b=y.hour
    c= x.minute
    d=y.minute
    e= x.second
    f=y.second
    while a>b:
        while c>d:
            while e>f:
                print(True)
                #return print(True)
                
            else:
                return 'No'
                break
        break
    else:
        return 'No'
        
#print(is_after(s,y))

'''

'''
class Point:
    """Represents a point in 2-D space."""
     
blank = Point()
#print(blank)
blank.x = 3.0
blank.y = 4.0


class Rectangle:
    """Represents a rectangle.
    attributes: width, height, corner.
    """

box = Rectangle()
box.width = 100.0
box.height = 200.0
box.corner = Point()
box.corner.x = 0.0
box.corner.y = 0.0


def find_center(rect):
    p = Point()
    p.x = rect.corner.x + rect.width/2
    p.y = rect.corner.y + rect.height/2
    return p


p1 = Point()
p1.x=4
p1.y=5

p2 = copy.deepcopy(p1)


f=3
t=copy.copy(f)

#isinstance to check whether an object is an instance of a class
print (isinstance (box,Point))
print (isinstance (f,int))
print (hasattr (box,'width'))





#print('(%g, %g)' % (blank.x, blank.y))


def distance_between_points(a,b):
    return (a -b)
    
#print(distance_between_points(blank.x ,blank.y))
'''
'''
t = [1, 2, 3]
print(pickle.dumps(t))

t1 = [1,2,3]
s = pickle.dumps(t1)
t2 = pickle.loads(s)
print(t2)

print(t1==t2)
print(t1 is t2)

'''
'''
db = dbm.open('captions', 'c')

db['cleese.png'] = 'Photo of John Cleese doing a silly walk.'

for key in db:
    print(key, db[key])

#print(db['cleese.png'])

db.close()
'''

#cwd = os.getcwd()
#print(cwd)


#j=os.path.abspath('output.txt')
#print(j)
#print(os.path.exists('memo.txt'))
#print(os.path.isdir(r'C:\Users\Dmob\Desktop\ANDROID DEVELOPMENT\DS\repositories'))
#print(os.path.isfile('output.txt'))
#print(os.listdir(cwd))


'''
fout = open('output.txt', 'w')
line1 = "This here's the wattle,\n"
fout.write(line1)
fout.close()


m='I have spotted %d camels.' % (56.2)
e='I have spotted {} camels {}.'.format(56.24,9)

print(u)

u = 'In %d years I have spotted %g %s.' % (3, 0.1, 'camels')

#print(u)

'''


'''

def ope(d):
    open('{}.txt'.format(d),'w')
  
qw='rer'    
ope(qw)

####################
def ope(d):
    open(str(d)+'.txt','w')

#d='aa'
c= 'aqwe'
ope(123)


'''
'''
t = [1,2,3]
t.extend([1,2]*2)
print(t)


t = {'a': 0, 'c': 2, 'b': 1}
print(random.choice(t))

s = 'abcp'
t = [0, 1, 2,2]
p=[4,7,8,0]
print(zip(s, t,p))

for pair in zip(s, t,p):
    print(pair)
'''

'''
def printall(*args):
    print(args)


printall(1, 2.0, '3')


t = (7, 3)
#divmod(t)

divmod(*t)

'''
'''
addr = 'monty@python.org'
uname, domain = addr.split('@')

print (uname, domain)

quot, rem = divmod(7, 3)
print(quot,rem)


def min_max(t):
    #return min(t), max(t),type(t)
    p = min(t) 
    l= max(t)
    m =type(t)
    return p,l,m
     

w = (2,3,4,5,6)
e,j,k = min_max(w)

print(e,j,k)
#print(type(e,j,k))


a = ( 5,1,3)*2
t = ('a', 'b', 'c', 'd', 'e')
type (a)
#print (a)

x=(a+t)
#print (x)
#print (type (x))
#a[0]='q'
'''    

'''
count = [2]


def example3():
    #global count
    count.append(3)
    print(count)

example3()

'''
'''
If a global variable refers to a mutable value, you can modify the value without declaring
the variable:
known = {0:0, 1:1}
def example4():
known[2] = 1
So you can add, remove and replace elements of a global list or dictionary, but if you want
to reassign the variable, you have to declare it:
def example5():
global known
known = dict()
Global variables can be useful, but if you have a lot of them, and you modify them frequently,
they can make programs hard to debug.
'''




'''

def histogram(s):
    d = dict()
    for c in s:
        d[c]=1+ d.get(c,0) 
            
    return d


#h = histogram('bronkkkkkkkkkkkkkkkkktosaurus')
#print(h)

#print(h.get('s'))
def print_hist(h):
    for c in h:
        print(c, h[c])

h = histogram('parrot')
print_hist(h)

def reverse_lookup(d, v):
    for k in d:
        if d[k] == v:
            return k
    raise LookupError()

'''

'''
s=[1]*2
e=np.array([1])*2

t1 = ['a','z', 'b', 'c']
t= ['d', 'e']

x=4
t=t+x

print(t)
'''

'''
word = open('words.txt')
#print(fin.readline())
#line =fin.readline()

def is_abecedarian(word):
    if len(word) <= 1:
        return True
    if word[0] > word[1]:
        return False
    return is_abecedarian(word[1:]) #this intell moves the tracker
                                    #to the nxt word and checks it

#Another option is to use a while loop:
    
def is_abecedarian(word):
    i = 0
    while i < len(word)-1:
        if word[i+1] < word[i]:
            return False
        i = i+1
    return True

'''

'''
def is_abecedarian(word):
    index = 0
    while index < len(word) -1:
        if word[index] > word[index+1]:
            return False
        else:
            index +=1
    return True
    
fin = open('words.txt')
count = 0
for line in fin:
    word = line.strip()
    if is_abecedarian(word):
        count += 1
print('There are {} abecedarian words.'.format(count))   
#print(is_abecedarian('banana'))
#print(is_abecedarian('abcdefg'))

'''







'''
def is_abecedarian(word):
    index = 0
    while(index <= len(word)):
    
        for line in fin:
            word = line.strip()
            while index<1000:
               letter = word[index]
               letter2 = word[index+1]
               if letter < letter2:
                    print (letter)
                    print (letter2)
                
                    index=index+1
                    print(word)
            #else:
                #index=index+1
            #is_abecedarian()
            #return True
        #x=x+1     
      
print(is_abecedarian(word))
'''

'''
fin = open('words.txt')
#print(fin.readline())
#line =fin.readline()


for line in fin:
    word = line.strip() 
    if word > word[0:19]:
        print(word)
'''

'''
def has_no_e():
    counte=0
    x=0
    for line in fin:
        word = line.strip()
        x=x+1
        if 'e' not in word:
            counte=counte+1
            #print(word)
    print(counte)
    print(x)
    print(x/counte)
            
has_no_e()
'''


'''
def rotate_letter(letter, n):
    """Rotates a letter by n places.  Does not change other chars.

    letter: single-letter string
    n: int

    Returns: single-letter string
    """
    if letter.isupper():
        start = ord('A')
    elif letter.islower():
        start = ord('a')
    else:
        return letter

    c = ord(letter) - start
    i = (c + n) % 26 + start
    return chr(i)


def rotate_word(word, n):
    """Rotates a word by n places.

    word: string
    n: integer

    Returns: string
    """
    res = ''
    for letter in word:
        res += rotate_letter(letter, n)
    return res


if __name__ == '__main__':
    print(rotate_word('cheer', 7))
    print(rotate_word('melon', -10))
    print(rotate_word('sleep', 9))

'''
'''
def is_reverse(word1, word2):
    if len(word1) != len(word2):
        return False
    i = 0
    j = len(word2)-1
    while j >= 0:
        print(i,j)
        if word1[i] != word2[j]:
            return False
        i = i+1
        j = j-1
    return True

print(is_reverse('pots', 'stop'))
'''


'''
def find(word, letter,index):
    #index = 0
    while index < len(word):
        if word[index] == letter:
            return index
        index = index + 1
    return -1

print(find('mango','o',7))

'''

'''
index = 0
while index < len(fruit):
    letter = fruit[index]
    print(letter)
    index = index + 1
'''


'''
prefixes = 'JKLMNOPQ'
suffix = 'ack'


for letter in prefixes:
    index = 0
    if letter not in ('O', 'Q'):

        print(letter + suffix)
        index = index + 1
    else:
        print(letter + 'u'+suffix) 


for letter in prefixes:
        if letter in ('O', 'Q'):  # if the letter is O or Q
            print letter + 'u' + suffix
        else:
            print letter + suffix
'''



'''
fruit = 'banana'

index = len(fruit)-1
while index >=0:
    letter = fruit[index]
    print(letter)
    index = index - 1

'''
'''
def eval_loop():
    while True:
        n = raw_input('Input?\n:: ')
        if n == 'done':
            break
        else:
            result = eval(n)
            print (result)
    print (result)

eval_loop()
'''

'''

def eval_loop(x):
    line = input('> ')
    if line == 'done':
        result = 0
        result = eval(x)
    while True:
        print(result)
        break
        

print(eval_loop(4))
'''


'''
def is_power(a,b):
    if(a%b != 0):
        return False
    elif(a/b == 1):
        return True
    else:
        return is_power(a/b,b)

print(is_power(81,9))      
'''

'''
def is_power(a,b):
    i=1
    if a % b ==0 and b==a**(1/i):
        return True
    else:
        for i in range(5):
            z=i+1
            if a % b ==0 and b==a**(1/z):
                return True
            else:
                return False
 ''' 
        
'''
    elif a % b !=0 or b!=a**(1/i):
        for i in range (5):
            if a % b !=0 and b!=a**(1/i):
                i=i+1
                return True
            else:
                return False
    
print(is_power(81,9))
    #elif b = a**(1/(n+1))

'''


#print(np.roots(9))










'''
def A(m,n):
    if m==0:
        return n+1
    elif m>0 and n==0:
        return A(m-1,1)
    elif m>0 and n>0:
        return A(m-1,A(m,n-1))
    
print(A(3,4))

'''

'''
def factorial(n):
    if n == 0:
        return 1
    else:
        recurse = factorial(n-1)
        result = n * recurse
        return result
'''
'''
def is_between(x,y,z):
    if x<=y<=z:
        return True
    else:
        return False
    
print(is_between(1,2,3))    
'''
'''
def is_divisible(x, y):
    if x % y == 0:
        return True
    else:
        return False 
'''

'''
def distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    dsquared = dx**2 + dy**2
    result = math.sqrt(dsquared)
    return result


def circle_area(xc, yc, xp, yp):
    radius = distance(xc, yc, xp, yp)
    result = area(radius)
    return result
'''

'''
def recurse(n, s):
    if n == 0:
        print(s)
    else:
        recurse(n-1, n+s)
        recurse(3, 0)

recurse(-1,0)
'''
'''
def is_triangle(a,b,c):
    if a != b or a != c:
        print ('No')
    else:
        print ('Yes')
    

is_triangle(2,2,2)
'''
'''   
def check_fermat(a,b,c,n):
    if n > 2 and (a**n + b**n == c**n):
        print('Holy smokes, Fermat was wrong!')
    else:
        print("No, that doesn’t work.")   

def q():
    a1 = np.random.randint
    b1 = np.random.randint
    c1 = np.random.randint
    n1 = np.random.randint
    a=a1
    b=b1
    c=c1
    n=n1
    return check_fermat(a, b, c, n)

q()

'''
'''
x = np.random.randint
a=x
print (a)   
'''
#check_fermat(124,7,89,2)   
    

    




'''
for i in range (30):
    a= time.time()
    print(a)
    time.sleep(0.001)
'''

'''
s='hello'

def print_n(s, n):
    if n <= 0:
            return
    print(s)
    print_n(s, n-1)

print_n(s,3)

'''
'''
def countdown(n):
    if n <= 0:
        print('Almost thereeeeeeeeeeee')
        time.sleep(2)
        print('Blastoff!')
    else:
        print(n)
        time.sleep(0.2)
        countdown(n-1)

countdown(50)
'''




'''
def z():
    x = np.random.randint(-10,10)
    return x
    
    #reset_selective x
    

    #del x


def q():
    if x > 0:
        print('x is positive')
    elif x<0:
        print ('x is negative')
    else:
        print ('x must be zero')


for i in range (25):
    x = np.random.randint(-10,10)
    #z()
    q()
    
'''
'''
    for i in range(25):
        x = np.random.randint(-10,10)
    #z()
        q()
'''



'''
for i in a:
    q()
    i=i+1
'''





'''
minutes = 105
a = minutes // 60
r = minutes % 10
print(a)
print(r)
'''

'''
def a():
    print('+', '-'*4,end=' ')
    print('+', '-'*4,end=' ')
    print('+')



def b():
    print ('|','',' ',' ','|','',' ',' ','|')
    print ('|','',' ',' ','|','',' ',' ','|')
    print ('|','',' ',' ','|','',' ',' ','|')
    print ('|','',' ',' ','|','',' ',' ','|')


a()
b()
a()
b()
a()

def c():
    return a(),b(),a(),b(),a()
    


c()

'''










'''
#('monty')

def do_twice(f,p):
    f(p)
    f(p)


def print_spam():
    print('spam')



def print_twice(arg):
    """Prints the argument twice.

    arg: anything printable
    """
    print(arg)
    print(arg)

def do_four(m,z):
    do_twice(m,z)
    do_twice(m,z)

p=4
do_twice(print_twice,p)

do_four(print_twice,'spam')

'''



'''
def print_twice(bruce,x):
    a=3
    c=x*a
    print(bruce*2)
    print(c)

q=12
print_twice('Spam '*2,q)
    
print(math.sqrt(5))
'''




'''
repeat_lyrics()

def print_lyrics():
    
    print("I'm a lumberjack, and I'm okay.")
    print("I sleep all night and I work all day.")



def repeat_lyrics():
    print_lyrics()
    print_lyrics()

'''



'''
a= str(32)
a+a
g=3
q=g*math.pi
x = math.sin(q / 360.0 * 2 * math.pi)

print(x)
'''













'''
#d=s/t
x = 6.52
s1=8.15
s2 = 7.12
d1=1
d2=3
d3=1
t=d1/s1+d2/s2+d3/s1
ft = t+x
print(ft)
'''




















'''
r=5
pi=3.14
v=4*(pi*r**2)/r
print(v)
'''
'''
n=60
p= 24.95
b=p- 0.4*p
s=3+0.75*n-1
#n = ? #no of copies
w=n*b + s
print(w)
'''












###
'''
a=(1,2,3)*2
b=[1,3,4,5]*2
#c=np.array(1,2,3)
d=np.array([5,6,7,8,9])*2
#e=pd.Dataframe(4,9)

print(a)
print(b)
print(d)
'''

'''
# create a new figure
plt.figure()

# plot the point (1.5, 1.5) using the circle marker
plt.plot(1.5, 1.5, 'o')
# plot the point (2, 2) using the circle marker
plt.plot(2, 2, 'o')
# plot the point (2.5, 2.5) using the circle marker
plt.plot(2.5, 1.5, 'o')
# get the current axes
ax = plt.gca()

# Set axis properties [xmin, xmax, ymin, ymax]
ax.axis([0,6,0,10])

ax = plt.gca()
ax.get_children()'''

###