'''
Created on Oct 26, 2016

@author: sulea
'''
# Addition of two numbers
def add(a,b):
    return a+b

# Addition of 5 to a number
def addFixedValue(a):
    return a+5

# Call the methods and print the result
print add(1,2)
print addFixedValue(4) 

assert(add(1,2)==3)

i = 1
for i in range(1,10):
    if i <= 5:
        print "value is lower than 5 or equal\n"
    else:
        print "value is bigger than 5\n"
        
welcome = "Hello World"
print "the " + welcome +" string is " + str(len(welcome)) + " character long"

mylist = ["Nem", "En", "Vagyok", "Aki", "Valamit", "Akar", "Akar"]
print mylist
print mylist[0]
print mylist[-2]
print mylist[0:2]

for element in mylist:
    print element
    
print list(set(mylist))

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    def __str__(self):
        return "x value:"+str(self.x)+" y value:"+str(self.y)
    def __add__(self, other):
        p = Point()
        p.x = self.x+other.x
        p.y = self.y+other.y
        return p
        
p1 = Point(1,5)
p2 = Point(4,3)
print p1
print p1.x
print p1+p2
