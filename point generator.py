# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:21:47 2018

@author: Jérémie
"""

from numpy.random import uniform
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

width = 1
height = 1

list_of_circles = []

def generate_circles(n):
    while len(list_of_circles) < n: #as long as the list of circles is not full     
        x = uniform(0.05, high = width - 0.05)
        y = uniform(0.05, high = width - 0.05)
        r = uniform(0.05, high = 0.05)
        list_of_circles.append((x,y,r))
    
def take_radius_out(list):
    list_without_radius = []
    for item in list:
        list_without_radius.append((item[0],item[1]))
    return list_without_radius

generate_circles(10)
print("list:",list_of_circles)  
print("there are",len(list_of_circles)," elements in the list")   
print ("list without radius :",take_radius_out(list_of_circles))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.axis('equal')
ax1.set_xlim((0,1))
ax1.set_ylim((0,1))

for i in list_of_circles:
    x = i[0]
    y = i[1]
    r = i[2]
    circle = plt.Circle((x,y),r,fill=False)
    ax1.add_artist(circle)

#fig.savefig('plotcircles.png')   
            
        
        