# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:04:43 2018

@author: Jérémie
"""

from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from numpy.random import uniform
from matplotlib.patches import Circle

width = 1
height = 1

list_of_circles = []
list_without_radius = []

def generate_circles(n):
    while len(list_of_circles) < n: #as long as the list of circles is not full     
        x = uniform(0.05, high = width - 0.05)
        y = uniform(0.05, high = width - 0.05)
        r = uniform(0.05, high = 0.05)
        list_of_circles.append((x,y,r))
    
def take_radius_out(list):
    for item in list:
        list_without_radius.append((item[0],item[1]))
    return list_without_radius

generate_circles(10)
print("list:",list_of_circles)  
print("there are",len(list_of_circles)," elements in the list")   

list_without_radius = take_radius_out(list_of_circles)
print ("list without radius :",list_without_radius)

# =============================================================================
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.axis('equal')
# ax1.set_xlim((0,1))
# ax1.set_ylim((0,1))
# 
# for i in list_of_circles:
#     x = i[0]
#     y = i[1]
#     r = i[2]
#     circle = plt.Circle((x,y),r,fill=False)
#     ax1.add_artist(circle)
# =============================================================================

dist_matrix = pdist(list_without_radius,'euclidean')
print("dist matrix:",dist_matrix)

linkage_matrix = linkage(dist_matrix, method = 'single')
print('linkage matrix is:\n', linkage_matrix)

print("dendrogram:\n")
dendrogram = dendrogram(linkage_matrix)

plt.figure()
set_link_color_palette(['m', 'c', 'y', 'k'])
plt.show()