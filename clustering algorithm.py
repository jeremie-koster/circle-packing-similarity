# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:04:43 2018

@author: Jérémie
"""

from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from numpy.random import uniform, choice
from matplotlib.patches import Circle
from math import sqrt,atan,degrees,cos,sin,atan2,pi


width = 1
height = 1

list_of_circles = []
list_without_radius = []
list_of_clusters = []

def generate_circles(n):
    while len(list_of_circles) < n: #as long as the list of circles is not full     
        x = uniform(0.05, high = width - 0.05)
        y = uniform(0.05, high = width - 0.05)
        r = uniform(0.02, high = 0.05)
        list_of_circles.append((x,y,r))
    
def take_radius_out(list):
    for item in list:
        list_without_radius.append((item[0],item[1]))
    return list_without_radius

generate_circles(10)
print("\nlist:",list_of_circles)  
print("\nthere are",len(list_of_circles)," elements in the list")   

list_without_radius = take_radius_out(list_of_circles)
print ("\nlist without radius :",list_without_radius)

fig1 = plt.figure()
ax1 = fig1.add_subplot(221)
ax1.set_aspect('equal')
ax1.set_xlim((0,1.5))
ax1.set_ylim((0,1.5))

for i in list_of_circles:
    x = i[0]
    y = i[1]
    r = i[2]
    circle = plt.Circle((x,y),r,fill=False)
    ax1.add_artist(circle)

dist_matrix = pdist(list_without_radius,'euclidean')
print("\ndist matrix:",dist_matrix)

linkage_matrix = linkage(dist_matrix, method = 'single')
print('\nlinkage matrix is:\n', linkage_matrix)

# keep only the clusters' id and convert them into int for following processing
clean_linkage_matrix = []
for i in linkage_matrix:
    cluster_one = int(i[0])
    cluster_two = int(i[1])
    clean_linkage_matrix.append([cluster_one,cluster_two])
print("\nClean linkage table : ",clean_linkage_matrix)

# set up the subplot for the dendrogram tree    
plt.subplot(222)
set_link_color_palette(['m', 'c', 'y', 'k'])

dendrogram = dendrogram(linkage_matrix)

# set up figure n°2 for the circle packing layout
plt.figure(2, figsize=(10,10))
ax2 = plt.subplot(223)
plt.axis([-0.1,1,0,1])
ax2.set_aspect('equal')

def new_cluster_id():
    """ Determines the id of each new cluster to be entered in the list of clusters """
    id = len(list_of_circles) - 1
    for i in list_of_clusters:
        if i[0] > id:
            id = i[0]
    id += 1
    return id

def plot_two_first_circles():
    """ Plot the two first circles of the linkage table """
    id_of_circle = int(clean_linkage_matrix[0][0])
    x1 = list_of_circles[id_of_circle][0]
    y1 = list_of_circles[id_of_circle][1]
    r1 = list_of_circles[id_of_circle][2]
    circle = plt.Circle((x1,y1),r1,fill=False)
    ax2.add_artist(circle)
    
    id_of_circle = int(clean_linkage_matrix[0][1])
    r2 = list_of_circles[id_of_circle][2]
    x2 = uniform(x1-r1-r2,high = x1+r1+r2) # select a random float for x2 which has to be at r2 distance of C1
    # it creates 2 possibilities for y2, we then choose randomly one of them to have y2
    y2a = y1 + sqrt((r1+r2)**2 - (x2-x1)**2)
    y2b = y1 - sqrt((r1+r2)**2 - (x2-x1)**2)
    two_possib = []
    two_possib.extend((y2a,y2b))
    y2 = choice(two_possib)
    circle2 = plt.Circle((x2,y2),r2,fill=False)
    ax2.add_artist(circle2)
    id_of_cluster = new_cluster_id()
    list_of_clusters.append((id_of_cluster,(x1,y1,r1),(x2,y2,r2)))
    clean_linkage_matrix.pop(0)
    print("LT after removal of first element",clean_linkage_matrix)
    print("list of clusters : ",list_of_clusters)
    
plot_two_first_circles()
print("list of circles already packed : ",list_of_clusters)

def add_singleton():
    # temporary version
    two_circles = [] # initialize list to contain the 2 circles of the cluter
    r3 = None
    for i in clean_linkage_matrix: # looking for the next element in the LT that has the 1st cluster of 2 and a singleton
        if ((i[0] == list_of_clusters[0][0]) and (i[1] < list_of_clusters[0][0])):
            print("i:",i)
            id_of_singleton = i[1]
            id_of_existing_cluster = i[0]
            print("id of singleton", i[1])
            print("id of existing cluster",i[0])            
            r3 = list_of_circles[id_of_singleton][2]
            for j in list_of_clusters:
                if (j[0] == id_of_existing_cluster):
                    print("j[0] = ",j[0])
                    two_circles.append((j[1][0],j[1][1],j[1][2]))
                    two_circles.append((j[2][0],j[2][1],j[2][2]))
        elif ((i[1] == list_of_clusters[0][0]) and (i[0] < list_of_clusters[0][0])):
            print("i:",i)
            id_of_singleton = i[0]
            id_of_existing_cluster = i[1]            
            print("id of existing cluster",i[1])            
            print("id of singleton", i[0])
            r3 = list_of_circles[id_of_singleton][2]
            for j in list_of_clusters: # get coord of the 2 circles of the cluster to pass to next function to determine coord of 3rd circle
                if (j[0] == id_of_existing_cluster):
                    print("j[0] = ",j[0])
                    two_circles.append((j[1][0],j[1][1],j[1][2]))
                    two_circles.append((j[2][0],j[2][1],j[2][2]))
        else:
            continue
    print("two circles:",two_circles)
    c1 = (two_circles[0])
    c2 = (two_circles[1])               
    return find_coord_centre_of_third_circle(c1,c2,r3)
                    
def find_coord_centre_of_third_circle(c1,c2,r3):
    """ Function that computes coordinates (x,y) and (x',y') of 3rd circle C3 that touches the 2 other circles C1 and C2. Takes as parameters tuples of the coordinates and radius of C1 and C2, and the radius of C3 """
    x1,y1,r1,x2,y2,r2,r3 = c1[0], c1[1], c1[2], c2[0], c2[1], c2[2], r3
    # solving coordinates of C3 given that C1 and C2 centres are on x axis (Pythagorus)
    x = ((r1+r2)**2 + (r1+r3)**2 - (r2+r3)**2)/(2*(r1+r2))
    y = sqrt((r1+r3)**2 - x**2)
    
    # rotating and translating the result for general case
    alpha = atan2(y2-y1,x2-x1)
    
    x3 = x*cos(alpha)-y*sin(alpha) + x1
    y3 = x*sin(alpha)+y*cos(alpha) + y1
    circle = plt.Circle((x3,y3),r3,ec = 'r',fill=False)
    ax2.add_artist(circle)
    return (x3,y3,r3)

add_singleton()
    
    
    



