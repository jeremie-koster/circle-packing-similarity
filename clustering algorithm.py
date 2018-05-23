# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:04:43 2018

@author: Jérémie
"""

from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from numpy.random import uniform, choice
import numpy as np
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
    x2 = uniform(x1-r1-r2,high = x1+r1+r2) # select a random float for x2 which has to be at r2 distance of C1's edge
    # it creates 2 possibilities for y2, we then choose randomly one of them to have y2
    y2a = y1 + sqrt((r1+r2)**2 - (x2-x1)**2)
    y2b = y1 - sqrt((r1+r2)**2 - (x2-x1)**2)
    two_possib = []
    two_possib.extend((y2a,y2b))
    y2 = choice(two_possib)
    circle2 = plt.Circle((x2,y2),r2,fill=False)
    ax2.add_artist(circle2)
    id_of_cluster = new_cluster_id()
    list_of_clusters.append((id_of_cluster,[(x1,y1,r1),(x2,y2,r2)]))
    clean_linkage_matrix.pop(0)
    print("LT after removal of first element",clean_linkage_matrix)
    print("list of clusters : ",list_of_clusters)
    
plot_two_first_circles()

def transform(translation, angle):
    """ Creates an array to apply translation and rotation to a point """
    [tx,ty]=translation
    c = cos(angle)
    s = sin(angle)
    return np.array([[c,-s,tx], [s,c,ty], [0,0,1]]) #standard rotation/tranlation matrix

def check_intersection(nb_cluster,circle):
    """ Checks if a circle is intersecting with a cluster. Returns True if so, if not, returns False. Note: can check with all the circles, easier and not too long"""
    x_to_check = circle[0]
    y_to_check = circle[1]
    r_to_check = circle[2]
    circles_in_cluster = [] # list of circles that are in the cluster to which we want to check any intersection
    intersec = False
    for i in list_of_clusters: # put circles of the cluster in a list for easier checking
        if (nb_cluster == i[0]):
            circles_in_cluster.append(i[1])
    print("circles in cluster",circles_in_cluster)
    for i in circles_in_cluster[0]:
        if ((x_to_check - i[0])**2 + (y_to_check - i[1])**2 < (r_to_check + i[2])**2):
            print("coord du cercle qui provoque l'intersec",i)
            intersec = True
    return intersec

def add_singleton():
    # temporary version - only takes 1 of the 2 solutions, works on a cluster of 2 only so far
    global list_of_clusters 
    two_circles = [] # initialize list to contain the 2 circles of the cluter
    r3 = None
    id_of_existing_cluster = None
    for i in clean_linkage_matrix: # looking for the next element in the LT that has the 1st cluster of 2 and a singleton
        if ((i[0] == list_of_clusters[0][0]) and (i[1] < list_of_clusters[0][0])): # if the 1st element is the cluster and the 2nd the singleton
            print("i:",i)
            id_of_singleton = i[1]
            id_of_existing_cluster = i[0]
            print("id of singleton", i[1])
            print("id of existing cluster",i[0])            
            r3 = list_of_circles[id_of_singleton][2]
            for j in list_of_clusters: # get coord of the 2 circles of the cluster to pass to next function to determine coord of 3rd circle
                if (j[0] == id_of_existing_cluster):
                    print("j[0] = ",j[0])
                    two_circles.append((j[1][0][0],j[1][0][1],j[1][0][2]))
                    two_circles.append((j[1][1][0],j[1][1][1],j[1][1][2]))
            clean_linkage_matrix.pop(clean_linkage_matrix.index(i)) # remove element from LT
        elif ((i[1] == list_of_clusters[0][0]) and (i[0] < list_of_clusters[0][0])): # same as before but with 1st element as the singleton and 2nd one as the cluster
            print("i:",i)
            id_of_singleton = i[0]
            id_of_existing_cluster = i[1]            
            print("id of existing cluster",i[1])            
            print("id of singleton", i[0])
            r3 = list_of_circles[id_of_singleton][2]
            for j in list_of_clusters: 
                if (j[0] == id_of_existing_cluster):
                    print("j[0] = ",j[0])
                    two_circles.append((j[1][0][0],j[1][0][1],j[1][0][2]))
                    two_circles.append((j[1][1][0],j[1][1][1],j[1][1][2]))
            clean_linkage_matrix.pop(clean_linkage_matrix.index(i))
        else:
            continue
    print("two circles:",two_circles)
    c1 = (two_circles[0])
    c2 = (two_circles[1])               
    c3 = None
    two_possible_solutions = find_coord_centre_of_third_circle(c1,c2,r3)
    print("two possible solutions for C3 : ",two_possible_solutions)
    
    which_circles_are_intersecting = []
    # TODO: CHECK INTERSECTION AND TAKE BEST CASE
    for i in two_possible_solutions:
        if (check_intersection(id_of_existing_cluster,i) == True):
#            two_possible_solutions.pop(two_possible_solutions.index(i))
#            two_possible_solutions.remove(i)
            which_circles_are_intersecting.append(i)
            print("this circle intersects with the cluster",i)
            print("two possib solutions after removal of element",two_possible_solutions)
    print("c3 after removal of all intersecting circles",c3)
    print("which circles are interescting",which_circles_are_intersecting)
    c3 = two_possible_solutions[0]
        
        
    # Update list_of_clusters: add singleton to previous cluster
    temp_list_of_circles = [circles for id,circles in list_of_clusters]
    temp_list_of_circles[0].append(c3)
    temp_list_of_circles = [item for sublist in temp_list_of_circles for item in sublist]
    print("temp list of circles with c3 : ",temp_list_of_circles)
    list_of_clusters += [(new_cluster_id(),temp_list_of_circles)]
    list_of_clusters.pop(0)
    print("list of clusters after adding singleton : ",list_of_clusters)
                    
def find_coord_centre_of_third_circle(c1,c2,r3):
    """ Function that computes coordinates (x,y) and (x',y') of 3rd circle C3 that touches the 2 other circles C1 and C2. Takes as parameters tuples of the coordinates and radius of C1 and C2, and the radius of C3 """
    list_to_return = []
    r1,r2,r3 = c1[2], c2[2], r3
    C1 = np.array([[c1[0]],[c1[1]],[1]])
    C2 = np.array([[c2[0]],[c2[1]],[1]])

    # solving coordinates of C3 given that C1 and C2 centres are on x axis (Pythagorus)
    x = ((r1+r2)**2 + (r1+r3)**2 - (r2+r3)**2)/(2*(r1+r2))
    y = sqrt((r1+r3)**2 - x**2)
    y_symmetric = -y
    C3_before_transformation = np.array([[x],[y],[1]])
    C3_symmetric_before_transformation = np.array([[x],[y_symmetric],[1]])
    
    # rotating and translating (transformation) the result for general case
    alpha = atan2(C2[1]-C1[1],C2[0]-C1[0])
    matrix = transform([C1[0],C1[1]],alpha)
    C3 = matrix@C3_before_transformation
    C3_symmetric = matrix@C3_symmetric_before_transformation
    print("C3 : ",C3)
    print("C3 symmetric : ",C3_symmetric)

    circle = plt.Circle((C3[0],C3[1]),r3,ec = 'r',fill=False)
    ax2.add_artist(circle)
    circle2 = plt.Circle((C3_symmetric[0],C3_symmetric[1]),r3,ec = 'green',fill=False)
    ax2.add_artist(circle2)
    plt.savefig("four circles")
    
    list_to_return.append((C3[0][0],C3[1][0],r3))
    list_to_return.append((C3_symmetric[0][0],C3_symmetric[1][0],r3))
    
    return list_to_return

add_singleton()
    
    
    



