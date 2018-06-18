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
from math import sqrt,cos,sin,atan2,pi
import itertools as it

# --------------------------------------------------------------- #
# --------------------- GLOBAL VARIABLES ------------------------ #
# --------------------------------------------------------------- #

width = 1
height = 1

list_of_circles = [] # to compute the linkage table only
list_without_radius = [] # to compute the linkage table only
list_of_clusters = [] # list that stores all LeafCircle and Cluster objects
counter_for_cluster_id = 0 # counter to set id of new cluster objects
number_of_leaves = 0 # only counts the nb of leaves (generated in the beginning)

# ------------------------------------------------------------- #
# ---------------------- CLASSES ------------------------------ #
# ------------------------------------------------------------- #

class LeafCircle:
    def __init__(self,id,x,y,radius):
        self.id = id
        self.x = x
        self.y = y
        self.r = radius
        
    def getHomoCoord(self):
        return np.array([[self.x],[self.y],[1]])
    
    def joinWith (self,circle_to_join): # general function
        if (circle_to_join.id < number_of_leaves): # if joining singleton with singleton
            self.joinToSingleton(circle_to_join)
            print("Join done! 1 + 1 done!")
        elif (circle_to_join.id >= number_of_leaves): # or if joining singleton with cluster
            circle_to_join.addSingleton(self)
            list_of_clusters.remove(self)
            print("Join done! n + 1 done!")
        
    def joinToSingleton (self,circle_to_add): # 1 + 1
        global counter_for_cluster_id
        circle_to_add.x = self.r + circle_to_add.r # 2nd circle touches 1st on 1 point 
        circle_to_add.y = 0 # 2nd circle on x axis
        list_of_clusters.append(Cluster(counter_for_cluster_id,self,circle_to_add)) # create new cluster
        print("homo coord",circle_to_add.getHomoCoord())
        print("x of circle to add",circle_to_add.x)
        list_of_clusters.remove(self) # and remove both leaf objects from list of clusters
        list_of_clusters.remove(circle_to_add) # same
        counter_for_cluster_id += 1
        print("counter cluster id:",counter_for_cluster_id)
           
class Cluster:
    def __init__(self,id,LeafCircle1,LeafCircle2):
        self.id = id
        self.circles = []
        self.add_circle_to_cluster(LeafCircle1)
        self.add_circle_to_cluster(LeafCircle2)
        
    def changeId(self,new_id):
        """ Function to use when 'creating' new cluster after a join of 2 sub-clusters. Instead of deleting old cluster and create a new one after the join, we change the id and increment the counter for later use """
        global counter_for_cluster_id
        self.id = new_id
        counter_for_cluster_id += 1
        print("counter cluster id:",counter_for_cluster_id)
        
    def add_circle_to_cluster (self,circle_to_add):
        self.circles.append(circle_to_add)
        
    def joinWith(self,circle_to_join):
        if (circle_to_join.id < number_of_leaves): # joining cluster with singleton
            self.addSingleton(circle_to_join)
            print("length of list cluster before removal of singleton ",len(list_of_clusters))
            list_of_clusters.remove(circle_to_join)
            print("length of list cluster after removal of singleton, should be -1 -> ",len(list_of_clusters))
            print("Join done! Singleton added to cluster!")
        elif (circle_to_join.id >= number_of_leaves): # joining cluster with cluster
            print("Not developped yet") # TODO: function n + n (see Mike's idea)
#            print("Join done!")
        
    def addSingleton(self,singleton): # n + 1 function
        """ Function that adds a singleton to a cluster. 
        For each pair of the cluster, if the pair is not too distant, do 1+2 function and store the 2 possibilities. Then, exclude all overlapping elements and choose best possib. Adds the circle object to the list of circles of the cluster, and 'creates' new cluster by changing id   """
        list_of_possib = [] # list of possibilities (should contain LeafCircle objects)
        list_of_checked_elements = [] # list that contains legal elements (no overlapping)
        i = 0
        print("There are {0} circles in the cluster".format(len(self.circles)))
        for pair in it.combinations(self.circles,2):
            if ((pair[1].x - pair[0].x)**2 + (pair[1].y - pair[0].y)**2 <= ((2*singleton.r) + pair[0].r + pair[1].r)**2): # check if the circles of the pair are not too distant from each other
                list_of_possib.extend(twoPlusOne(pair,singleton)) # add the 2 possib when doing 2 + 1
        print("There are {0} elem in list of possib".format(len(list_of_possib)))
        for element in list_of_possib: # remove overlapping elements
            print("{0}th element is going to be checked for intersection".format(i))
            i += 1
            if (checkIntersection(self,element) == False):
                list_of_checked_elements.append(element)           
                print("An element has been added to list of checked elements")
        print("{0} elem have been checked".format(i))
        print("{0} elements in list of possib are going to be compared to take best one".format(len(list_of_checked_elements)))
        self.add_circle_to_cluster(self.chooseBestPossib(list_of_checked_elements)) # take best possib and add it to the cluster
        self.changeId(counter_for_cluster_id) # pretend to create new cluster. Instead, change cluster id and increment counter
        
    def addCluster(self,cluster2):
        """ Cluster 1 is fixed. Adding cluster 2 agains cluster 1. """
        # TODO: To reduce computational cost, make sure that both pairs are on the outside layer of the cluster
        list_of_possib = combineTwoPairs(self,cluster2)
        # Check intersections
        # Choose best possibility
                        
        
    
    def chooseBestPossib(self,list):
        """ Function that computes the best possibility to place a new circle in a cluster to have a circle shape. Requires a Cluster object and a list of LeafCircle possibilities """
        longest_distances = [] # list of the biggest distances for each possibility
        
        for possib in list:
            list_of_distances = [] # contains the longest dist for the current possib
            for circle in self.circles:
                d = sqrt((possib.y - circle.y)**2 + (possib.x - circle.x)**2)    
                if (len(list_of_distances) == 0): # for the first iteration
                    list_of_distances.append((d,possib))
                elif (d > list_of_distances[0][0]): # replace element if d is bigger   
                    list_of_distances.pop(0)
                    list_of_distances.append((d,possib))
            longest_distances.append(list_of_distances[0])
        
        while len(longest_distances) > 1: # extract the LeafCircle corresp. to small dist
            if (longest_distances[0][0] >= longest_distances[1][0]):
                del longest_distances[0]
            elif (longest_distances[0][0] < longest_distances[1][0]):
                del longest_distances[1]
        print("length longest distances list",len(longest_distances))
        return longest_distances[0][1] # return LeafCircle object corresponding to smallest dist
    
# ------------------------------------------------------------------- #     
# ------------------------- FUNCTIONS ------------------------------- #
# ------------------------------------------------------------------- #

def runLinkageTable():
    while len(clean_linkage_matrix) > 0: # as long as LT is not empty...
        id_of_first_cluster = clean_linkage_matrix[0][0]
        id_of_second_cluster = clean_linkage_matrix[0][1]
        print("-------I want to join {0} with {1}------".format(id_of_first_cluster,id_of_second_cluster))
        for c in list_of_clusters:
            for d in list_of_clusters:
                if ((c.id == id_of_first_cluster) & (d.id == id_of_second_cluster)):
                    print("Trying to join {0} with {1}".format(c.id,d.id))
                    c.joinWith(d)
                    clean_linkage_matrix.pop(0) # deletes current element after use
                else:
                    print("Wrong combination: c.id = {0} and d.id = {1} // LT asks for {2} and {3}".format(c.id,d.id,id_of_first_cluster,id_of_second_cluster))
                    
def transform(translation, angle):
    """ Creates an array to apply translation and rotation to a point """
    [tx,ty]=translation
    c = cos(angle)
    s = sin(angle)
    return np.array([[c,-s,tx], [s,c,ty], [0,0,1]]) #standard rotation/tranlation matrix

def checkIntersection(cluster,circle_to_check):
    """ Checks if a circle is intersecting with any circle of a cluster. Returns True if so, if not, returns False """
    x_to_check = circle_to_check.x
    y_to_check = circle_to_check.y
    r_to_check = circle_to_check.r
    circles_in_cluster = cluster.circles # list of circles (as LeafCircle objects) that are in the cluster to which we want to check any intersection
    intersec = False
    for i in circles_in_cluster:
        if (round(sqrt((x_to_check - i.x)**2 + (y_to_check - i.y)**2),4) < round((r_to_check + i.r),4)): # I'm using round() because I was stuck
            intersec = True
            print("circle is intersecting")
    print("partie gauche",round((x_to_check - i.x)**2 + (y_to_check - i.y)**2,4))
    print("partie droite",round(((r_to_check + i.r)**2),4))
    print("bool check intersection ->",intersec)
    return intersec
    
def twoPlusOne(cluster,singleton): # 2 + 1 function
    """ Function that computes coordinates (x,y) and (x',y') of 3rd circle C3 that touches the 2 other circles C1 and C2. Takes as parameters tuples of the coordinates and radius of C1 and C2, and the radius of C3 """
    C1 = cluster[0].getHomoCoord()
    C2 = cluster[1].getHomoCoord()
    r1,r2,r3 = cluster[0].r, cluster[1].r, singleton.r
    list_to_return = [] # will contain 2 LeafCircle objects of the 2 solutions

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
#    print("C3 : {0},{1},{2}".format(C3[0][0],C3[1][0],r3))
#    print("C3 symmetric : {0},{1},{2}".format(C3_symmetric[0][0],C3_symmetric[1][0],r3))
# =============================================================================
#     circle = plt.Circle((C3[0],C3[1]),r3,ec = 'r',fill=False)
#     ax2.add_artist(circle)
#     circle2 = plt.Circle((C3_symmetric[0],C3_symmetric[1]),r3,ec = 'green',fill=False)
#     ax2.add_artist(circle2)
#     plt.savefig("four circles")
# =============================================================================
    list_to_return.append(LeafCircle(singleton.id,C3[0][0],C3[1][0],r3))
    list_to_return.append(LeafCircle(singleton.id,C3_symmetric[0][0],C3_symmetric[1][0],r3))
    
    return list_to_return        

def combineTwoPairs(cluster1,cluster2):
    for pair1 in it.combinations(cluster1,2):
        if (round((pair1[1].x - pair1[0].x)**2 + (pair1[1].y - pair1[0].y)**2,4) == round((pair1[0].r + pair1[1].r)**2,4)): # check if the 2 circles of the pair are touching themselves
            for pair2 in it.combinations(cluster2.circles,2):
                if (round((pair2[1].x - pair2[0].x)**2 + (pair2[1].y - pair2[0].y)**2,4) == round((pair2[0].r + pair2[1].r)**2,4)): # check if the 2 circles of the pair are touching themselves
                    possib_circle_1 = twoPlusOne(pair1,pair2[0]) # My decision: take only 1 set of solutions                    
    
# ------------------------------------------------------------------ #
# ------------------------- LINKAGE TABLE CREATION ----------------- #
# ------------------------------------------------------------------ #        

def generate_circles(n):
    global counter_for_cluster_id
    global number_of_leaves
    while len(list_of_circles) < n: # as long as the list of circles is not full     
        print("id: ",len(list_of_circles))
        x = uniform(0.05, high = width - 0.05)
        y = uniform(0.05, high = width - 0.05)
        r = uniform(0.02, high = 0.05)
        list_of_clusters.append(LeafCircle(len(list_of_circles),0,0,r)) # add LeafCircle object
        list_of_circles.append([x,y,r])
    counter_for_cluster_id = n
    number_of_leaves = n
        
    
def take_radius_out(list):
    for item in list:
        list_without_radius.append((item[0],item[1]))
    return list_without_radius

generate_circles(5)
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

# keep only the clusters' id from linkage table and convert them into int for following processing
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

runLinkageTable()

# set up figure n°2 for the circle packing layout
plt.figure(2, figsize=(10,10))
ax2 = plt.subplot(223)
plt.axis([-0.1,0.5,-0.3,0.5])
ax2.set_aspect('equal')

for circle in list_of_clusters[0].circles:
    x = circle.x
    y = circle.y
    r = circle.r
    circle = plt.Circle((x,y),r,fill=False)
    ax2.add_artist(circle)


# ----- END ------- #


    
    
    



