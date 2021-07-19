# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 15:56:19 2021

@author: inet
"""

#!/usr/bin/env python
# coding: utf-8

import networkx as nx
#import matplotlib.pyplot as plt
from string import ascii_lowercase
import pandas as pd
import math
import random
import sys
import re
import itertools 
import xml.etree.ElementTree as ET
from sys import argv
import argparse
from time import sleep
import os
import subprocess
import json
import random
import copy

# graph name and required nodes
name_arg= sys.argv[1]
# name_arg= 'largeSNR.graphml'

no_subSet_Arg1= sys.argv[2]
# no_subSet_Arg1= 8


g = nx.read_graphml(name_arg)
x=int(no_subSet_Arg1)-1
g= nx.Graph(g)
pos = nx.kamada_kawai_layout(g)
print(nx.info(g))

#global lst_nodes_test1 = pd.DataFrame()
#global lst_nodes_new = pd.DataFrame()
# BW, lat, Cost
w1=float(sys.argv[3])
w2=float(sys.argv[4])
w3=float(sys.argv[5])


# w1=0.4    #bandwidth
# w2=.2     #latency
# w3=0.4    #cost

# ### Graph Labeling for Custom and TopologyZoo Networks

# edge_labels = nx.get_edge_attributes(g, 'bw')
# plt.figure(figsize=(12, 10))
# nx.draw_networkx_nodes(g,  pos, node_size=300, label=1)
# nx.draw_networkx_edges(g, pos, width=2, edge_color='black')
# nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=9)
# nx.draw_networkx_labels(g, pos, font_size=12)
# plt.title('Graph Representation of %s topology' % (name_arg) )
# plt.show()


# ### Delay Calculation Among Nodes for TopologyZoo Networks

def read_graph(name_arg):
    Topo = str(name_arg)
    if (Topo=='largeSNR'or Topo=='mediumSNR' or Topo=='smallSNR'):
        True
    else:
        xml_tree    = ET.parse(Topo+'.graphml')
        namespace   = "{http://graphml.graphdrawing.org/xmlns}"
        ns          = namespace # just doing shortcutting, namespace is needed often.

        #GET ALL ELEMENTS THAT ARE PARENTS OF ELEMENTS NEEDED LATER ON
        root_element    = xml_tree.getroot()
        graph_element   = root_element.find(ns + 'graph')

        # GET ALL ELEMENT SETS NEEDED LATER ON
        index_values_set    = root_element.findall(ns + 'key')
        node_set            = graph_element.findall(ns + 'node')
        edge_set            = graph_element.findall(ns + 'edge')

        # SET SOME VARIABLES TO SAVE FOUND DATA FIRST
        # memomorize the values' ids to search for in current topology
        node_label_name_in_graphml = ''
        node_latitude_name_in_graphml = ''
        node_longitude_name_in_graphml = ''
        # for saving the current values
        node_index_value     = ''
        node_name_value      = ''
        node_longitude_value = ''
        node_latitude_value  = ''
        # id:value dictionaries
        id_node_name_dict   = {}     # to hold all 'id: node_name_value' pairs
        id_longitude_dict   = {}     # to hold all 'id: node_longitude_value' pairs
        id_latitude_dict    = {}     # to hold all 'id: node_latitude_value' pairs

        # FIND OUT WHAT KEYS ARE TO BE USED, SINCE THIS DIFFERS IN DIFFERENT GRAPHML TOPOLOGIES
        for i in index_values_set:

            if i.attrib['attr.name'] == 'label' and i.attrib['for'] == 'node':
                node_label_name_in_graphml = i.attrib['id']
            if i.attrib['attr.name'] == 'Longitude':
                node_longitude_name_in_graphml = i.attrib['id']
            if i.attrib['attr.name'] == 'Latitude':
                node_latitude_name_in_graphml = i.attrib['id']

        # NOW PARSE ELEMENT SETS TO GET THE DATA FOR THE TOPO
        # GET NODE_NAME DATA
        # GET LONGITUDE DATK
        # GET LATITUDE DATA
        for n in node_set:

            node_index_value = n.attrib['id']

            #get all data elements residing under all node elements
            data_set = n.findall(ns + 'data')

            #finally get all needed values
            for d in data_set:

                #node name
                if d.attrib['key'] == node_label_name_in_graphml:
                    #strip all whitespace from names so they can be used as id's
                    node_name_value = re.sub(r'\s+', '', d.text)
                #longitude data
                if d.attrib['key'] == node_longitude_name_in_graphml:
                    node_longitude_value = d.text
                #latitude data
                if d.attrib['key'] == node_latitude_name_in_graphml:
                    node_latitude_value = d.text

                #save id:data couple
                id_node_name_dict[node_index_value] = node_name_value
                id_longitude_dict[node_index_value] = node_longitude_value
                id_latitude_dict[node_index_value]  = node_latitude_value

        for e in edge_set:

            # GET IDS FOR EASIER HANDLING
            e[0] = e.attrib['source']
            e[1] = e.attrib['target']
            latitude_src= math.radians(float(id_latitude_dict[e[0]]))
            latitude_dst= math.radians(float(id_latitude_dict[e[1]]))
            longitude_src= math.radians(float(id_longitude_dict[e[0]]))
            longitude_dst= math.radians(float(id_longitude_dict[e[1]]))
            first_product               = math.sin(latitude_dst) * math.sin(latitude_src)
            second_product_first_part   = math.cos(latitude_dst) * math.cos(latitude_src)
            second_product_second_part  = math.cos(longitude_dst - longitude_src)
            distance = (math.acos(first_product + (second_product_first_part * second_product_second_part))) * 6378.137

            # t (in ms) = ( distance in km * 1000 (for meters) ) / ( speed of light / 1000 (for ms))
            # t         = ( distance       * 1000              ) / ( 1.97 * 10**8   / 1000         )
            latency = ( distance * 1000 ) / ( 197000 )
            g[e[0]][e[1]]['latency']=round(latency,2)
            g[e[0]][e[1]]['bw']=round (random.uniform(7.0,14.0), 1)
            g[e[0]][e[1]]['cost']=round (random.uniform(0.1,0.5), 2)
        return g
            

# ### Rank Based First Node Selection

def get_firstnode_RankBased():
    nodes=list(g.nodes)
    lst_nodes_test = list()
    lst_nodes_test1 = pd.DataFrame()
    lst_nodes_avg_wgt = pd.DataFrame()
    lst_nodes_avg_bw = pd.DataFrame()
    
    for node in nodes:
        neighbors_of_Node = g[node]
        sumLat=sumBW=sumCost = 0
        count = 0
        
        for n_node in neighbors_of_Node:
            d = g.get_edge_data(node,n_node,default=0)
            bw = d['bw']
            lat = d['latency']
            cost = d['cost']
            sumLat += lat
            sumBW += bw
            sumCost += cost
            count+=1
        lst_nodes_avg_wgt = lst_nodes_avg_wgt.append({"node": node, "latency": sumLat/count, "bw": sumBW/count,
                                                     "cost": sumCost/count},ignore_index=True)
    #print(lst_nodes_avg_wgt)
    bw_max = lst_nodes_avg_wgt.bw.max()
    lat_min = lst_nodes_avg_wgt.latency.min()
    cost_min = lst_nodes_avg_wgt.cost.min()
    
    bw_min = lst_nodes_avg_wgt.bw.min()
    lat_max = lst_nodes_avg_wgt.latency.max()
    cost_max = lst_nodes_avg_wgt.cost.max()
    
    bw_sum = lst_nodes_avg_wgt.bw.sum()
    lat_sum= lst_nodes_avg_wgt.latency.sum()
    cost_sum = lst_nodes_avg_wgt.cost.sum()

      
    for index, row in lst_nodes_avg_wgt.iterrows():
        if (bw_max - bw_min)==0:
            x=0
        else:
            x=w1*((bw_max - row.bw ) / (bw_max - bw_min))
        if (lat_max - lat_min) == 0:
            y=0
        else:
            y=w2*((lat_max - row.latency) / (lat_max - lat_min))
        if (cost_max - cost_min) == 0 :
            z=0
        else:
            z=w3*((cost_max - row.cost) / (cost_max - cost_min))
        # if (row.bw - bw_min) == 0 or (bw_max - bw_min)==0:
        #     x=0
        # else:
        #     x=w1*((row.bw - bw_min) / (bw_max - bw_min))
        # if (row.latency - lat_min) == 0 or (lat_max - lat_min) == 0:
        #     y=0
        # else:
        #     y=w2*((row.latency - lat_min) / (lat_max - lat_min))
        # if (row.cost- cost_min) == 0 or (cost_max - cost_min) == 0 :
        #     z=0
        # else:
        #     z=w3*((row.cost- cost_min) / (cost_max - cost_min))
        temp = {"node": row.node, "bw_nor": x, "lat_nor": y, "cost_nor": z}
        lst_nodes_test1= lst_nodes_test1.append(temp,  ignore_index=True )


    lst_nodes_test1["total_sum"]=lst_nodes_test1.sum(axis=1)
    lst_nodes_test1['ranking'] = lst_nodes_test1.total_sum.rank(ascending=False)
    #print('lst_node(fistnode):: ',lst_nodes_test1)
    #print("lst_nodes_test1['ranking'] ",lst_nodes_test1['ranking'] )

    # For bandwith, set the total sum to 'max'
    # For latency and cost, set the total sum to 'min'
    lst = lst_nodes_test1[lst_nodes_test1.total_sum  == lst_nodes_test1.total_sum.max()]
    lst_nodes_new = (lst_nodes_test1)
    #print('lst_nodes_new:: ',lst_nodes_new)

    #print('lst.node: ', l)
    #node_with_min_weight = int(lst.node)
    node_with_min_weight = lst.node
   
    #Print all the list
    return node_with_min_weight, lst_nodes_new

#print('lst_nodes_new:: ',lst_nodes_new)

def node_rank(node, lst_Neighbors):
    
    # lst_nodes_test1 = pd.DataFrame()
    # lst_nodes_test = pd.DataFrame()
    # #print('lst_Neighbors', lst_Neighbors)
    # bw_max = lst_Neighbors.bw.max()
    # lat_min = lst_Neighbors.latency.min()
    # cost_min = lst_Neighbors.cost.min()
    
    # bw_min = lst_Neighbors.bw.min()
    # lat_max = lst_Neighbors.latency.max()
    # cost_max = lst_Neighbors.cost.max()
    
    # bw_sum = lst_Neighbors.bw.sum()
    # lat_sum = lst_Neighbors.latency.sum()
    # cost_sum = lst_Neighbors.cost.sum()

    #lst_nodes_new['ranking'] = lst_nodes_new['total_sum'].rank(ascending=False)
    #lst_nodes_new=lst_nodes_test1
    #print('lst_nodes_new:: ',lst_nodes_new)

    lst_nodes_new["total_sum"]=lst_nodes_new["bw_nor"] + lst_nodes_new["lat_nor"] + lst_nodes_new["cost_nor"]
    #print('rank before ', lst_nodes_new)
    lst_nodes_new['ranking'] = lst_nodes_new['total_sum'].rank(ascending=True)
    lst_nodes_new.sort_values('ranking', inplace=True, ascending=[False])

    lst_nodes_new.reset_index(drop=False)
    #print('rank after: ', lst_nodes_new)

    min_wgt_node = lst_nodes_new.iloc[0,3]
    #print('insdeloop ', min_wgt_node)
    return min_wgt_node



## Algorithm for Selecting Subset of Nodes -- " SNR " Method

##  Node Rank & Neighbor 
def rank(node, lst_Neighbors):
    
    lst_nodes_test1 = pd.DataFrame()
    lst_nodes_test = pd.DataFrame()
    #print('lst_Neighbors', lst_Neighbors)
    bw_max = lst_Neighbors.bw.max()
    lat_min = lst_Neighbors.latency.min()
    cost_min = lst_Neighbors.cost.min()
    
    bw_min = lst_Neighbors.bw.min()
    lat_max = lst_Neighbors.latency.max()
    cost_max = lst_Neighbors.cost.max()
    
    bw_sum = lst_Neighbors.bw.sum()
    lat_sum = lst_Neighbors.latency.sum()
    cost_sum = lst_Neighbors.cost.sum()

    for index, row in lst_Neighbors.iterrows():
        if (bw_max - bw_min)==0:
            x=0
        else:
            x=w1*((bw_max - row.bw ) / (bw_max - bw_min))
        if (lat_max - lat_min) == 0:
            y=0
        else:
            y=w2*((lat_max - row.latency) / (lat_max - lat_min))
        if (cost_max - cost_min) == 0 :
            z=0
        else:
            z=w3*((cost_max - row.cost) / (cost_max - cost_min))
        temp = {"node": row.node, "neighbor": row.neighbor, "bw_nor": x, "lat_nor": y, "cost_nor": z}
        lst_nodes_test1= lst_nodes_test1.append(temp,  ignore_index=True)
    lst_nodes_test1["Total_sum"]=lst_nodes_test1["bw_nor"] + lst_nodes_test1["lat_nor"] + lst_nodes_test1["cost_nor"]
    lst_nodes_test1['ranking'] = lst_nodes_test1['Total_sum'].rank(ascending=False)
    lst_nodes_test1.reset_index(drop=True)
    min_wgt_node = lst_nodes_test1.iloc[0,1]
    return lst_nodes_test1


##  Alogrithm for Selecting Subset of Node Cluster
def subSet_of_Nodes2(Num_Of_Nodes, get_firstnode):
    firstnode = str(get_firstnode)
    currentnode = firstnode
    nodes=list(g.nodes)
    lst_VisitedNodes=[]
    lst_VisitedNodes.append(str(firstnode))
    neighborsList=[]
    #add neighbors of the first node
    tmp=g.neighbors(firstnode)
    for n in tmp:
        neighborsList.append(n)
        
    lst_nodes_new["total_sum"]=lst_nodes_new["bw_nor"] + lst_nodes_new["lat_nor"] + lst_nodes_new["cost_nor"]
    #print('rank before ', lst_nodes_new)
    lst_nodes_new['ranking'] = lst_nodes_new['total_sum'].rank(ascending=True)
    lst_nodes_new.reset_index(drop=False)
    # print('lst_nodes_new',lst_nodes_new)
    # print('neighborsList',neighborsList)
    #We look for nodes to add to the current set
    while(len(lst_VisitedNodes)<Num_Of_Nodes):
        # print('\n')
        maxRank=0
        bestNode=0
        for tmpNode in neighborsList:
            nextNodeRank=lst_nodes_new.iloc[int(tmpNode),5]
            nextNode=lst_nodes_new.iloc[int(tmpNode),3]
            #print('nextNode: ',nextNode)
            if nextNodeRank>maxRank:
                maxRank=nextNodeRank
                bestNode=nextNode
        if bestNode not in lst_VisitedNodes:
            lst_VisitedNodes.append(str(bestNode))
            neighborsList.remove(str(bestNode))
        #print ('lst_VisitedNodes: ',lst_VisitedNodes)
        #print ('neighborsList: ',neighborsList)
        tmp=g.neighbors(str(int(bestNode)))
        for no in tmp:
            if str(no) not in neighborsList and str(no) not in lst_VisitedNodes:
                neighborsList.append(str(no))
        
    #return the list of all visited nodes.
    #print ('lst_VisitedNodes: ',lst_VisitedNodes)
    return lst_VisitedNodes

def subSet_of_Nodes(Num_Of_Nodes, get_firstnode):
    firstnode = str(get_firstnode)
    currentnode = firstnode
    nodes=list(g.nodes)

    #list to maintain the all the visited nodes
    lst_VisitedNodes = pd.DataFrame()
    lst_VisitedNodes = lst_VisitedNodes.append({"node": firstnode, "visited node": firstnode, "path":'-'},
                                               ignore_index=True)
    path = firstnode
    lst_Neighbors_Rank = pd.DataFrame()
    lst= []
    for n in nodes:
    
        neighbors_of_CurrentNode = g[str(currentnode)]
        # print('neighbors_of_CurrentNode:', neighbors_of_CurrentNode)
        lst_Neighbors = pd.DataFrame()
        if(lst_VisitedNodes.shape[0] >1):

            neighbors_of_CurrentNode = [ele for ele in neighbors_of_CurrentNode if ele not in list(lst_VisitedNodes['visited node'])] 
            nodes = [i for i in nodes if i not in list(lst_VisitedNodes['visited node'])]
            lst_Neighbors_Rank['neighbor'] = list(map(int, lst_Neighbors_Rank['neighbor']))
            lst_Neighbors_Rank = lst_Neighbors_Rank[~lst_Neighbors_Rank['neighbor'].isin(list(lst_VisitedNodes['visited node']))] 
        
        #Add the neighbors of the current node to the list of all neighbors
        for node in neighbors_of_CurrentNode:
            d = g.get_edge_data(str(currentnode),str(node),default=0)
            bw = d['bw']
            lat = d['latency']
            cost = d['cost']
            lst_Neighbors = lst_Neighbors.append({"node": currentnode, "neighbor": node, "latency": lat, "bw": bw, "cost":cost},
                                                 ignore_index=True)
            
        if(len(lst_Neighbors)>0):
            ranktable = node_rank(currentnode, lst_Neighbors)
            # print('ranktable: ', ranktable)
            lst_Neighbors_Rank = lst_Neighbors_Rank.append(ranktable)       
            lst_Neighbors_Rank['neighbor'] = list(map(int, lst_Neighbors_Rank['neighbor']))
            lst_Neighbors_Rank = lst_Neighbors_Rank[~lst_Neighbors_Rank.neighbor.isin(list(map(int, list(lst_VisitedNodes['visited node']))) )]

        # Check if the list of the neighbor nodes is not empty
        if(lst_Neighbors_Rank.shape[0] >0):
            
            # ****** On the base of sum ****** #
            lst = lst_Neighbors_Rank[lst_Neighbors_Rank.Total_sum  == lst_Neighbors_Rank.Total_sum.max()]
            #node_with_max_rank = int(lst.neighbor)
            #print('lst:' , lst.iloc[0]['neighbor'])
            node_with_max_rank = int(lst.iloc[0]['neighbor'])
            
            # ****** On the basis of rank ******#
            #lst2 = ranktable[ranktable.ranking  == ranktable.ranking.min()]
            #node_with_min_weight = int(lst2.neighbor)
          
            # Update the path with the current node's neighbor
            path = str(path) + ',' + str(node_with_max_rank)
            if (lst_VisitedNodes.shape[0] > Num_Of_Nodes):
                return lst_VisitedNodes
            
            else:
                # add the node with minimum weight to the visited node list
                lst_VisitedNodes = lst_VisitedNodes.append({"node": currentnode, "visited node": node_with_max_rank,'path': path}, ignore_index=True)
                currentnode = node_with_max_rank
        else: 
            currentnode = firstnode
            path = firstnode
            
    #return the list of all visited nodes.
    return lst_VisitedNodes



# ### Get Average Delay/Cost/BW of the Selected Nodes:

def getAvgDelay(path_final):
    avg = []
    sum=0
    for x,y in itertools.combinations(path_final, 2):
        sum = getPathDelay(x, y, Final_graph)
        avg.append(round(sum,2))
    return avg

def getAvgCost(path_final):
    avg = []
    sum=0
    for x,y in itertools.combinations(path_final, 2):
        sum = getPathCost(x, y, Final_graph)
        avg.append(round(sum,2))
    return avg

def getAvgBW(path_final):
    avg = []
    sum=0
    for x,y in itertools.combinations(path_final, 2):
        sum = getPathBW(x, y, Final_graph)
        avg.append(round(sum,2))
    return avg

def getPathDelay(firstNode, EndNode, p_graph):
    path = nx.shortest_path(p_graph, str(firstNode), str(EndNode))
    sub_graph = g.subgraph(path)
    sum = 0
    for edge in sub_graph.edges(data=True):
        lat= edge[2]['latency']
        sum+=lat
    return (sum)

def getPathCost(firstNode, EndNode, p_graph):
    path = nx.shortest_path(p_graph, str(firstNode), str(EndNode))
    sub_graph = g.subgraph(path)
    sum = 0
    for edge in sub_graph.edges(data=True):
        lat= edge[2]['cost']
        sum+=lat
    return (sum)

def getPathBW(firstNode, EndNode, p_graph):
    path = nx.shortest_path(p_graph, str(firstNode), str(EndNode))
    sub_graph = g.subgraph(path)
    sum = 0
    lst =[]
    for edge in sub_graph.edges(data=True):
        lat= edge[2]['bw']
        lst.append(lat)
        #sum+=lat
    return (min(lst))

def hop(path_final,g):
    hop  = []
    for x,y in itertools.combinations(path_final, 2):
        path = nx.shortest_path(g, str(x), str(y))
        hop1=(len(path)-1)
        hop.append(hop1)
    return hop

def sum_list(items):
    sum_num = 0
    for x in items:
        sum_num += x
    return sum_num


# ### Selected Required Subset of Node & Results (SNR)

print(' ')
print('**** SNR Method ****')
print('**** Required Subset of Node & Results: Starting node as node with highest Rank ***** ')
print(' ')

#Result of the Subset of nodes
Topo=read_graph(name_arg.replace(".graphml", ""))
startNode,lst_nodes_new=(get_firstnode_RankBased())
print('startNode: ', startNode)
hh=list(startNode)
#print('hh:',hh[0],type(hh))

#starting Node for the node slection


startNode=hh[0]
#startNode=str(0)

#starting Node for the node slection




result = subSet_of_Nodes2(x,startNode)
# print('result:',result)
# final_path=list(result.iloc[-1].tail(2).head(1))
# final_path=final_path[0]
final_path=result
#print('final_path:',final_path)
# final_path=final_path.split(",")
#print('final_path:',final_path)
Final_graph = g.subgraph(final_path)
#print('Final_graph:',type(Final_graph))
path_final  = list(Final_graph.nodes)

#Delays Calculation
AvgDelay=list(getAvgDelay(final_path))
avg_delay_value = round(sum_list(AvgDelay)/len(AvgDelay),2)
min_delay_value = min(AvgDelay)
max_delay_value = max(AvgDelay)

#BW Calculation
AvgBW=list(getAvgBW(final_path))
avg_bw_value = round(sum_list(AvgBW)/len(AvgBW),2)
min_bw_value = min(AvgBW)
max_bw_value = max(AvgBW)

#Cost Calculation
AvgCost=list(getAvgCost(final_path))
avg_cost_value = round(sum_list(AvgCost)/len(AvgCost),2)
min_cost_value = min(AvgCost)
max_cost_value = max(AvgCost)

#No. of Hop Calculation
Hop_No = hop(final_path, Final_graph)
avg_hop_value = round(sum_list(Hop_No)/len(Hop_No),2)
min_hop_value = min(Hop_No)
max_hop_value = max(Hop_No)

#Results Output
print('The Required Subset of Nodes:', x)
print('Selected Starting Node is: ', startNode)
print('list of Selected Subset of Nodes: ', final_path)
print(' ')
print('List of Delays', AvgDelay)
print('The min latency value in selected subset:', min_delay_value)
print('The max latency value in selected subset:', max_delay_value)
print('The avg latency value in selected subset:', avg_delay_value)
#print('sum of the delay in selected subset:', sum_list(AvgDelay))
print(' ')
print('List of BW', AvgBW)
print('The min bw value in selected subset:', min_bw_value)
print('The max bw value in selected subset:', max_bw_value)
print('The avg bw value in selected subset:', avg_bw_value)
print(' ')
print('List of Cost', AvgCost)
print('The min cost value in selected subset:', min_cost_value)
print('The max cost value in selected subset:', max_cost_value)
print('The avg cost value in selected subset:', avg_cost_value)
print('sum of the cost in selected subset:', round(sum_list(AvgCost),2))
print(' ')
print('List of hops', Hop_No)
print('The min no. of hops traversed in selected subset:', min_hop_value)
print('The max no. of hops traversed in selected subset:', max_hop_value)
print('The avg no. of hops traversed in selected subset:', avg_hop_value)
print(' ')

#print (result)
