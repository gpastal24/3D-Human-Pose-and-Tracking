#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 13:21:19 2022

@author: gpastal
"""
import numpy as np
import math
def centers_3c(xyz,xyz2,xyz3):
    centers1=[]
    centers2=[]
    centers3=[]
    def _non_zero(array):
        mask=np.empty(17)
        for ind,ar in enumerate(array):
            if ar.all()==0 or not np.any(ar):
                mask[ind]=0
            else:
                mask[ind]=1
             
        return mask
                
    # for ind,array in enumerate(xyz):
        
    #     center=np.array([[0,0,0]])
    #     counter=0
    #     for l in array:
    #         if l.all()==0 or not np.any(l):
    #             continue
    #         else:
    #             center=center+l
    #             counter = counter +1
    #     centers1.append(center/(counter))
    # for ind,array in enumerate(xyz2):
    #     center=np.array([[0,0,0]])
    #     counter=0
    #     for l in array:
    #         if l.all()==0 or not np.any(l):
    #             continue
    #         else:
    #             center=center+l
    #             counter = counter +1
    #     centers2.append(center/(counter))
        
    # for ind,array in enumerate(xyz3):
    #     center=np.array([[0,0,0]])
    #     counter=0
    #     for l in array:
    #         if l.all()==0 or not np.any(l):
    #             continue
    #         else:
    #             center=center+l
    #             counter = counter +1
    #     centers3.append(center/(counter))
    # for ind,array in enumerate(xyz):
    #     # if array[11].all()!=0 and array[12].all()!=0:
    #     masks = _non_zero(array)
    #     count1 = masks[5]+masks[6]+masks[11]+masks[12]
    #     centers1.append((array[5]+array[6]+array[11]+array[12])/count1)
    # for ind,array in enumerate(xyz2):
    #     masks = _non_zero(array)
    #     count2 = masks[5]+masks[6]+masks[11]+masks[12]
    #     centers2.append((array[5]+array[6]+array[11]+array[12])/count2)
    # for ind,array in enumerate(xyz3):
    #     masks = _non_zero(array)
    #     count3 = masks[5]+masks[6]+masks[11]+masks[12]
    #     centers3.append((array[5]+array[6]+array[11]+array[12])/count3)
    for ind,array in enumerate(xyz):
        masks = _non_zero(array)
        count1 = masks[0]
        centers1.append(array[0])
    for ind,array in enumerate(xyz2):
        masks = _non_zero(array)
        count2 = masks[0]
        centers2.append(array[0])
    for ind,array in enumerate(xyz3):
        masks = _non_zero(array)
        count3 = masks[0]
        centers3.append(array[0])
    return centers1,centers2,centers3

def cross_id_3c(centers1,centers2,centers3,threshold = .50):
     list_=[]
     list1=[]
     list2=[]
     list3=[]
     list4=[]
     list5=[]
     list6=[]
     matrix2=np.empty((len(centers1),len(centers2)))
     matrix = np.empty((len(centers1),len(centers3)))
     matrix3 = np.empty((len(centers2),len(centers3)))
     # if len(centers1)>=len(centers2):
     for ind,array in enumerate(centers1):
             # if centers1.all==0:
             #     continue
             # else:
                 # print(array)
                 # max_dist=1000
                 for ind2,array2 in enumerate(centers2):
                     
                     dist = np.linalg.norm(array-array2)
                     matrix2[ind,ind2]=dist
     if matrix2.size!=0:
         for inde,array in enumerate(matrix2):        
             res= np.unravel_index(array.argmin(),(1,len(centers2)))
             # print(array)
             # print(res)
             if array[res[1]]<threshold:
                 list1.append(inde)
                 list2.append(res[1])
         
     for ind,array in enumerate(centers1):
             # if centers1.all==0:
             #     continue
             # else:
                 # print(array)
                 # max_dist=1000
                 for ind2,array2 in enumerate(centers3):
                     
                     dist = np.linalg.norm(array-array2)
                     matrix[ind,ind2]=dist
     if matrix.size!=0:
         for inde,array in enumerate(matrix):        
             res= np.unravel_index(array.argmin(),(1,len(centers3)))
             # print(array)
             # print(res)
             if array[res[1]]<threshold:
                 list3.append(inde)
                 list4.append(res[1])
         
     for ind,array in enumerate(centers2):
            # if centers1.all==0:
            #     continue
            # else:
                # print(array)
                # max_dist=1000
                for ind2,array2 in enumerate(centers3):
                    
                    dist = np.linalg.norm(array-array2)
                    matrix3[ind,ind2]=dist
     if matrix3.size!=0:
        for inde,array in enumerate(matrix3):        
           res= np.unravel_index(array.argmin(),(1,len(centers3)))
           # print(array)
           # print(res)
           if array[res[1]]<threshold:
               list5.append(inde)
               list6.append(res[1])
            # for i in range(len(list2)):
            #     for j in range(len(list2)):
            #         if list2[i]==list2[j] and i!=j:list2[j]=-1
    
     return list1,list2,list3,list4,list5,list6

def cross_id_3c_new(centers1,centers2,centers3,threshold = .50):
     list_=[]
     list1=[]
     list2=[]
     list3=[]
     list4=[]
     list5=[]
     list6=[]
     matrix2=np.empty((len(centers1),len(centers2)))
     matrix = np.empty((len(centers1),len(centers3)))
     matrix3 = np.empty((len(centers2),len(centers3)))
     # if len(centers1)>=len(centers2):
     for ind,array in enumerate(centers1):
             # if centers1.all==0:
             #     continue
             # else:
                 # print(array)
                 # max_dist=1000
                 for ind2,array2 in enumerate(centers2):
                     
                     dist = np.linalg.norm(array-array2)
                     matrix2[ind,ind2]=dist
     if matrix2.size!=0:
         for inde,array in enumerate(matrix2):        
             res= np.unravel_index(array.argmin(),(1,len(centers2)))
             # print(array)
             # print(res)
             if array[res[1]]<threshold:
                 list1.append(inde)
                 list2.append(res[1])
         
     for ind,array in enumerate(centers1):
             # if centers1.all==0:
             #     continue
             # else:
                 # print(array)
                 # max_dist=1000
                 for ind2,array2 in enumerate(centers3):
                     
                     dist = np.linalg.norm(array-array2)
                     matrix[ind,ind2]=dist
     if matrix.size!=0:
         for inde,array in enumerate(matrix):        
             res= np.unravel_index(array.argmin(),(1,len(centers3)))
             # print(array)
             # print(res)
             if array[res[1]]<threshold:
                 list3.append(inde)
                 list4.append(res[1])
         
     for ind,array in enumerate(centers2):
            # if centers1.all==0:
            #     continue
            # else:
                # print(array)
                # max_dist=1000
                for ind2,array2 in enumerate(centers3):
                    
                    dist = np.linalg.norm(array-array2)
                    matrix3[ind,ind2]=dist
     if matrix3.size!=0:
        for inde,array in enumerate(matrix3):        
           res= np.unravel_index(array.argmin(),(1,len(centers3)))
           # print(array)
           # print(res)
           if array[res[1]]<threshold:
               list5.append(inde)
               list6.append(res[1])
            # for i in range(len(list2)):
            #     for j in range(len(list2)):
            #         if list2[i]==list2[j] and i!=j:list2[j]=-1
    
     return matrix,matrix2,matrix3
            
            
def fin_points_3c(xyz,xyz2,xyz3,ids1,ids2,ids3,ids4,ids5,ids6,confs1,confs2,confs3):
    last_points=[]
    # merged_ids=[ids1]+[ids2]
    processed_2=[]
    processed_3=[]
    id_counter=0
    id_counter2=0
    for i in range(len(xyz)):
       # id_counter=0
       # id_counter2=0
       if i not in ids1 and i not in ids3:
          
           last_points.append(xyz[i])
         
       elif i not in ids3 and i in ids1:
           
           temp=[]
           
           for j in range(17):
               if xyz[i][j].any()!=0 and xyz2[ids2[id_counter]][j].any()!=0:
                   array=xyz[i][j] +xyz2[ids2[id_counter]][j]
                   temp.append(array/2)
               elif xyz[i][j].all()==0:
                   temp.append(xyz2[ids2[id_counter]][j])
               elif xyz2[ids2[id_counter]][j].all()==0:
                       temp.append(xyz[i][j])
               else:
                   temp.append(xyz[i][j])
                   
           last_points.append(temp)
           processed_2.append(ids2[id_counter])
           # ids2[id_counter]=-1
           id_counter=id_counter+1
       elif i not in ids1 and i in ids3:
            
            temp=[]
            
            for j in range(17):
                if xyz[i][j].any()!=0 and xyz3[ids4[id_counter2]][j].any()!=0:
                    array=xyz[i][j] +xyz3[ids4[id_counter2]][j]
                    temp.append(array/2)
                elif xyz[i][j].all()==0:
                    temp.append(xyz3[ids4[id_counter2]][j])
                elif xyz3[ids4[id_counter2]][j].all()==0:
                        temp.append(xyz[i][j])
                else:
                    temp.append(xyz[i][j])
                    
            last_points.append(temp)
            processed_3.append(ids4[id_counter2])
            # ids2[id_counter]==None
            # ids4[id_counter2] = -1
            id_counter2=id_counter2+1
       else:
           temp=[]
           
           for j in range(17):
               if xyz[i][j].any()!=0 and xyz2[ids2[id_counter]][j].any()!=0 and xyz3[ids4[id_counter2]][j].any()!=0:
                   array=xyz[i][j] +xyz2[ids2[id_counter]][j] + xyz3[ids4[id_counter2]][j] 
                #    ar = np.asarray([confs1[i][j],confs2[ids2[id_counter][j]],confs3[ids4[id_counter2][j]]])
                #    array = np.dot(array,ar)
                #    ar = sorted(ar)
                   # print(array)
                #    print('True')
                   temp.append(array/3)
               elif xyz[i][j].all()==0 and xyz2[ids2[id_counter]][j].any()!=0 and xyz3[ids4[id_counter2]][j].any()!=0:
                   array = xyz2[ids2[id_counter]][j] + xyz3[ids4[id_counter2]][j]
                   temp.append(array/2)
               elif xyz[i][j].all()==0 and xyz2[ids2[id_counter]][j].any()==0 and xyz3[ids4[id_counter2]][j].any()!=0:
                       temp.append(xyz3[ids4[id_counter2]][j])
                       
               elif xyz[i][j].all()==0 and xyz2[ids2[id_counter]][j].any()!=0 and xyz3[ids4[id_counter2]][j].any()==0:
                     temp.append(xyz2[ids2[id_counter]][j])
                     
               elif xyz[i][j].all()!=0 and xyz2[ids2[id_counter]][j].any()==0 and xyz3[ids4[id_counter2]][j].any()!=0:
                    array = xyz[i][j] + xyz3[ids4[id_counter2]][j]
                    temp.append(array/2)
               elif xyz[i][j].all()!=0 and xyz2[ids2[id_counter]][j].any()!=0 and xyz3[ids4[id_counter2]][j].any()==0:
                    array = xyz[i][j] + xyz2[ids2[id_counter]][j]
                    temp.append(array/2)
               else:
                   temp.append(xyz[i][j])
                   
           last_points.append(temp)
           processed_2.append(ids2[id_counter])
           processed_3.append(ids4[id_counter2])
           # print(ids1)
           # print(ids2[id_counter])
           # print(ids4[id_counter2])
           # print(temp)
           # ids2[id_counter]==None
           # ids2[id_counter] = -1
           # ids4[id_counter2] = -1
           # print(id_counter)
           id_counter2 = id_counter2+1
           id_counter = id_counter+1
    id_counter=0 
    for i in range(len(xyz2)):
        # id_counter=0
        if i not in ids2 and i not in ids5:
            last_points.append(xyz2[i])
        # elif i in ids2 and i in ids5:
        #     id_counter=id_counter+1
        elif  i in ids5: #i not in ids2 and 
            # id_counter=0
            if i in processed_2:
                # print('True')
                id_counter=id_counter+1 
                continue 
                # id_counter=id_counter+1 
            else:
                
                temp=[]
                # print(id_counter)
                for j in range(17):
                    if xyz2[i][j].any()!=0 and xyz3[ids6[id_counter]][j].any()!=0:
                        # print(array)
                        array=xyz2[i][j] +xyz3[ids6[id_counter]][j]
                        # print(array)
                        # print(i)
                        temp.append(array/2)
                    elif xyz2 and xyz2[i][j].all()==0:
                        temp.append(xyz3[ids6[i]][j])
                    elif xyz3[ids6[id_counter]][j].all()==0 and xyz:
                            temp.append(xyz2[i][j])
                    else:
                        temp.append(xyz2[i][j])
                        
                last_points.append(temp)
                # processed_3.append(ids6[i])
                # ids2[id_counter]==None
                # ids6[id_counter] = -1
                id_counter=id_counter+1
              
    for i in range(len(xyz3)):
        if i not in ids4 and i not in ids6:
            last_points.append(xyz3[i])
    return last_points

# def fin_points_3c(xyz,xyz2,xyz3,ids)

# xyz = np.random.randint(5)
# for i in range(3):
#     # xyz = np.empty((3,17))
#     x = np.random.rand(17) * 5
#     y = np.random.rand(17) * 2
#     z = np.random.rand(17) * 5

#     xyz = np.stack((x,y,z),axis=-1)
# #     print(xyz.shape)
xyz=[]
xyz2=[]
xyz3=[]
for i in range(3):
    temp=[]
    for j in range(17):
        temp.append(np.ones(3)*(i+1) + np.random.rand(3)/5)
    xyz.append(temp)
for i in range(3):
    temp=[]
    for j in range(17):
        temp.append(np.ones(3)*(i*0.5+1))
    xyz2.append(temp)
for i in range(3):
    temp=[]
    for j in range(17):
        temp.append(np.ones(3)*(7-i))
    xyz3.append(temp)


def match_tids_matrices(tids1,tids2,tids3,mat1,mat2,mat3,threshold = 0.3):
    matched_ids = []
    if mat1.size!=0:
        for ind,v in enumerate(mat1):
            res= np.unravel_index(v.argmin(),(1,len(tids3)))
            if v[res[1]]<threshold:
                print(res)
            # else:
    if mat2.size!=0:
        for ind,v in enumerate(mat2):
            res= np.unravel_index(v.argmin(),(1,len(tids2)))
            if v[res[1]]<threshold:
                print(res)
    if mat3.size!=0:
        for ind,v in enumerate(mat3):
            # print(v)
            res= np.unravel_index(v.argmin(),(1,len(tids3)))
            if v[res[1]]<threshold:
                print(res)
# centers1,centers2,centers3 = centers_3c(xyz,xyz2,xyz3)
# crossid1,crossid2,crossid3,crossid4,crossid5,crossid6 = cross_id_3c(centers1,centers2,centers3,threshold =0.5)
# a=fin_points_3c(xyz,xyz2,xyz3,crossid1,crossid2,crossid3,crossid4,crossid5,crossid6)
# tids1 = [1,3,5]
# tids2 = [2,4,6]
# tids3 = [57,8,9]


# mat1=np.random.random((len(tids1), len(tids3)))
# mat2 = np.random.random((len(tids1), len(tids2)))
# mat3 = np.random.random((len(tids2), len(tids3)))
# mat1 = np.array([[0,1,1],[1,0,1],[0,0,1]])
# print(mat2)

def match_tids_matrices(tids1,tids2,tids3,mat1,mat2,mat3,threshold = 0.5):
    matched_ids1 = []
    matched_ids2 = []
    matched_ids3 = []
    final_matched = []
    if mat1.size!=0:
        for ind,v in enumerate(mat1):
            # print(ind)

            res= np.unravel_index(v.argmin(),(len(tids1),len(tids3)))
            # print(res[1])
            # print(v)
            if v[res[1]]<threshold:
                # print(v[res[1]])
                matched_ids3.append([tids1[ind],tids3[res[1]]])
                
            # else:
    if mat2.size!=0:
        for ind,v in enumerate(mat2):
            # print(ind)
            res= np.unravel_index(v.argmin(),(len(tids1),len(tids2)))
            if v[res[1]]<threshold:
                matched_ids1.append([tids1[ind],tids2[res[1]]])
                # print("")
    if mat3.size!=0:
        for ind,v in enumerate(mat3):
            # print(v)
            res= np.unravel_index(v.argmin(),(len(tids2),len(tids3)))
            if v[res[1]]<threshold:
                matched_ids2.append([tids2[ind],tids3[res[1]]])
                # print("")
    for i in tids1:
        flat1 = [item[0] for item in matched_ids1 ]
        flat2 = [item[0] for item in matched_ids3 ]
        if i not in flat1 and i not in flat2:
            # print('yeah',i)
            final_matched.append([i,None,None])
    for i in tids2:
        flat1 = [item[1] for item in matched_ids1 ]
        flat2 = [item[0] for item in matched_ids2 ]
        if i not in flat1 and i not in flat2:
            # print('yeah',i)
            final_matched.append([None,i,None])
    for i in tids3:
        flat1 = [item[1] for item in matched_ids2 ]
        flat2 = [item[1] for item in matched_ids3 ]
        if i not in flat1 and i not in flat2 :
            # print('yeah',i)
            final_matched.append([None,None,i])
    # print(matched_ids1)
    # print(matched_ids2)
    # print(matched_ids3)
    return matched_ids1,matched_ids2,matched_ids3,final_matched

# m1 ,m2,m3,final_matched =match_tids_matrices(tids1,tids2,tids3,mat1,mat2,mat3)
# print(m1)
# print(m2)
# print(m3)
# print(final_matched)

def find_final_tids_matched(m1,m2,m3,final_matched):
    ids12 = [i[0] for i in m1]
    ids13 = [i[0] for i in m3]
    common = list(set(ids12).intersection(ids13))
    non_common = [i for i in ids12 if i not in common]
    non_common2 = [i for i in ids13 if i not in common]
    ids12 = [i[1] for i in m1]
    ids23 = [i[0] for i in m2]
    common_ = list(set(ids12).intersection(ids23))
    non_common_ = [i for i in ids12 if i not in common]
    non_common2_ = [i for i in ids23 if i not in common]
    for i in common:
        tid = [k for k in m1 if k[0]==i]
        tid= tid[0].copy()
        
        tid2 = [k for k in m3 if k[0]==i]
        tid2 = tid2[0].copy()
        tid.append(tid2[1])
        if tid not in final_matched:
            final_matched.append(tid)
    for i in common_:
        tid = [k for k in m1 if k[1]==i]
        tid= tid[0].copy()
        
        tid2 = [k for k in m2 if k[0]==i]
        tid2 = tid2[0].copy()
        tid.append(tid2[1])
        if tid not in final_matched:
            final_matched.append(tid)
    for i in non_common:  
        tid = [k for k in m1 if k[0]==i]
        if tid:
            tid = tid[0].copy()
            f = [j for j in final_matched]
            found=False
            for j in f:
                # print("j=" , j)
                if tid[0] in j and tid[1] in j:
                    # print('found')
                    found = True
            if not found:
                tid.append(None)
                
                if tid not in final_matched:
                    final_matched.append(tid)
    for i in non_common2:
        tid = [k for k in m3 if k[0]==i]
        if tid: 
            tid= tid[0].copy()
            f = [j for j in final_matched]
            found=False
            for j in f:
                # print("j=" , j)
                if tid[0] in j and tid[1] in j:
                    # print('found')
                    found = True
            if not found:
                tid.append(None)
                tid[1],tid[2] = tid[2],tid[1]
                if tid not in final_matched:
                    final_matched.append(tid)  
        
    for i in non_common_:  
        tid = [k for k in m1 if k[1]==i]
        if tid:
            tid = tid[0].copy()
            f = [j for j in final_matched]
            found=False
            for j in f:
                # print("j=" , j)
                if tid[0] in j and tid[1] in j:
                    # print('found')
                    found = True
            if not found:
                tid.append(None)
                # tid[0],tid[2] = tid[2],tid[0]
                if tid not in final_matched:
                    final_matched.append(tid)
    for i in non_common2_:
        tid = [k for k in m2 if k[0]==i]
        if tid: 
            tid= tid[0].copy()
            f = [j for j in final_matched]
            found=False
            for j in f:
                # print("j=" , j)
                if tid[0] in j and tid[1] in j:
                    # print('found')
                    found = True
            # print(tid)
            if tid not in final_matched and not found:
                tid.append(None)
            # print(tid)
                tid[1],tid[2] = tid[2],tid[1]
                tid[0],tid[1] = tid[1],tid[0]
                
                final_matched.append(tid)
        
    return final_matched
    # print(final_matched)
    # print(m1)
    # print(m2)
    # print(m3)
# fin_ids = find_final_tids_matched(m1, m2, m3, final_matched)
# print(fin_ids)

def calculate_final_poses(fin_ids,tids1,tids2,tids3,xyz,xyz2,xyz3,pose_confs1,pose_confs2,pose_confs3):
    # print(fin_ids)
    final_pose_list = []
    # print(fin_ids)

    for i in fin_ids:
        t1,t2,t3 = zip(i)
        # print(t1[0])
        # valid_poses = [False]*3
        valid_poses = [True  if t!=None else False for t in i ]
        # print(valid_poses)
        # print(t1)
        
        # print(t2)
        # print(t3)
        # print(fin_ids)
        list_ind1 = tids1.index(t1[0]) if t1[0]!=None else None
        list_ind2 = tids2.index(t2[0]) if t2[0]!=None else None
        list_ind3 = tids3.index(t3[0]) if t3[0]!=None else None
        # print(list_ind1,list_ind2,list_ind3)
        pose1,conf1 = (xyz[list_ind1],pose_confs1[list_ind1]) if list_ind1 is not None else (np.zeros((17,3)),np.zeros((17,1)))
        # print(pose1)
        pose2,conf2 = (xyz2[list_ind2],pose_confs2[list_ind2]) if list_ind2 is not None else (np.zeros((17,3)),np.zeros((17,1)))
        pose3,conf3 = (xyz3[list_ind3],pose_confs3[list_ind3]) if list_ind3 is not None else (np.zeros((17,3)),np.zeros((17,1)))
        pose_merged = average_kpt(pose1, pose2, pose3, conf1, conf2, conf3, valid_poses)
        final_pose_list.append(pose_merged)
    return final_pose_list
def average_kpt(pose1,pose2,pose3,conf1,conf2,conf3,valid_poses,mode='weighted_sum'):
    len_valid = [1 for j in valid_poses if j]
    len_valid = sum(len_valid)
    # print(len_valid)
    pose = np.zeros((17,3))
    for j in range(17):
        points = [pose1[j],pose2[j],pose3[j]]
        confs = [conf1[j],conf2[j],conf3[j]]
        poi_conf = zip(points,confs)
        try:
            xyFiltered = [(x, y) for x, y in poi_conf if x.any()!=0.0 ]
            points, confs = zip(*xyFiltered)
        except:
            xyFiltered = zip([],[])
            points,confs = [],[]
        if mode=='average':
            point = sum(points)/len(points) if points else np.zeros((3))
        elif mode=='top1k':
            if points:
                # print(confs)
                # print(points)
                sorted_list = [(y,x) for y, x in sorted(zip(confs, points),key= lambda confs:confs[0],reverse=True)]
                confs,points = zip(*sorted_list)
                point = points[:1] if points else np.zeros(3)
            else:
                point = np.zeros(3)
        elif mode == 'top2k':
            if points:
                sorted_list = [(y,x) for y, x in sorted(zip(confs, points),key= lambda confs:confs[0],reverse=True)]
                confs,points = zip(*sorted_list)
                if len(points) >= 2:
                    point = (points[0]+points[1])/2
                elif len(points) == 1:
                    point = points[0]
                else:
                    point = np.zeros((3))
            else:
                point = np.zeros((3))
        elif mode =='weighted_sum':
            if points:
                # sorted_list = [(y,x) for y, x in sorted(zip(confs, points),key= lambda confs:confs[0],reverse=True)]
                # confs,points = zip(*sorted_list)
                confs = np.asarray(confs)
                point = np.dot(np.asarray(points).T,confs)/sum(confs)
        else:
            point=np.zeros((3))

        pt = np.array(point)
        # print(pt)
        pose[j] = pt
    return pose
#  def filter_point(point):
    
        
# calculate_final_poses(fin_ids=fin_ids, tids1=tids1, tids2=tids2, tids3=tids3, xyz=xyz, xyz2=xyz2, xyz3=xyz3,\
#     pose_confs1=np.random.random((3,17)), pose_confs2=np.random.random((3,17)), pose_confs3=np.random.random((3,17)))
    
class ids(object):
    def __init__(self):
        # self.id1 = None
        # self.id2 = None
        # self.id3 = None
        self.f_id = None
        self.tracked = False
        self.head_pose = []
        self.last_updated=0

class Merged_ids(object):
    def __init__(self):
        self.f_id = -1
        self.f_ids = []
        # pass
    def increase_id(self):
        self.f_id -=1
    def assign_tid(self,final_ids):
        final_ids = final_ids.copy()
        if self.f_ids:
            tids = [t for t in self.f_ids if t.tracked]
            tids1 = [t.id1 for t in tids]
            tids2 = [t.id2 for t in tids]
            tids3 = [t.id3 for t in tids]

        else:
            tids1,tids2,tids3 = [],[],[]
        for f in final_ids:
            t1,t2,t3 = zip(f)
            # print(t1)
            t1=t1[0]
            t2=t2[0]
            t3 = t3[0]
            if t1 is not None and t1 in tids1:
                inde = tids1.index(t1)
                # print(inde)

                self.f_ids[inde].id1 = t1
                self.f_ids[inde].id2 = t2
                self.f_ids[inde].id3 = t3
                
            elif t2 is not None and t2 in tids2:
                inde = tids2.index(t2)
                # print(inde)

                self.f_ids[inde].id1 = t1
                self.f_ids[inde].id2 = t2
                self.f_ids[inde].id3 = t3
            elif t3 is not None and t3 in tids3:
                inde = tids3.index(t3)
                # print(inde)
                self.f_ids[inde].id1 = t1
                self.f_ids[inde].id2 = t2
                self.f_ids[inde].id3 = t3
            elif t1 is not None or t2 is not None or t3 is not None:
                t = ids()
                t.id1 = t1
                t.id2 = t2
                t.id3 = t3
                t.f_id = self.f_id
                t.tracked = True
                self.increase_id()
                self.f_ids.append(t)
        if final_ids:
            # t1,t2,t3 = [(f[0],f[1],f[0]) for f in final_ids]
            f = [f for f in final_ids]
            t1 = [t[0] for t in f]
            t2 = [t[1] for t in f]
            t3 = [t[2] for t in f]


            # [print(f) for f in final_ids]

        else:
            t1,t2,t3 = [],[],[]
        # print(self.f_ids)
        for indee,t_ in enumerate(zip(tids1,tids2,tids3)):
            # print(t_)
            any_= False
            if t_[0] in t1:
                any_=True
            if t_[1] in t2:
                any_=True
            if t_[2] in t3:
                any_=True
            if not any_:
                self.f_ids[indee].tracked = False
                self.f_ids[indee].last_updated +=1
        # self.delete_ids(final_ids)
        kept_ids = self.clear_ids()
        # self.delete_ids(final_ids)
        return kept_ids
    def clear_ids(self):
        to_keep = [f for f in self.f_ids if  f.tracked]
        return to_keep
    # @staticmethod
    # def assign_tracklets_to_tids(self,final_ids)
    #     if final_ids:
    #             # t1,t2,t3 = [(f[0],f[1],f[0]) for f in final_ids]
    #             f = [f for f in final_ids]
    #             t1 = [t[0] for t in f]
    #             t2 = [t[1] for t in f]
    #             t3 = [t[2] for t in f]


    #             # [print(f) for f in final_ids]

    #         else:
    #             t1,t2,t3 = [],[],[]
    #         # print(self.f_ids)
    #     if self.f_ids:
    #         tids = [t for t in self.f_ids if t.tracked]
    #         tids1 = [t.id1 for t in tids]
    #         tids2 = [t.id2 for t in tids]
    #         tids3 = [t.id3 for t in tids]

    #     else:
    #         tids1,tids2,tids3 = [],[],[]
    #     for inde, t in enumerate(zip(t1,t2,t3)):
    #         for indee,t_ in enumerate(zip(tids1,tids2,tids3)):
    #             max_occur = 0
    #             max_occur_index=0
    #             indices = []
    #             # for inde,
    #                 # for indee,t_ in enumerate(zip(tids1,tids2,tids3)):
    #             # print(t_)
    #             # for inde,t in enumerate(zip(t))
    #             any_= [False]*3
    #             if t_[0] in t1:
    #                 any[0]_= True
    #             if t_[1] in t2:
    #                 any_[1]=True
    #             if t_[2] in t3:
    #                 any_[2]=True
    #             len_valid = [1 for j in any_ if j]
    #             len_valid = sum(len_valid)
    #             if len_valid>max_occur:
    #                 max_occur = len_valid
    #                 max_occur_index = indee
    #                 indices.append(indee)
            
    # def delete_ids(self,final_ids):
    #     if self.f_ids:
    #         tids = [t for t in self.f_ids if t.tracked]
    #         tids1 = [t.id1 for t in tids]
    #         tids2 = [t.id2 for t in tids]
    #         tids3 = [t.id3 for t in tids]

    #     else:
    #         tids1,tids2,tids3 = [],[],[]
        
    #     # print(f)
    #     f1 = [f[0] for f in final_ids]
    #     f2  = [f[1] for f in final_ids]
    #     f2 = [f[2] for f in final_ids]
    #     # print(len(self.f_ids))
    #     for t in tids1:
    #         # [print(x.id1) for i, x in enumerate(self.f_ids) ]
    #         indices = [i for i, x in enumerate(self.f_ids) if x.id1 == t]
    #         # print(indices)
    #         lens_ = []
    #         first = True
    #         for i in indices:
    #             if first:
    #                 first=False
    #             else:
    #                 self.f_ids[i].tracked=False

                
                

    #     for t in tids2:
    #         # [print(x.id1) for i, x in enumerate(self.f_ids) ]
    #         indices = [i for i, x in enumerate(self.f_ids) if x.id2 == t]
    #         # print(indices)
    #         lens_ = []
    #         first = True
    #         for i in indices:
    #             if first:
    #                 first=False
    #             else:
    #                 self.f_ids[i].tracked=False
    #     for t in tids3:
    #         # [print(x.id1) for i, x in enumerate(self.f_ids) ]
    #         indices = [i for i, x in enumerate(self.f_ids) if x.id3 == t]
    #         lens_ = []
    #         first = True
    #         for i in indices:
    #             if first:
    #                 first=False
    #             else:
    #                 self.f_ids[i].tracked=False
        
            # print(indices)
        # for t in tids2:
        #     for f in final_ids:
        #         if t == f[1]: print(t)
        # for t in tids3:
        #     for f in final_ids:
        #         if t == f[2]: print(t)
        
        # for f in self.f_ids
        # indices = [i for i, x in enumerate(self.f_ids) if x == "whatever"]
        # self.f_ids = [f for f in self.f_ids if f.last_updated<25 and f.tracked]
            # print(t_)
            # valid = filter(None,t_)
            # print(valid)
        # for ind,i in enumerate(self.f_ids):
        #     if i.tracked:
        #         print(i.f_id)
        #         id1 = i.id1
        #         id2 = i.id2
        #         id3 = i.id3
        #         if id1 is not None and id1 not in t1:
        #             self.f_ids.pop(ind)
        #         elif id2 is not None and id2 not in t2:
        #             self.f_ids.pop(ind)
        #         elif id3 is not None and id3 not in t3:
        #             self.f_ids.pop(ind)
        #     if not i.tracked:
        #         self.f_ids.pop(ind)
class ids(object):
    def __init__(self):
        self.id1 = None
        self.id2 = None
        self.id3 = None
        self.yaw = []
        self.dist = []
        self.cam_yaw = []
        self.pose = None
        self.f_id = None
        self.tracked = False
        self.head_pose = []
        self.center_med = None
        self.updated = False
        self.last_updated=0
class pose_3d_tracker(object):
    def __init__(self):
        
        self.f_id = -1
        self.f_ids = []
        # pass
    def increase_id(self):
        self.f_id -=1

    def assign_ids(self,pose_last,final_ids):
        poses = pose_last.copy()
        poses = np.asarray(poses)
        # print(poses)
        if not self.f_ids:
            for inde,i in enumerate(poses):
                # print(i)
                j = i[~np.all(i == 0, axis=1)]
                # print(j)
                j = np.median(j,axis=0)
                # print(j)
                tracklet = ids()
                tracklet.f_id = self.f_id
                tracklet.id1 = final_ids[inde][0]
                tracklet.id2 = final_ids[inde][1]
                tracklet.id3 = final_ids[inde][2]
                tracklet.center_med = j
                tracklet.tracked = True
                # self.update
                self.f_ids.append(tracklet)
                self.increase_id()
        else:
            indices = []
            for inde,i in enumerate(poses):

                j = i[~np.all(i == 0, axis=1)]

                # print(j)
                prev_centers = [cent.center_med for cent in self.f_ids]
                j = np.median(j,axis=0)
                # print(j)
                # print(final_ids[inde])
                res = self.cost_matrix(j,prev_centers)
                # print(res)
                if res is None:
                    # j = i[i!= np.array([0.,0.,0.])]
                    # # print(j)
                    # j = np.median(j,axis=0)
                    # print(j)
                    tracklet = ids()
                    tracklet.f_id = self.f_id
                    tracklet.id1 = final_ids[inde][0]
                    tracklet.id2 = final_ids[inde][1]
                    tracklet.id3 = final_ids[inde][2]
                    tracklet.center_med = j
                    tracklet.tracked = True
                    self.f_ids.append(tracklet)
                    ind_tracklet = [i for i,v in enumerate(self.f_ids) if v.f_id == tracklet.f_id]
                    indices.append(ind_tracklet[0])
                    self.increase_id()
                else:
                    self.f_ids[res].id1 = final_ids[inde][0]                
                    self.f_ids[res].id2 = final_ids[inde][1]
                    self.f_ids[res].id3 = final_ids[inde][2]
                    # print(self.f_ids[res].id1,self.f_ids[res].id2)
                    self.f_ids[res].center_med = j
                    self.f_ids[res].tracked = True
                    indices.append(res)
            # print(indices)
            indices_true = [i for i,v in enumerate(self.f_ids)]
            # print(indices_true)

            indices_to_remove = set(indices_true).symmetric_difference(indices)
            # print(indices_to_remove)
            # if indices_to_remove:
                # print(indices_to_remove)
            self.f_ids = [v for i,v in enumerate(self.f_ids) if i not in frozenset(indices_to_remove)]
        
        self.update_id()
        return self.f_ids

    @staticmethod
    def cost_matrix(pose,prev_pose,threshold=0.3):
        # print(pose.shape)
        matrix = np.empty((pose.shape[0],len(prev_pose)))
        # print(matrix.shape)
        # for ind,array in enumerate(pose):
        #     # print(array)
        for ind2,array2 in enumerate(prev_pose):
            # print(pose)
            # print(array2)
            dist = np.linalg.norm(pose-array2)
            matrix[0,ind2]=dist
        # print(matrix)
        if matrix.size!=0:
            for ind,v in enumerate(matrix):
                # print(ind)
                res= np.unravel_index(v.argmin(),(len(pose),len(prev_pose)))
                if v[res[1]]<threshold:
                    return res[1]
                else:
                    return None
    def update_id(self):
        ids = [v.f_id for v in self.f_ids]
        if ids:
            max_id = min(ids)
            self.f_id = max_id-1

    @staticmethod
    def dist_score(dist):
        return np.exp(-abs((dist-1.5)))
    @staticmethod
    def softmax(x):
        return(np.exp(x)/np.exp(x).sum())
    @staticmethod
    def cam_yaw_score(yaw):
        return np.exp(-abs((yaw)))
    def assign_head_poses(self,head_poses):
        t_ids1 = [f.id1 for f in  self.f_ids]
        t_ids2 = [f.id2 for f in self.f_ids]
        t_ids3 = [f.id3 for f in self.f_ids]
        # print(t_ids1)
        # print(t_ids2)

        for i in range(len(self.f_ids)):
            self.f_ids[i].yaw=[]
            self.f_ids[i].dist=[]
            self.f_ids[i].cam_yaw=[]
        # if len(head_poses)>1:
        # print(len(head_poses))
        for i in head_poses:
            
            ids = [f['tid'] for f in i]
            euler_angles = [f['head_pose_rotated_euler'] for f in  i]
            distances = [f['cam_distance'] for f in i]
            cam_rot = [f['head_pose'] for f in i]
            # print(ids)
            yaw=[]
            for indee,v in enumerate(ids):
                if v in t_ids1 and v is not None:
                    index = t_ids1.index(v)
                    # print(euler_angles)
                    # print(indee)
                    self.f_ids[index].yaw.append(euler_angles[indee][1])
                    self.f_ids[index].dist.append(self.dist_score(distances[indee]))
                    self.f_ids[index].cam_yaw.append(self.cam_yaw_score(cam_rot[indee][1]))

                if v in t_ids2 and v is not None:
                    index = t_ids2.index(v)
                    self.f_ids[index].yaw.append(euler_angles[indee][1])
                    self.f_ids[index].dist.append(self.dist_score(distances[indee]))
                    self.f_ids[index].cam_yaw.append(self.cam_yaw_score(cam_rot[indee][1]))

                if v in t_ids3 and v is not None:
                    index = t_ids3.index(v)
                    self.f_ids[index].yaw.append(euler_angles[indee][1])
                    self.f_ids[index].dist.append(self.dist_score(distances[indee]))
                    self.f_ids[index].cam_yaw.append(self.cam_yaw_score(cam_rot[indee][1]))

        yaws = [f.yaw for f in self.f_ids]
        dists = [f.dist for f in self.f_ids]
        cam_yaws = [f.cam_yaw for f in self.f_ids]
        # # yaws_ = yaws[-10:]
        # dists_ = dists[-10:]
        # cam_yaws_ = cam_yaws[-10:]
        # print(yaws_)
        # print(dists_)
        # print(cam_yaws_)
        # print(len(yaws_))
        final_yaw=[]
        for yaw,dist,cam_yaw in zip(yaws,dists,cam_yaws):
            # print(len(yaw))
            if len(yaw)==0:
                final_yaw.append(None)
            elif len(yaw)==1:
                # print(yaw)
                # print(sum(yaw))
                final_yaw.append(sum(yaw)/len(yaw))
            else:
                # print(dist)
                scores = self.softmax(dist)
                scores_yaw = self.softmax(cam_yaw)
                fin_score = scores+scores_yaw
                final_scores = self.softmax(fin_score)
                # print(scores)
                # print(scores_yaw)
                print(final_scores)
                yaw_fin = np.dot(np.asarray(yaw),np.asarray(final_scores)/sum(final_scores))
                final_yaw.append(yaw_fin)
                # final_yaw.append(sum(yaw)/len(yaw))
        return final_yaw
class pose_3d_tracker_FVP(object):
    def __init__(self,inside):
         
        #self.f_id = 0
        self.f_ids = []
        self.inside = inside
        if self.inside:
            self.f_id=1
        else:
            self.f_id=-1
        # pass
    def increase_id(self):
        if self.inside:
            self.f_id +=1
        else:
            self.f_id -=1

    def assign_ids(self,pose_last):
        poses = pose_last.copy()
        poses = np.asarray(poses)
        # print(poses)
        if not self.f_ids:
            for inde,i in enumerate(poses):
                # print(i)
                j = i[~np.all(i == 0, axis=1)]
                # print(j)
                j = np.median(j,axis=0)
                # print(j)
                tracklet = ids()
                tracklet.f_id = self.f_id
                tracklet.center_med = j
                tracklet.pose = i
                tracklet.tracked = True
                # self.update
                self.f_ids.append(tracklet)
                self.increase_id()
        else:
            indices = []
            indices_pose = []
            for inde,i in enumerate(poses):

                j = i[~np.all(i == 0, axis=1)]

                # print(j)
                prev_centers = [cent.center_med for cent in self.f_ids]
                j = np.median(j,axis=0)
                # print(j)
                indices_pose.append(inde)
                res = self.cost_matrix(j,prev_centers)
                # print(res)
                if res is None:
                    # j = i[i!= np.array([0.,0.,0.])]
                    # # print(j)
                    # j = np.median(j,axis=0)
                    # print(j)
                    tracklet = ids()
                    tracklet.f_id = self.f_id

                    tracklet.center_med = j
                    tracklet.pose = i

                    tracklet.tracked = True
                    self.f_ids.append(tracklet)
                    ind_tracklet = [i for i,v in enumerate(self.f_ids) if v.f_id == tracklet.f_id]
                    indices.append(ind_tracklet[0])
                    self.increase_id()
                else:

                    self.f_ids[res].center_med = j
                    self.f_ids[res].pose = i

                    self.f_ids[res].tracked = True
                    indices.append(res)
            
            # print(indices_pose)
            self.f_ids = [self.f_ids[i] for i in indices]
            indices_true = [i for i,v in enumerate(self.f_ids)]
            # print(indices_true)

            indices_to_remove = set(indices_true).symmetric_difference(indices)
            # print(indices_to_remove)
            # print(indices)
            # if indices_to_remove:
                # print(indices_to_remove)
            self.f_ids = [v for i,v in enumerate(self.f_ids) if i not in frozenset(indices_to_remove)]
            # indices = [v for i,v in enumerate(self.f_ids) if i not in frozenset(indices_to_remove)]
            # if indices:
                # print(indices)
                # self.f_ids = [self.f_ids[i] for i in indices if i < len(self.f_ids)]
                # if len(poses) > self.f_ids:

            # poses_indeces = [i for i,v in enumerate(poses)]
        self.update_id()
        return self.f_ids

    @staticmethod
    def cost_matrix(pose,prev_pose,threshold=0.3):
        # print(pose.shape)
        matrix = np.empty((pose.shape[0],len(prev_pose)))
        # print(matrix.shape)
        # for ind,array in enumerate(pose):
        #     # print(array)
        for ind2,array2 in enumerate(prev_pose):
            # print(pose)
            # print(array2)
            dist = np.linalg.norm(pose-array2)
            matrix[0,ind2]=dist
        # print(matrix)
        if matrix.size!=0:
            for ind,v in enumerate(matrix):
                # print(ind)
                res= np.unravel_index(v.argmin(),(len(pose),len(prev_pose)))
                if v[res[1]]<threshold:
                    return res[1]
                else:
                    return None
    def update_id(self):
        # if ids:
        ids = [v.f_id for v in self.f_ids]
        if ids:
            if self.inside:
                max_id = max(ids)
                self.f_id = max_id+1
            else:
                max_id = min(ids)
                self.f_id = max_id-1
