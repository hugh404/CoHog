
"""
The code for the paper "CoHOG: A Light-weight, Compute-efficient and Training-free Visual Place Recognition Technique for Changing Environments" has been modified and implemented purely in Python.

If you need to access the original source code, please visit the link [https://github.com/MubarizZaffar/CoHOG_Results_RAL2019](https://github.com/MubarizZaffar/CoHOG_Results_RAL2019).
Created on Thu June 4 20:50:20 2024

@author: mobius
"""

import cv2
import numpy as np
import os   #Imported here if your image reading methodolgy uses os.listdirectory sort of implementation.

from matplotlib import pyplot as plt
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
import csv  
import time
from scipy import linalg

from hog import initialize, extract

####################### PARAMETERS #########################
magic_width=512
magic_height=512
cell_size=16  #HOG cell-size
bin_size=8  #HOG bin size
image_frames=1  #1 for grayscale, 3 for RGB
descriptor_depth=bin_size*4*image_frames # x4 is here for block normalization due to nature of HOG
ET=0.5   # Entropy threshold, vary between 0-1.

total_Query_Images=2
query_index_offset=0
total_Ref_Images=2
ref_index_offset=0

total_no_of_regions=int((magic_width/cell_size-1)*(magic_width/cell_size-1))

#############################################################

#################### GLOBAL VARIABLES ######################

d1d2dot_matrix=np.zeros([total_no_of_regions,total_no_of_regions],dtype=np.float32)
d1d2matches_maxpooled=np.zeros([total_no_of_regions],dtype=np.float32)
d1d2matches_regionallyweighted=np.zeros([total_no_of_regions],dtype=np.float32)

matched_local_pairs=[]
ref_desc=[]

############################################################

dataset_name='GardensSmall'   #Please modify according to your needs
save_visual_matches_dir='Visual_Matches/'+dataset_name+'/'
#os.makedirs(save_visual_matches_dir)     # If the directory doesn't exist already.


query_directory='/home/mobius/Documents/datasets/GardensPointWalking/query/'       #Please modify according to your needs
ref_directory='/home/mobius/Documents/datasets/GardensPointWalking/ref/'     #Please modify according to your needs

out_directory='/home/mobius/Documents/datasets/GardensPointWalking/entropy_extracted_regions/'        # Please modify. This directory is for visualizing the entropy-based regions extraction.

def save_visual_matches(query,GT,retrieved):                     #For visualizing the correct and incorrect matches
    query_img=cv2.imread(query_directory+get_query_image_name(query))
    query_img=cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    gt_img=cv2.imread(ref_directory+get_ref_image_name(GT))
    gt_img=cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    retrieved_img=cv2.imread(ref_directory+get_ref_image_name(retrieved))
    retrieved_img=cv2.cvtColor(retrieved_img, cv2.COLOR_BGR2RGB)

    fig = plt.figure()

    ax1 = fig.add_subplot(131)  
    plt.axis('off')
    ax1.imshow(query_img)
    ax2 = fig.add_subplot(132)
    plt.axis('off')
    ax2.imshow(retrieved_img)
    ax3 = fig.add_subplot(133)
    plt.axis('off')
    ax3.imshow(gt_img)
    ax1.title.set_text('Query Image')
    ax2.title.set_text('Retrieved Image')
    ax3.title.set_text('Ground-Truth')

#    plt.show()
    
    fig.savefig(save_visual_matches_dir+str(query)+'.jpg',bbox_inches='tight')

def largest_indices_thresholded(ary):
    good_list=np.where(ary>=ET)
#    no_of_good_regions=len(good_list[0])
    print('len(good_list): ', len(good_list))
    print('len(good_list[0]): ', len(good_list[0]))
        
#    if (no_of_good_regions<min_no_of_regions):
#        good_list=largest_indices(ary,back_up_regions)
    
    return good_list 

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
      
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)
       

def get_query_image_name(j):
    k=str(j+query_index_offset)
     
    return k + '.jpg'

def get_ref_image_name(j):
    k=str(j+ref_index_offset)
    
    return k + '.jpg'

    

#@jit(nopython=False)
def conv_match_dotproduct(d1,d2,regional_gd,total_no_of_regions):            #Assumed aspect 1:1 here
     
    global d1d2dot_matrix
    global d1d2matches_maxpooled
    global d1d2matches_regionallyweighted
    global matched_local_pairs
    

    np.dot(d1,d2,out=d1d2dot_matrix)

    np.max(d1d2dot_matrix,axis=1,out=d1d2matches_maxpooled)               #Select best matched ref region for every query region

    np.multiply(d1d2matches_maxpooled,regional_gd,out=d1d2matches_regionallyweighted)   # Weighting regional matches with regional goodness

    score=np.sum(d1d2matches_regionallyweighted)/np.sum(regional_gd)    #compute final match score

    return score
#    
##############################################################################################

for ref in range(total_Ref_Images):
    print('ref start!!!')
    
    try:
        img_1 = cv2.imread(ref_directory+get_ref_image_name(ref), 0)
#        img_1 = cv2.imread(ref_directory+get_ref_image_name(ref))
    
    except (IOError, ValueError) as e:
        img_1=None
        print('Exception! \n \n \n \n',ref)        
        
    if (img_1 is not None):
        
        img_1=cv2.resize(img_1,(magic_height,magic_width))

        
        startencodetimer=time.time()
        
        height,width,angle_unit=initialize(img_1, cell_size, bin_size)
        vector_1 = extract()
    
    
#        vector_1=np.asarray(vector_1.reshape(total_no_of_regions,len(vector_1[0][0])))
        vector_1=np.asfortranarray(vector_1.transpose(),dtype=np.float32)

        print('vector_1 len :',vector_1.shape)
        print('vector_1:',vector_1)

        print('     ')
        ref_desc.append(vector_1)
#        ref_desc[:,:,ref]=vector_1
    

    
for query in range(total_Query_Images):
    print('query start!!!')
    
    confusion_vector=np.zeros(total_Ref_Images,dtype=np.float32)
    try:    
        img_2 = cv2.imread(query_directory+get_query_image_name(query), 0)
        img_2rgb=cv2.imread(query_directory+get_query_image_name(query))
        
#        img_2 = cv2.imread(query_directory+get_query_image_name(query))

    except (IOError, ValueError) as e:
        img_2=None        
        print('Exception! \n \n \n \n')    
       
    if (img_2 is not None):
        
        img_2=cv2.resize(img_2,(magic_height,magic_width))
        img_2rgb=cv2.resize(img_2rgb,(magic_height,magic_width))
        
        startencodetimer=time.time()
 
        height,width,angle_unit=initialize(img_2, cell_size, bin_size)
        vector_2 = extract()
        vector_2=np.asfortranarray(vector_2,dtype=np.float32)
        
        ################# Entropy Map ###############################
#        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_as_ubyte(img_2),(100,100))
        ent_time=time.time()
        entropy_image=cv2.resize(entropy(img_gray, disk(5)),(magic_width,magic_height))

        # # 显示图像
        # plt.imshow(entropy_image, cmap='gray')
        # plt.title('Entropy Image')
        # plt.show()
        # print('Entropy Time:',time.time()-ent_time)
        
        ################# Finding Regions #####################
        local_goodness=np.zeros([int(magic_height/cell_size-1),int(magic_width/cell_size-1)],dtype=np.float32)
        
        for a in range (int(magic_height/cell_size-1)):
            for b in range (int(magic_width/cell_size-1)):
                local_staticity=1 #Disabling staticity here, can be accommodated in future by employing YOLO etc.
                local_entropy = np.sum(entropy_image[a*cell_size:a*cell_size + 2*cell_size, b*cell_size:b*cell_size + 2*cell_size])/(8*(cell_size*4*cell_size))

                
                if (local_entropy>=ET):    
                    local_goodness[a,b]=1
                else :
                    # local_goodness[a,b]=local_entropy
                    local_goodness[a,b]=0
                    
        print('local_goodness length: ',local_goodness.shape)
        regional_goodness=local_goodness.flatten()
        regions = largest_indices_thresholded(local_goodness)
        no_of_good_regions=np.sum(regional_goodness)
        encodetime=time.time() - startencodetimer  
        print('encodetime: ',encodetime)
                
#        print(regions)
        
        img_regions_display=cv2.cvtColor(img_2rgb, cv2.COLOR_BGR2RGB)
#        
        for reg_index in range(len(regions[0])): 
            img_regions_display[regions[0][reg_index]*8 - 4, regions[1][reg_index]*8 - 4,:]=0
#             img_regions_display[regions[0][reg_index]*cell_size - cell_size/2, regions[1][reg_index]*cell_size - cell_size/2,:]=0
            cv2.rectangle(img_regions_display,(regions[1][reg_index]*cell_size,regions[0][reg_index]*cell_size),(regions[1][reg_index]*cell_size + cell_size,regions[0][reg_index]*cell_size + cell_size),(0,255,0),3)            

        # plt.imshow(img_regions_display)   # Uncomment for visualizing extracted regions
        # plt.show()
#        cv2.imwrite(out_directory+'ESSEX3IN1 Image:'+str(query+query_index_offset)+'ET:'+str(ET)+'.jpg',cv2.cvtColor(img_regions_display, cv2.COLOR_BGR2RGB))   # Uncomment for storing visual samples of extracted regions
     

        startmatchtimer=time.time()

#        confusion_vector=conv_match_tensordotproduct(vector_2,ref_desc,regional_goodness,total_no_of_regions)
        
        for ref in range(total_Ref_Images):
            score=conv_match_dotproduct(vector_2,ref_desc[ref],regional_goodness,total_no_of_regions)  
            print('result: ',score, ref)
            confusion_vector[ref]=score             
        
        matchingtime=time.time() - startmatchtimer
        print('matchingtime: ', matchingtime)
        
        row=str(query) + ',' + str(np.argmax(confusion_vector)) + ',' + str(np.amax(confusion_vector))+ ',' + str(encodetime) + ',' + str(matchingtime) + ',' + str(no_of_good_regions)
        print('row: ', row)
        
        save_visual_matches(query,query,np.argmax(confusion_vector))   # Uncomment this for saving visual samples of correctly and incorrectly matched images.

##########   Uncomment the below block to store CSV file of matching results   ############
        with open('matching_results.csv', 'a') as csvfile:
           my_writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
           row=str(query) + ',' + str(np.argmax(confusion_vector)) + ',' + str(np.amax(confusion_vector)) + ',' + str(encodetime) + ',' + str(matchingtime) + ',' + str(no_of_good_regions)
           my_writer.writerow([row])
           print(row)
           csvfile.close()
        
##########################################################################################        
 
#close_model()
