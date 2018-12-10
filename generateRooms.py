from vapory import *

import numpy as np
import scipy.io as sp

import matplotlib.image as mpimg
import os

numImages = 100000
offset = 0;

# light source
room_topwall = np.random.rand(numImages)*(7-2) + 2;
room_bottomwall = -(np.random.rand(numImages)*(7-2) + 2);
room_leftwall = -(np.random.rand(numImages)*(11-5) + 5);
room_ceiling = np.random.rand(numImages)*(3.5-2) + 2;

light_x = np.random.rand(numImages)*room_leftwall+0.075;
light_y = room_ceiling-0.125;
light_z = np.random.rand(numImages)*(room_topwall-room_bottomwall-0.125) + room_bottomwall+0.075;

cyl_radius = 0.22;
cyl_height = 1.8;

cyl_num =  np.random.randint(4, size=numImages);

cyl_x_1 = np.random.rand(numImages)*(room_leftwall+cyl_radius +2)-2;
cyl_z_1 = np.random.rand(numImages)*(room_topwall- 2*cyl_radius -room_bottomwall) + cyl_radius + room_bottomwall;
cyl_r_1 = np.random.rand(numImages);
cyl_g_1 = np.random.rand(numImages);
cyl_b_1 = np.random.rand(numImages);

cyl_x_2 = np.random.rand(numImages)*(room_leftwall+cyl_radius +2)-2;
cyl_z_2 = np.random.rand(numImages)*(room_topwall- 2*cyl_radius -room_bottomwall) + cyl_radius + room_bottomwall;
cyl_r_2 = np.random.rand(numImages);
cyl_g_2 = np.random.rand(numImages);
cyl_b_2 = np.random.rand(numImages);

cyl_x_3 = np.random.rand(numImages)*(room_leftwall+cyl_radius +2)-2;
cyl_z_3 = np.random.rand(numImages)*(room_topwall- 2*cyl_radius -room_bottomwall) + cyl_radius + room_bottomwall;
cyl_r_3 = np.random.rand(numImages);
cyl_g_3 = np.random.rand(numImages);
cyl_b_3 = np.random.rand(numImages);


cyl_x_1[np.where(cyl_num < 1)[0]] = 0;
cyl_z_1[np.where(cyl_num < 1)[0]] = 0;
cyl_r_1[np.where(cyl_num < 1)[0]] = 0;
cyl_g_1[np.where(cyl_num < 1)[0]] = 0;
cyl_b_1[np.where(cyl_num < 1)[0]] = 0;

cyl_x_2[np.where(cyl_num < 2)[0]] = 0;
cyl_z_2[np.where(cyl_num < 2)[0]] = 0;
cyl_r_2[np.where(cyl_num < 2)[0]] = 0;
cyl_g_2[np.where(cyl_num < 2)[0]] = 0;
cyl_b_2[np.where(cyl_num < 2)[0]] = 0;

cyl_x_3[np.where(cyl_num < 3)[0]] = 0;
cyl_z_3[np.where(cyl_num < 3)[0]] = 0;
cyl_r_3[np.where(cyl_num < 3)[0]] = 0;
cyl_g_3[np.where(cyl_num < 3)[0]] = 0;
cyl_b_3[np.where(cyl_num < 3)[0]] = 0;


room_topwall = room_topwall.tolist();
room_bottomwall = room_bottomwall.tolist();
room_leftwall = room_leftwall.tolist();
room_ceiling = room_ceiling.tolist();

light_x = light_x.tolist();
light_y = light_y.tolist();
light_z = light_z.tolist();

cyl_num = cyl_num.tolist();
cyl_x_1 = cyl_x_1.tolist();
cyl_x_2 = cyl_x_2.tolist();
cyl_x_3 = cyl_x_3.tolist();
cyl_z_1 = cyl_z_1.tolist();
cyl_z_2 = cyl_z_2.tolist();
cyl_z_3 = cyl_z_3.tolist();



cameraLoc = [0.0, 0.00000001, 0.0]
cameraLook = [-1.0, 0.0, 0.0]
cameraAng = 180 # degrees

dr = "C:/Users/cs13/Google Drive/Deep learning/DataGen/Test1/"

imgPaths = [str(x) for x in np.arange(0+offset, numImages+offset)]
imgPaths = [s + '.png' for s in imgPaths]
matPaths = dr+'data_2.mat';

image_array=[]
room_topwall_array=[]
room_bottomwall_array=[]
room_leftwall_array=[]
room_ceiling_array=[]
light_pos_array=[]
cyl_num_array=[]
cyl1_pos_array=[]
cyl2_pos_array=[]
cyl3_pos_array=[]
cyl1_col_array=[]
cyl2_col_array=[]
cyl3_col_array=[]

for ii in range(numImages):
    print(ii)
    camera = Camera('panoramic',''
                    'angle',    cameraAng,
                     'location', cameraLoc,
                     'look_at',  cameraLook)
    
    light = LightSource([light_x[ii], light_y[ii], light_z[ii]], 'color', [1,1,1])

    ceiling = Plane([0, 1, 0], room_ceiling[ii],
               Texture(Pigment('color', [1,1,1])),
               Finish('diffuse', 1))

    
    wallTop = Plane([0, 0, 1], room_topwall[ii],
               Texture(Pigment('color', [1,1,1])),
               Finish('diffuse', 1))
    
    
    wallLeft = Plane([1, 0, 0], room_leftwall[ii],
               Texture(Pigment('color', [1,1,1])),
               Finish('diffuse', 1))
    
    
    wallBottom = Plane([0, 0, 1], room_bottomwall[ii],
               Texture(Pigment('color', [1,1,1])),
               Finish('diffuse', 1))

    if cyl_num[ii] == 0:
        allObjects = [ceiling, light, wallTop, wallBottom, wallLeft]
        
    if cyl_num[ii] == 1:
        cyl = Cylinder([cyl_x_1[ii],0,cyl_z_1[ii]],[cyl_x_1[ii],cyl_height,cyl_z_1[ii]],cyl_radius,
               Texture( Pigment( 'color', [cyl_r_1[ii],cyl_g_1[ii],cyl_b_1[ii]]),
               Finish('diffuse', 1)))
        allObjects = [ceiling, light, wallTop, wallBottom, wallLeft,cyl]
                
    if cyl_num[ii] == 2:
        cyl = Cylinder([cyl_x_1[ii],0,cyl_z_1[ii]],[cyl_x_1[ii],cyl_height,cyl_z_1[ii]],cyl_radius,
               Texture( Pigment( 'color', [cyl_r_1[ii],cyl_g_1[ii],cyl_b_1[ii]]),
               Finish('diffuse', 1)))
        cyl2 = Cylinder([cyl_x_2[ii],0,cyl_z_2[ii]],[cyl_x_2[ii],cyl_height,cyl_z_2[ii]],cyl_radius,
               Texture( Pigment( 'color', [cyl_r_2[ii],cyl_g_2[ii],cyl_b_2[ii]]),
               Finish('diffuse', 1)))
        allObjects = [ceiling, light, wallTop, wallBottom, wallLeft,cyl,cyl2]
        
        
    if cyl_num[ii] == 3:
        cyl = Cylinder([cyl_x_1[ii],0,cyl_z_1[ii]],[cyl_x_1[ii],cyl_height,cyl_z_1[ii]],cyl_radius,
               Texture( Pigment( 'color', [cyl_r_1[ii],cyl_g_1[ii],cyl_b_1[ii]]),
               Finish('diffuse', 1)))
        cyl2 = Cylinder([cyl_x_2[ii],0,cyl_z_2[ii]],[cyl_x_2[ii],cyl_height,cyl_z_2[ii]],cyl_radius,
               Texture( Pigment( 'color', [cyl_r_2[ii],cyl_g_2[ii],cyl_b_2[ii]]),
               Finish('diffuse', 1)))
        cyl3 = Cylinder([cyl_x_3[ii],0,cyl_z_3[ii]],[cyl_x_3[ii],cyl_height,cyl_z_3[ii]],cyl_radius,
               Texture( Pigment( 'color', [cyl_r_3[ii],cyl_g_3[ii],cyl_b_3[ii]]),
               Finish('diffuse', 1)))        
        allObjects = [ceiling, light, wallTop, wallBottom, wallLeft,cyl,cyl2,cyl3]
        
        
    scene = Scene(camera, allObjects, [], included=['colors.inc', 'textures.inc'])
    
    try:
        scene.render(imgPaths[ii], width=360*2, height=360, antialiasing=0.05)
    
        img = mpimg.imread(imgPaths[ii])
        os.remove(imgPaths[ii])
        
        image_array.append(np.mean(img[1:int(np.size(img,0)/2),:],0))
        room_topwall_array.append(room_topwall[ii])
        room_bottomwall_array.append(room_bottomwall[ii])
        room_leftwall_array.append(room_leftwall[ii])
        room_ceiling_array.append(room_ceiling[ii])
        light_pos_array.append([light_x[ii],light_y[ii],light_z[ii]])
        cyl_num_array.append(cyl_num[ii])
        cyl1_pos_array.append([cyl_x_1[ii], cyl_z_1[ii]])
        cyl2_pos_array.append([cyl_x_2[ii], cyl_z_2[ii]])
        cyl3_pos_array.append([cyl_x_3[ii], cyl_z_3[ii]])
        cyl1_col_array.append([cyl_r_1[ii], cyl_g_1[ii], cyl_b_1[ii]])
        cyl2_col_array.append([cyl_r_2[ii], cyl_g_2[ii], cyl_b_2[ii]])
        cyl3_col_array.append([cyl_r_3[ii], cyl_g_3[ii], cyl_b_3[ii]])
    except:
        print('Error')

sv = {'image':image_array,
      'room_topwall':room_topwall_array,
     'room_bottomwall':room_bottomwall_array,
     'room_leftwall':room_leftwall_array,
     'room_ceiling':room_ceiling_array,
     'light_pos':light_pos_array,
     'cyl_num':cyl_num_array,
     'cyl1_pos': cyl1_pos_array,
     'cyl2_pos':cyl2_pos_array,
     'cyl3_pos':cyl3_pos_array,
     'cyl1_col':cyl1_col_array,
     'cyl2_col':cyl2_col_array,
      'cyl3_col':cyl3_col_array
}
sp.savemat(matPaths, sv,do_compression=True)

print('Done!')