from PIL import Image
from numpy import asarray
from operator import itemgetter
import numpy as np
import cv2
import math
from PIL import Image, ImageDraw

def openImage(image, senNum):


    grayimg = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)

    _,threshimg = cv2.threshold(grayimg, 220,255, cv2.THRESH_BINARY)
    #convert image to numpy array
    data1 = asarray(threshimg)

    #summarize data
    data1.shape
    
    data1_copy = data1.copy()
    print(data1_copy)
   
    dataMean = data1_copy.mean(axis = 1) 
    staffNum = senNum * 10

    lineMin = sorted(dataMean)[:staffNum * 2]
    print(lineMin)
    location = []
    begin = 0
    staff_ditection(image, data1_copy, lineMin, location, dataMean, begin, staffNum)

    
def staff_ditection(image, data1_copy, lineMin, location, dataMean, begin, staffNum):
    
    amount_location = len(location)
    for i in range(0, staffNum - amount_location):
        lineLocation = np.where(dataMean == lineMin[i + amount_location + begin])
        
        if(len(lineLocation[0]) != 1):
            for r in range(0, len(lineLocation[0])):
                lineNum = int(lineLocation[0][r])
                if(lineNum not in location and ((lineNum+1)not in location) and ((lineNum-1)not in location) and ((lineNum+2)not in location) and ((lineNum-2)not in location) and (lineNum not in location)):
                    location.append(lineNum)
            
        else:      
            lineNum = int(lineLocation[0][0])
            if(lineNum not in location and ((lineNum+1)not in location) and ((lineNum-1)not in location) and ((lineNum+2)not in location) and ((lineNum-2)not in location)):
                location.append(lineNum)
                
    if len(location) < staffNum:
        begin = begin + 1
        staff_ditection(image_name, data1_copy, lineMin, location, dataMean, begin, staffNum)
    
    else:
        location_sorted = sorted(location)
        print(location_sorted)
        
        distance_total = 0
        for r in range(0, int(staffNum/10)):
            distance_total += (location_sorted[10*r + 4] - location_sorted [10*r] + location_sorted[ 10*r + 9] - location_sorted [10*r+5])
    
        distance = distance_total/(0.8 * staffNum)
        print(distance)
        distance_e = distance
        distance = math.ceil(distance)
        print(distance)
        compress_sample ("sample_dot.png", distance, False, 0)
        annosize = compress_sample ("sample_dot.png", distance, False, 0)
        compress_sample ("note.png", distance, False, 0)
        compress_sample ("Sample_note2.png", distance, False, 0)
        compress_sample ("key1.png", distance, False, 1)
        compress_sample ("key3.png", distance, False, 2)
        
        barset = []
        notelocation = []
        notelocation2 = []
        notelocation3 = []
        keylocation1 = []
        keylocation2 = []
        
        note1formula = int(76552.5*distance_e*distance_e - 480240*distance_e + 1.60644e+06)
        note2formula = int(80952.2*distance_e*distance_e - 933532*distance_e + 3.57478e+06)
        note3formula = int(49347*distance_e*distance_e - 227481*distance_e + 605004)
        key1formula = int(186596*(math.pow( distance_e,1.52642 )))
        key2formula = int(16946.8*(math.pow( distance_e,2.24538)))
        
        note1condition = [70, note1formula-250000]
        note2condition = [30, note2formula-250000]
        note3condition = [10, note3formula-250000]
        key1condition = [10, key1formula+250000]
        key2condition = [10, key2formula-250000]
        
        upDownup = False
        
        for l in range(0,int(staffNum/10)):
            bar = []
            for i in range(0, int(data1_copy[0].size)):
                for r in range(location_sorted[10*l], location_sorted[10*l+9]+1):
                    if(data1_copy[r][i] == 255 ):
                        break
                    if(r == location_sorted[10*l+9]):
                        bar.append(i)

            bar_sorted = sorted(bar)
            bar_set = []

            for bar_ in bar_sorted:
                #print(bar_)
                if (bar_ - 5) not in bar_sorted and (bar_ - 4) not in bar_sorted and (bar_ - 3) not in bar_sorted and (bar_ - 2) not in bar_sorted and (bar_ - 1) not in bar_sorted:
                    bar_set.append(bar_)

            print(bar_set)

            for i in bar:
                for r in range(location_sorted[10* l], location_sorted[10*l + 9]+1):
                    data1_copy[r][i] = 255

            bar_sizee = len(bar_set) -1
            for z in range(0, bar_sizee):
                print(bar_sizee)
                
                left = bar_set[z]
                right = bar_set[z + 1]
                
                if (right - left) <5:
                    continue
                
                upper_stuff1 = location_sorted[10* l] 
                lower_stuff1 = location_sorted[10* l + 4] 

                upper_stuff2 = location_sorted[10* l + 5] 
                lower_stuff2 = location_sorted[10* l + 9] 
                
                
                upper1 = 0
                upper1 = upper_bound(upper1, l, upper_stuff1 , bar_set, data1_copy, right, left)
                print(upper1)

                lower1 = 0
                lower1 = lower_bound(lower1, l, lower_stuff1, bar_set, data1_copy, right, left)
                print(lower1)
                
                upper2 = 0
                upper2 = upper_bound(upper2, l, upper_stuff2 , bar_set, data1_copy, right, left)
                print(upper2)

                lower2 = 0
                lower2 = lower_bound(lower2, l, lower_stuff2, bar_set, data1_copy, right, left)
                print(lower2)
                

                image3 = Image.fromarray(data1_copy)
                
                crop_photo(z, image3, l, left, upper1, right, lower1,1)
                crop_photo(z, image3, l, left, upper2, right, lower2,2)
                
                convolution("ibu" + str(l)+ str(z) + "1.bmp", notelocation, left, upper1, 'result_sample_dot.png', note1condition)
                convolution("ibu" + str(l)+ str(z) + "2.bmp", notelocation, left, upper2, 'result_sample_dot.png', note1condition)
                
                convolution("ibu" + str(l)+ str(z) + "1.bmp", notelocation2, left, upper1, 'result_note.png', note2condition)
                convolution("ibu" + str(l)+ str(z) + "2.bmp", notelocation2, left, upper2, 'result_note.png', note2condition)
                
                convolution("ibu" + str(l)+ str(z) + "1.bmp", notelocation3, left, upper1, 'result_Sample_note2.png', note3condition)
                convolution("ibu" + str(l)+ str(z) + "2.bmp", notelocation3, left, upper2, 'result_Sample_note2.png', note3condition)
                
                if ( z == 0):
                    key1location = []
                    key2location = []
                    convolution("ibu" + str(l)+ str(z) + "1.bmp", key1location, left, upper1, 'result_key1.png', key1condition)
                    convolution("ibu" + str(l)+ str(z) + "2.bmp", key2location, left, upper2, 'result_key1.png', key1condition)
                    keylocation1.append(key1location)
                    keylocation2.append(key2location)
                    print(keylocation1)
                    if key1location == [] and key2location == []:
                        convolution("ibu" + str(l)+ str(z) + "1.bmp", key1location, left, upper1, 'result_key3.png', key2condition)
                        convolution("ibu" + str(l)+ str(z) + "2.bmp", key2location, left, upper2, 'result_key3.png', key2condition)
                        upDownup = True
                        keylocation1.append(key1location)
                        keylocation2.append(key2location)
                        print(keylocation1)
        
        #final_paint(notelocation,notelocation2,key1location,key2location)
        
        stuffss = []
        for r in range(0, len(location_sorted)):
            stuffss.append(location_sorted[r])
            if (r%5) != 4:
                stuffss.append (location_sorted[r] + distance_e/2)

        stuffss = sorted(stuffss)
        print(stuffss)
        
        if upDownup == False:
            heightt = 3*distance_e/2
        else:
            heightt = distance_e/2
        
        global length_key
        
        if (len(keylocation1[0]) == 0) and (len(keylocation1[0]) == 0):
            key1 = 0
            numnumkey1 = 0
            numnumkey2 = 0
            key2 = 0
            location_delete = 0
            length_key = 0
        else:
            key1_list = []
            key2_list = []
            length = []
            x_list = []
            
            for i in range(0, len(keylocation1)):
                locationkey1_num = []
                locationnum(stuffss, keylocation1[i], locationkey1_num, heightt,0)
                locationkey1_num = sorted(locationkey1_num, key=itemgetter(1)) 
                length.append(len(locationkey1_num))
                key11 = locationkey1_num[len(locationkey1_num)-1][3]
                key1_list.append(key11)
                x_list.append(locationkey1_num[len(locationkey1_num)-1][1])
                
                locationkey2_num = []
                locationnum(stuffss, keylocation2[i], locationkey2_num, heightt,0)
                locationkey2_num = sorted(locationkey2_num, key=itemgetter(1)) 
                length.append(len(locationkey2_num))
                if (len(locationkey2_num)== 0):
                    continue
                else:
                    key22 = locationkey2_num[len(locationkey2_num)-1][3]
                    key2_list.append(key22)
                    x_list.append(locationkey2_num[len(locationkey2_num)-1][1])
                
            key1 = max(key1_list, key = key1_list.count)
            key2 = max(key2_list,key=key2_list.count)
            location_delete = max(x_list,key=x_list.count)
            print(x_list)
            print(key1)
            print(key2)
            print(location_delete)
            length_key = max(length, key = length.count)

        if upDownup == False:
            numnumkey1 = 7
            if length_key == 5:
                numnumkey1 = 0
        else:
            numnumkey1 = 4
            if(length_key == 5):
                numnumkey1 = -3      
        
        if upDownup == False:
            numnumkey2 = -7
            if length_key == 3:
                numnumkey2 = 0
        else:
            numnumkey2 = -10
            if length_key == 5 or length_key == 7:
                numnumkey2 = -3

        locationnote1_num = []
        locationnum(stuffss, notelocation, locationnote1_num, distance_e/2,location_delete)
        print(locationnote1_num)

        locationnote2_num = []
        locationnum(stuffss, notelocation2, locationnote2_num,distance_e/2, location_delete)
        print(locationnote2_num)
        
        locationnote3_num = []
        locationnum(stuffss, notelocation3, locationnote3_num,distance_e/2, location_delete)
        print(locationnote3_num)

        locationnote1_final = []
        locationnote1_fina2 = []
        locationnote1_fina3 = []

        locationNum2(locationnote1_num,key1, key2, distance_e, stuffss,locationnote1_final, numnumkey1, numnumkey2)
        locationNum2(locationnote2_num,key1, key2, distance_e, stuffss,locationnote1_fina2, numnumkey1, numnumkey2)
        locationNum2(locationnote3_num,key1, key2, distance_e, stuffss,locationnote1_fina3, numnumkey1, numnumkey2)

        print(locationnote1_final)
        print(locationnote1_fina2)

        locationnote_final = locationnote1_final + locationnote1_fina2 + locationnote1_fina3
        imageannotation( locationnote_final, image, annosize, distance)
    
    
def imageannotation(locationnote_final, image, annosize, distance):
   
  
    # copying image to another image object 
    ima2 = image.copy() 
    ima2.save("Copy23.png")
    
    
    for i in range(0, len(locationnote_final)):
        
        im1 = Image.open("Copy23.png")
        im2 = Image.open('n' + str(locationnote_final[i][2]) + '.png').convert("RGBA")
        w, h = im2.size
        compress_rate = distance/13
        im3 = im2.resize((int(w*compress_rate), int(h*compress_rate)))

        back_im = im1.copy()
        draw = ImageDraw.Draw(back_im)
        draw.rectangle((locationnote_final[i][1], locationnote_final[i][0], locationnote_final[i][1] + annosize[1], locationnote_final[i][0] + annosize[0]), outline= 170)
        
        (x, y) = (locationnote_final[i][1] + 1, locationnote_final[i][0] +1)
        back_im.paste(im3, (x, y),  mask=im3)
        back_im.save("Copy23.png", quality=95)
    

def locationNum2(locationnote_num,key1, key2, distance, location,locationnote1_final, numnumkey1, numnumkey2):
    for i in range(0, len(locationnote_num)):
        if locationnote_num[i][2] > 8:
            if locationnote_num[i][4] == 0:
                final_num = numnumkey2-(locationnote_num[i][3] - key2)
                locationnote1_final.append([locationnote_num[i][0], locationnote_num[i][1], final_num])
            if locationnote_num[i][4] == 1:
                if locationnote_num[i][3] == 8:
                    stuffsplocation = location[18* locationnote_num[i][5] + 17]
                    print(18* locationnote_num[i][5] + 17)
                    print(stuffsplocation)
                    print(locationnote_num[i][0] + 10 - stuffsplocation)
                    final_num1 = int((locationnote_num[i][0] + 10 - stuffsplocation)*2/distance)
                    final_num =  numnumkey2-(locationnote_num[i][3] - key2 + final_num1)
                    locationnote1_final.append([locationnote_num[i][0], locationnote_num[i][1], final_num])
                if locationnote_num[i][3] == 0:
                    stuffsplocation = location[18* locationnote_num[i][5] + 9]
                    final_num1 = int((locationnote_num[i][0] + 10 - stuffsplocation)*2/distance)
                    final_num =  numnumkey2-(locationnote_num[i][3] - key2 + final_num1)
                    locationnote1_final.append([locationnote_num[i][0], locationnote_num[i][1], final_num])
        
        if locationnote_num[i][2] <= 8:
            if locationnote_num[i][4] == 0:
                final_num = numnumkey1-(locationnote_num[i][3] - key1)
                locationnote1_final.append([locationnote_num[i][0], locationnote_num[i][1], final_num])
            if locationnote_num[i][4] == 1:
                if locationnote_num[i][3] == 8:
                    stuffsplocation = location[18* locationnote_num[i][5] + 8]
                    final_num1 = int((locationnote_num[i][0] + 10 - stuffsplocation)*2/distance)
                    final_num =  numnumkey1-(locationnote_num[i][3] - key1 + final_num1)
                    locationnote1_final.append([locationnote_num[i][0], locationnote_num[i][1], final_num])
                if locationnote_num[i][3] == 0:
                    stuffsplocation = location[18* locationnote_num[i][5]]
                    final_num1 = int((locationnote_num[i][0] + 10 - stuffsplocation)*2/distance)
                    final_num =  numnumkey1 -(locationnote_num[i][3] - key1 + final_num1)
                    locationnote1_final.append([locationnote_num[i][0], locationnote_num[i][1], final_num])


def locationnum(stuffss, location_initial,location_num, height, location_delete):
    print(location_delete)
    
    for i in range(0,len(location_initial)):
        inline = 0
        #locationss =[(location_initial[i][0] + location_initial[i][3] + 10), (location_initial[i][1] + location_initial[i][2] + 11.5)]
        
        nearest = find_nearest(stuffss, (location_initial[i][0] + location_initial[i][3] + height))
        if ((location_initial[i][0] + location_initial[i][3] + height) - nearest > (height*2)) or ((location_initial[i][0] + location_initial[i][3] + height) - nearest < -(height*2)):
            inline = 1
        #print(nearest)
        location_stuffnum = stuffss.index(nearest)
        updown = location_stuffnum%18
        updownlist = int(location_stuffnum//18)
        location_stuffnums = location_stuffnum%9
        
        #print(location_stuffnum)
        if (location_initial[i][1] + location_initial[i][2]) > location_delete:
            location_num.append([(location_initial[i][0] + location_initial[i][3]), (location_initial[i][1] + location_initial[i][2]),updown ,location_stuffnums, inline, updownlist])

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
        


def final_paint(notelocation,notelocation2,key1location,key2location):
    image = cv2.imread('1.png')

    num = len(notelocation)

    for i in range(0, num):
        cv2.rectangle(image,(notelocation[i][1] + notelocation[i][2],notelocation[i][0] + notelocation[i][3]),(notelocation[i][1] + notelocation[i][2] + 23,notelocation[i][0] + notelocation[i][3] + 20), (0, 0, 255), 2) 
    
    
    num = len(notelocation2)

    for i in range(0, num):
        cv2.rectangle(image,(notelocation2[i][1] + notelocation2[i][2],notelocation2[i][0] + notelocation2[i][3]),(notelocation2[i][1] + notelocation2[i][2] + 26,notelocation2[i][0] + notelocation2[i][3] + 20), (0, 255, 0), 2) 
        
    num = len(key1location)

    for i in range(0, num):
        cv2.rectangle(image,(key1location[i][1] + key1location[i][2],key1location[i][0] + key1location[i][3]),(key1location[i][1] + key1location[i][2] + 22,key1location[i][0] + key1location[i][3] + 60), (255, 0, 0), 2)  

    
    cv2.imwrite('xixixixixi2.png', image)
    
                
def convolution(image_name, notelocation, left, upper, sample_name, condition):
    
    image_sample = Image.open(sample_name).convert('RGB') 
    data_sample = asarray(image_sample).astype(int)
    
    height_sample = len(data_sample)
    width_sample = len(data_sample[0])
    print("h" + str(height_sample))
    print(width_sample)

    #upload test image
    image = Image.open(image_name).convert('RGB') 
    data = asarray(image)
    height = len(data)
    width = len(data[0])
    print(height)
    print(width)

    myScore = [[0 for x in range(width - width_sample)] for y in range(height - height_sample)]

    for i in range(0, height - height_sample):
        for r in range(0, width - width_sample):
            cropped = image.crop((r, i, r + width_sample, i + height_sample)) # (left, upper, right, lower)
            data_test = asarray(cropped).astype(int)
            myScore[i][r] = sum(np.multiply((128 - data_test),(128 - data_sample)).flatten())
    
    X = np.array(myScore)

    num_sum = condition[0]
    lineMin = sorted(X.flatten())[-num_sum:]
    print(lineMin)

    note_location = []
    less_location = []

    for i in range(0, num_sum):
        if(lineMin[num_sum-i-1] < condition[1]):
            break
        solutions = np.argwhere(X == lineMin[num_sum-i-1])
        if ([solutions[0][0], solutions[0][1]] not in note_location) and([solutions[0][0]-1, solutions[0][1] - 1] not in note_location) and([solutions[0][0], solutions[0][1] - 1] not in note_location)and ([solutions[0][0] + 1, solutions[0][1] - 1] not in note_location)and ([solutions[0][0]-1, solutions[0][1]] not in note_location) and ([solutions[0][0] + 1, solutions[0][1]] not in note_location)and ([solutions[0][0]-1, solutions[0][1] + 1] not in note_location) and ([solutions[0][0], solutions[0][1] + 1] not in note_location)and ([solutions[0][0] + 1, solutions[0][1] +1] not in note_location):
            less_location.append([solutions[0][0], solutions[0][1]])
            notelocation.append([solutions[0][0], solutions[0][1], left, upper]) #(upper, left, left, upper)
            #print(solutions)
            #print(i)
        note_location.append([solutions[0][0], solutions[0][1]])

    print(less_location)
    
    #print(myScore[88][327])

    num = len(less_location)

                
def compress_sample (sample_name,distance, show, key):
    img_sample = Image.open(sample_name)


    w, h = img_sample.size
    t = h
    if key == 1:
        t = 13
    elif key == 2:
        t = 33
    compress_rate = distance/t

    img_resize = img_sample.resize((int(w*compress_rate)+1, int(h*compress_rate)))
    
    resize_w, resize_h = img_resize.size
    print(resize_w)
    print(resize_h)
    img_resize.save('result_'+ str(sample_name))
    if show:
        img_resize.show()  # 在照片应用中打开图片
    return(resize_w,resize_h)


def crop_photo(z, image3, l, left, upper, right, lower, k):
    cropped = image3.crop((left, upper, right, lower)) # (left, upper, right, lower)

    #cropped.show()
    if cropped.mode == "F":
        cropped = cropped.convert('RGB')
    image_name = "ibu" + str(l)+ str(z) + str(k) + ".bmp"
    cropped.save(image_name)
    

def upper_bound(upper, l, upper_stuff, bar_set, data1_copy, right, left):
        for r in range(0, upper_stuff):
            bound = upper_stuff - r
            for i in range(0, right - left):
                if(data1_copy[bound][i + left] < 250 ):
                    break
                if(i == (right - left -1)):
                    upper = bound
                    return upper

def lower_bound(lower, l, lower_stuff, bar_set, data1_copy, right, left):
        for r in range(0, len(data1_copy) - lower_stuff):
            bound = r + lower_stuff
            for i in range(0, right - left):
                if(data1_copy[bound][i + left] < 250 ):
                    break
                if(i == (right - left -1)):
                    lower = bound
                    forlocation = r
                    return lower

