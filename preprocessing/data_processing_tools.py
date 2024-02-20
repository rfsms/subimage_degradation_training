

from PIL import Image
import sys
import cv2
import math
import os
import shutil




def split_then_img2block(src_path,output_path,delete_tmp_folder_flag=False):

    tmp_folder = "original"
    if os.path.isdir(src_path) and os.path.isdir(output_path):

        imagenames = os.listdir(src_path)
        if len(imagenames) > 0:
            for imagename in imagenames:
                if ".gif" in imagename:
                    file_name = imagename.split(".")[0]
                    if tmp_folder not in os.listdir(output_path):
                        os.mkdir(output_path+tmp_folder)
                    processImage(src_path + imagename,output_path+tmp_folder+"/")

                    img2blocks(output_path,file_name + '.png',tmp_folder)
                else:
                    print(imagename, " Not .gif image")
        else:
            print("No files in the source folder")

    else:
        if not os.path.isdir(src_path):
            print("Source folder doesn.t exist check: ",src_path)
        if not os.path.isdir(output_path):
            print("output folder doesn.t exist check: ",output_path)

    if delete_tmp_folder_flag:
        shutil.rmtree(output_path + tmp_folder)


def img2blocks(out_path,img_name,tmp_folder,w=224,h=224):
    img = cv2.imread(out_path +tmp_folder+"/" + img_name)
    if img_name.split(".")[0] not in os.listdir(out_path):
        os.mkdir(out_path + img_name.split(".")[0])
    shutil.copy(out_path +tmp_folder+"/" + img_name,out_path + img_name.split(".")[0]+"/"+img_name)
    out_path = out_path + img_name.split(".")[0] + "/"
    h_count , w_count = math.floor(img.shape[0]/h) , math.floor(img.shape[1]/w)

    for i_h in range(h_count):
        for i_w in range(w_count):
            window = img[i_h*h:(i_h+1)*h,i_w*w:(i_w+1)*w]
            im = Image.fromarray(window)
            im.save(out_path + img_name.split(".")[0] + "_" + str(i_h)+ "_" + str(i_w)  + '.png')

        window = img[i_h*h:(i_h+1)*h,img.shape[1] - w:img.shape[1]]
        im = Image.fromarray(window)
        out = im.transpose(Image.FLIP_LEFT_RIGHT)
        out.save(out_path + img_name.split(".")[0] + "_" + str(i_h)+ "_" + str(i_w+1) + '.png')

    for i_w in range(w_count):
        window = img[img.shape[0] - h:img.shape[0], i_w * w:(i_w + 1) * w]
        im = Image.fromarray(window)
        im.save(out_path + img_name.split(".")[0] + "_" + str(i_h+1) + "_" + str(i_w) + '.png')

    window = img[img.shape[0] - h:img.shape[0], img.shape[1] - w:img.shape[1]]
    im = Image.fromarray(window)
    out = im.transpose(Image.FLIP_LEFT_RIGHT)
    out.save(out_path + img_name.split(".")[0] + "_" + str(i_h+1) + "_" + str(i_w + 1) + '.png')





def processImage(infile,output_path,multiple_frame_flag = False):
    file_name = infile.split("/")[-1].split(".")[0]
    try:
        im = Image.open(infile)
    except IOError:
        print("Cant load", infile)
        sys.exit(1)

    i = 0
    mypalette = im.getpalette()

    if multiple_frame_flag:
        try:
            while 1:
                try:
                    im.putpalette(mypalette)
                    new_im = Image.new("RGBA", im.size)
                    new_im.paste(im)
                    new_im.save(output_path + file_name + "_" +str(i)+'.png')

                    i += 1
                    im.seek(im.tell() + 1)
                except:
                    im.save(output_path + file_name +'.png')
                    i += 1
                    im.seek(im.tell() + 1)

        except EOFError:
            pass # end of sequence

    else:
        if mypalette:
            im.putpalette(mypalette)
        new_im = Image.new("RGBA", im.size)
        new_im.paste(im)
        new_im.save(output_path + file_name + '.png')





