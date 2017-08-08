#coding=utf-8
import os
import shutil
from BeautifulSoup import BeautifulSoup
#train.txt可通过运行脚本caffe/data/get_ilsvrc_aux.sh下载获得
'''with open("../imagenet/train.txt") as f:
    with open("../imagenet/darknet_train.txt",'w') as w:
        for l in f.readlines():
            w.writelines('/home/research/disk1/imagenet/ILSVRC2015/Data/CLS-LOC/train/'+l.split()[0]+'\n')'''


#val
'''dataroot='/home/research/disk1/imagenet/ILSVRC2015/'
vallabel=dataroot+'Annotations/CLS-LOC/val'
valimage=dataroot+'Data/CLS-LOC/val'
with open("../imagenet/darknet_val.txt",'w') as w:
    for l in os.listdir(vallabel):

        xml = ""
        with open(os.path.join(vallabel,l)) as f:
            xml = f.readlines()
        xml = ''.join([line.strip('\t') for line in xml])

        label=BeautifulSoup(xml).find('name').string
        filename=BeautifulSoup(xml).find('filename').string+'.JPEG'

        saveroot='../temp/'+label
        if os.path.exists(saveroot) is False:
            os.makedirs(saveroot)
        shutil.copy(os.path.join(valimage,filename),os.path.join(saveroot,filename))
        w.writelines('/home/research/disk1/compress_yolo/temp/' + filename+ '\n')'''


with open("../imagenet/darknet_val.txt",'w') as w:
    root='/home/research/disk1/compress_yolo/temp/'
    classify_temp = os.listdir(root)
    classify_file = []
    for c in classify_temp:
        classify_file.append(os.path.join(root, c))
    for f in classify_file:
        for cf in os.listdir(f):
            w.writelines(os.path.join(f,cf) + '\n')














