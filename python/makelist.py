import  os
images=os.listdir("../vs2015/vs2015/cifar/test")
with open('test.list','w') as f:
	for image in images:
		path="cifar/test/"+image+'\n'
		f.writelines(path)
