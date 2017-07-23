import  os
images=os.listdir("../data/cifar/test")
with open('test.list','w') as f:
	for image in images:
		path="data/cifar/test/"+image+'\n'
		f.writelines(path)
