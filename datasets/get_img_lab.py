import os
path = r"/media/shan/F/XieXS/datasets/SCUT-FBP/train"  # 图片集路径

classes = ["0","1","2","3","4"]
# classes = ["0","1"]
files = os.listdir(path)
train = open(r"/media/shan/F/XieXS/datasets/SCUT-FBP/train/train.txt", 'w')
for i in classes:
    print(i)
    s = 1
    for imgname in os.listdir(os.path.join(path, i)):
        name = os.path.join(path, i) + '/' + imgname + '   ' + str(classes.index(i)) + '\n'
        # print(str(classes.index(i)))

        train.write(name)
        s += 1
train.close()
