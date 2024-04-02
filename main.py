import time
import sys
import os
from BroadLearningSystem import ER_BLSNet, ER_BLS_AddEnhanceNodes, ER_BLS_AddFeatureEnhanceNodes, ER_BLS_AddNewData

def make_print_to_file(path='./'):
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass
    fileName = "ER-BLS_SCUT-FBP5500"
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))

make_print_to_file(path='./results')

if __name__ == '__main__':
    
    # 获取数据
    # t1=time.time()

    # filePath_train = r'/media/xie/F/XieXS/datasets/LSAFBD/train.txt'
    # filePath_test = r'/media/xie/F/XieXS/datasets/LSAFBD/test.txt'

    # filePath_train = r'/media/xie/F/XieXS/datasets/CelebA/train.txt'
    # filePath_test = r'/media/xie/F/XieXS/datasets/CelebA/test.txt'

    # filePath_train = r'/media/xie/F/XieXS/datasets/SCUT-FBP/train.txt'
    # filePath_test = r'/media/xie/F/XieXS/datasets/SCUT-FBP/test.txt'

    filePath_train = r'/media/xie/F/XieXS/datasets/SCUT-FBP5500/train.txt'
    filePath_test = r'/media/xie/F/XieXS/datasets/SCUT-FBP5500/test.txt'

    # LSAFBD参数设置c15,62.124
    # N1 = 25  #  # of nodes belong to each window
    # N2 = 72  #  # of windows -------Feature mapping layer
    # N3 = 3088 #  # of enhancement nodes -----Enhance layer
    # s = 0.14  #  shrink coefficient
    # C = 2**-15 # Regularization coefficient

    # N1 = 25  #  # of nodes belong to each window
    
    # N2 = 60
    # N2 = 64
    # N2 = 68
    # N2 = 72  
    # N2 = 74 
    # N2 = 76

    # N3 = 1588
    # N3 = 2088
    # N3 = 2588
    # N3 = 3088 
    # N3 = 3588
    # N3 = 4088

    # s = 0.14  #  shrink coefficient
    # C = 2**-15

    # L = 5
    # M1 = 20
    # M2 = 50
    # M3 = 250

    # SCUT-FBP5500参数设置74.68
    N1 = 12
    N2 = 54
    N3 = 2966
    s = 0.42
    C = 2**-10

    # adding new data
    # N1 = 10
    # N2 = 140
    # N3 = 2372
    # s = 0.07
    # C = 2**-30

    # l = 5
    # m = 2000

    # N1 = 25
    # N2 = 64
    # N3 = 2036
    # s = 0.56
    # C = 2**-30

    t1=time.time()

    for i in range(1):
        print("*****************start*******************")
        print('-------------------ER-BLS---------------------------')
        ER_BLSNet(filePath_train, filePath_test, s, C, N1, N2, N3)
        # print('-------------------BLS_AddEnhanceNodes------------------------')
        # ER_BLS_AddEnhanceNodes(filePath_train, filePath_test, s, C, N1, N2, N3, L, M1)
        # print('-------------------BLS_AddFeatureEnhanceNodes----------------')
        # ER_BLS_AddFeatureEnhanceNodes(filePath_train, filePath_test, s, C, N1, N2, N3, L, M1, M2, M3)
        # print('-------------------BLS_INPUT--------------------------')
        # ER_BLS_AddNewData(filePath_train, filePath_test,s,C,N1,N2,N3,l,m)
        print("*****************end*******************")   

    t2=time.time()
    traintime=t2-t1
    print("all_time is:",traintime)