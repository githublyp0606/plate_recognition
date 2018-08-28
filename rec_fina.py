#!python3
#pylint: disable=R, W0401, W0614, W0703

#!python3
#pylint: disable=R, W0401, W0614, W0703

from ctypes import *
import random
import os
import configparser
import cv2
import pandas as pd
import numpy as np

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

class Darknet:
    def __init__(self, metaPath, configPath, weightPath, hasGPU=False):
        lib = CDLL("yolo_cpp_nogpu_dll.dll", RTLD_GLOBAL)
        lib.network_width.argtypes = [c_void_p]
        lib.network_width.restype = c_int
        lib.network_height.argtypes = [c_void_p]
        lib.network_height.restype = c_int

        predict = lib.network_predict
        predict.argtypes = [c_void_p, POINTER(c_float)]
        predict.restype = POINTER(c_float)

        if hasGPU:
            print("GPU using")
            set_gpu = lib.cuda_set_device
            set_gpu.argtypes = [c_int]
        else:
            print("Don't use Gpu.")
        make_image = lib.make_image
        make_image.argtypes = [c_int, c_int, c_int]
        make_image.restype = IMAGE

        self.get_network_boxes = lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        make_network_boxes = lib.make_network_boxes
        make_network_boxes.argtypes = [c_void_p]
        make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        free_ptrs = lib.free_ptrs
        free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        network_predict = lib.network_predict
        network_predict.argtypes = [c_void_p, POINTER(c_float)]

        reset_rnn = lib.reset_rnn
        reset_rnn.argtypes = [c_void_p]

        load_net = lib.load_network
        load_net.argtypes = [c_char_p, c_char_p, c_int]
        load_net.restype = c_void_p

        load_net_custom = lib.load_network_custom
        load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        load_net_custom.restype = c_void_p

        do_nms_obj = lib.do_nms_obj
        do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = lib.free_image
        self.free_image.argtypes = [IMAGE]

        letterbox_image = lib.letterbox_image
        letterbox_image.argtypes = [IMAGE, c_int, c_int]
        letterbox_image.restype = IMAGE

        load_meta = lib.get_metadata
        lib.get_metadata.argtypes = [c_char_p]
        lib.get_metadata.restype = METADATA

        self.load_image = lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        rgbgr_image = lib.rgbgr_image
        rgbgr_image.argtypes = [IMAGE]

        self.predict_image = lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.meta = load_meta(metaPath.encode("ascii"))
        self.net  = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1

        config = configparser.RawConfigParser()
        config.read(metaPath)

        self.altNames = []

        with open(config['name']['names'], 'r') as file:
            self.altNames = file.read().splitlines()

    def __sample(self, probs):
        s = sum(probs)
        probs = [a/s for a in probs]
        r = random.uniform(0, 1)
        for i in range(len(probs)):
            r = r - probs[i]
            if r <= 0:
                return i
        return len(probs)-1

    def __c_array(self, ctype, values):
        arr = (ctype*len(values))()
        arr[:] = values
        return arr

    def array_to_image(self, arr):
        import numpy as np
        # need to return old values to avoid python freeing memory
        arr = arr.transpose(2,0,1)
        c = arr.shape[0]
        h = arr.shape[1]
        w = arr.shape[2]
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        data = arr.ctypes.data_as(POINTER(c_float))
        im = IMAGE(w,h,c,data)
        return im, arr

    def classify(self, net, meta, im):
        out = self.predict_image(net, im)
        res = []
        for i in range(meta.classes):
            res.append((altNames[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def detect(self, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
        if type(image).__name__ == 'str':
            im = self.load_image(image, 0, 0)
        else:
            im = image

        if debug: print("Loaded image")
        num = c_int(0)
        if debug: print("Assigned num")
        pnum = pointer(num)
        if debug: print("Assigned pnum")
        self.predict_image(self.net, im)
        if debug: print("did prediction")
        dets = self.get_network_boxes(self.net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
        if debug: print("Got dets")
        num = pnum[0]
        if debug: print("got zeroth index of pnum")
        if nms:
            self.do_nms_sort(dets, num, self.meta.classes, nms)
        if debug: print("did sort")
        res = []
        if debug: print("about to range")
        for j in range(num):
            if debug: print("Ranging on "+str(j)+" of "+str(num))
            if debug: print("Classes: "+str(self.meta), self.meta.classes, self.meta.names)
            for i in range(self.meta.classes):
                if debug: print("Class-ranging on "+str(i)+" of "+str(self.meta.classes)+"= "+str(dets[j].prob[i]))
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    nameTag = self.altNames[i]
                    if debug:
                        print("Got bbox", b)
                        print(nameTag)
                        print(dets[j].prob[i])
                        print((b.x, b.y, b.w, b.h))
                    res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        if debug: print("did range")
        res = sorted(res, key=lambda x: -x[1])
        if debug: print("did sort")
        # self.free_image(im)
        if debug: print("freed image")
        self.free_detections(dets, num)
        if debug: print("freed detections")

        return res

class CharsCroped:
    # def __init__(self, cv_image):
    def __init__(self,image_croped):
    #     path = "/Users/Quantum/Desktop/"
    #     file = "青A11812"
    #     self.image = cv2.imread(path + "青A11812.jpg")
        self.result = []
        self.result_double = []
        self.image = image_croped
        # path =    "/Users/Quantum/Desktop/"
        # file = "藏A9BB28"

    def Croped(self):
        image = self.image
        gray = cv2.medianBlur(image, 3)
        # cv2.imwrite("/Users/Quantum/Desktop/"+"medianBlur.jpg", gray)
        gray1 = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("/Users/Quantum/Desktop/"+"Grayscale.jpg", gray)
        ret2, gray = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret1, gray_or = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # cv2.imwrite("/Users/Quantum/Desktop/gray.jpg", gray)

        # 图片大小获取
        sp = image.shape
        # print("维度" + str(sp))
        rows = sp[0]  # height(rows) of image
        colums = sp[1]  # width(colums) of image

        #  黑白字体判断
        black_num = 0
        white_num = 0
        tag = 0
        for y in range(colums):
            num = 0
            for x in range(rows):
                if (gray[x, y] == 255):
                    white_num += 1

                else:
                    black_num += 1
        if (black_num > white_num):
            tag = 255
            # print("白色的字体")
        else:
            tag = 0
            # print("黑色的字体")

        #  对横向 row 的投影
        sum_rows = [0 for n in range(rows)]
        for x in range(rows):
            num = 0
            for y in range(0, colums):
                if (gray[x, y] == tag):
                    sum_rows[x] = sum_rows[x] + 1
        # print("横向的投影: " + str(sum_rows))
        sum_row = 0
        for i in range(rows):
            sum_row += sum_rows[i]
        tag_row = int(sum_row / rows)
        # print("mean_sum_rows:  " + str(tag_row))

        # 上方的起始点,去掉车牌外部区域
        index1 = 0
        for i in range(0, int(rows/2)+2):
            #  判断可能因为边框的选取多出来的黑色区域,所以加上了sum_sum[i]< rows*3/4 ,这个判断条件
            if (sum_rows[i] < colums * 1 / 5 and sum_rows[i + 1] > colums * 1 / 5):
            # if (sum_rows[i] < colums * 1 / 5):
                index1 = i
                break
            if(i == int(rows/2)):
                index1 = 0
                break

        # 下方的起始点,去掉车牌外部区域
        index2 = 0
        # for i in range(rows - 1, -1, -1):
        for i in range(rows-1,int(rows/2)-2,-1):
            #  判断可能因为边框的选取多出来的黑色区域,所以加上了sum_sum[i]< rows*3/4 ,这个判断条件
            if (sum_rows[i] < colums * 1 / 5 and sum_rows[i - 1] > colums * 1 / 5):
            # if (sum_rows[i] < colums * 1 / 5):
                index2 = i - 1
                break
            if(i==int(rows/2)):
                index2 = rows-1
                break


        rows_length = index2 - index1 + 1

        # print("index1:"+str(index1)+"   "+"index2:"+str(index2))
        # cv2.imwrite("/Users/Quantum/Desktop/分割原图.jpg", image[range(index1, index2), :])
        # cv2.imwrite("/Users/Quantum/Desktop/分割灰度图.jpg", gray[range(index1, index2), :])


        double_row = [0 for n in range(colums)]
        for j in range(int(colums / 4), int(colums / 2)):
            for i in range(index1, index2):
                if (gray[i, j] == tag):
                    double_row[j] = double_row[j] + 1
        # print("double_row"+str(double_row))


        double = 0
        double_avg = 0

        if (rows > colums * 2 / 5):
            double = 1

        # 双排车辆
        if (double == 1):
            # print("double")

            for i in range(int(rows / 5), int(rows / 2)):
                if (double_row[i] == 0):
                    double_avg = i
                    break

            if (double_avg == 0):
                double_avg = double_row.index(min(double_row[:]))
            double_avg += int(rows / 5)

            index1 = double_avg

            #  对双排车牌的下半部分进行处理,重新进行赋值处理
            # 存储上下两个部分
            # cv2.imwrite(path+"double_up.jpg", image[range(double_avg + 1), :])
            # cv2.imwrite(path+"double_down.jpg", image[range((int)(double_avg), rows), :])

            # 存储两排车牌的上方字符
            # path_file_1 = path + file[0] + "____" + file + ".jpg"
            # cv2.imwrite(path_file_1,
            #             image[range(double_avg + 1), :][:, range((int)(colums * 1 / 7), (int)(colums / 2))])
            # path_file_2 = path + file[1] + "____" + file + ".jpg"
            # cv2.imwrite(path_file_2,
            #             image[range(double_avg + 1), :][:, range((int)(colums / 2), (int)(colums * 6 / 7))])

            self.result_double.append(0)
            self.result_double.append(double_avg)
            self.result_double.append((int)(colums * 1 / 7))
            self.result_double.append((int)(colums * 1 / 2))
            self.result_double.append((int)(colums * 1 / 2))
            self.result_double.append((int)(colums * 6 / 7))

            # image = cv2.imread(path+"double_down.jpg")
            # gray = cv2.medianBlur(image, 3)
            # # cv2.imwrite("/Users/Quantum/Desktop/medianBlur.jpg", gray)
            # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            # # cv2.imwrite("/Users/Quantum/Desktop/gray.jpg",gray)
            # ret2, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            gray = gray[range((int)(double_avg), rows), :]
            rows = image.shape[0]

            # print("double_dow_row:"+str(rows))

            rows_length = index2 - index1 + 1
            index2 = index2 - double_avg - 1
            index1 = 0
            self.result_double.append(double_avg)

            # print("index1:" + str(index1) + "   " + "index2:" + str(index2))

        if (double == 0):
            self.result.append(index1)
            self.result.append(index2)
        else:
            self.result_double.append(index2)

        #  纵向 colums 的投影
        sum_colums = [0 for n in range(colums)]
        for y in range(colums):
            num = 0
            for x in range(index1, index2 + 1):
                if (gray[x, y] == tag):
                    sum_colums[y] = sum_colums[y] + 1

        # print("纵向的投影: "+str(sum_colums))


        colum_start = 0
        colum_end = colums - 1
        #  去掉左右两边的可能的边框影响
        for i in range(colums):
            if (sum_colums[i] < rows_length * 3 / 4):
                colum_start = i
                break

        for i in range(colums - 1, -1, -1):
            if (sum_colums[i] < rows_length * 3 / 4):
                colum_end = i
                break

        #  平均列长度
        sum_colum = 0
        for i in range(colum_start, colum_end):
            sum_colum += sum_colums[i]
        tag_colum = int(sum_colum / colums)
        # print("mean_sum_colum:  " + str(tag_colum))

        #  标记列的,排除可能的噪声
        tag_colums = [0 for n in range(colums)]
        for i in range(colum_start, colum_end):
            if (sum_colums[i] > tag_colum / 3):
                tag_colums[i] = 1
            else:
                tag_colums[i] = 0

        # print("纵向的标记: "+str(tag_colums))


        #  记录列分割时,列的长度情况
        len_cos = []
        len_cos_start = []
        len_cos_end = []
        i = 0
        oo = -1
        for i in range(len(tag_colums)):
            if (i <= oo):
                continue
            if (tag_colums[i] == 1):
                start = i
                len_cos_start.append(start)
                len_co = 1
                for j in range(start + 1, len(tag_colums) + 1):
                    if (j == len(tag_colums)):
                        len_cos_end.append(j - 1)
                        len_cos.append(len_co)
                        oo = j
                    elif (tag_colums[j] == 0):
                        len_cos_end.append(j - 1)
                        oo = j
                        len_cos.append(len_co)
                        break
                    len_co += 1

        # 字符分割字符数目的判断
        char_num = 7
        if (double == 1):
            char_num = 5

        #  如果长度小于char_num个字符,表示字符的连接或者缺失, pass
        if (len(len_cos) < char_num):
            print("opps")
            # continue

        # print("len_cos :"+str(len_cos) )
        # print("len_cos_start :"+str(len_cos_start) )
        # print("len_cos_end :"+str(len_cos_end) )

        #  每个字符的平均所占大小
        sum_len_cos = 0
        for i in range(len(len_cos)):
            sum_len_cos += len_cos[i]
        mean_len_cos = int(sum_len_cos / len(len_cos))
        # print("mean_len_cos: "+str(mean_len_cos))


        # 针对那些比如像  使,领的车牌  ,因为字体是红色的,在二值化的时候会消失  这些字体只在第一位和最后一位
        if ((colums - len_cos_end[-1]) > mean_len_cos * 1.2):
            len_cos_start.append(len_cos_end[-1] + 2)
            len_cos_end.append(colums - 2)

        #  对于那些分割的片段
        tag_stop = 0
        first_len = 0

        if (len(len_cos) > char_num):
            # print("长度大于char_num")
            for i in range(len(len_cos) - char_num + 1):
                if (len_cos[i] >= mean_len_cos and first_len > mean_len_cos):
                    tag_stop = i
                    # print("********")
                    break
                if (i == len(len_cos) - char_num):
                    tag_stop = len(len_cos) - char_num + 1
                    # print("########")
                    break
                first_len += len_cos[i]

        #  第一个汉字是左右结构的进行合并处理
        if (tag_stop != 0):
            # print("汉字分隔")
            cos_start = len_cos_start[0]
            len_cos_start = len_cos_start[tag_stop:]
            len_cos_start.insert(0, cos_start)
            len_cos_end = len_cos_end[tag_stop - 1:]
            # print("汉字分隔len_cos_start :" + str(len_cos_start))
            # print("汉字分隔len_cos_end :" + str(len_cos_end))

        # 寻找最大的前七个字符分割点
        len_cos_again = []
        for i in range(len(len_cos_start)):
            len_cos_again.append(len_cos_end[i] - len_cos_start[i] + 1)
        # print("len_cos_again: "+str(len_cos_again))

        len_cos_again_copy = len_cos_again.copy()
        len_cos_again_copy.sort(reverse=True)

        mid = len_cos_again_copy[int(len(len_cos_again_copy)/2)]

        # len_cos_again = []
        # for i in range(len(len_cos_again)):
        #     if(len_cos_again[i] > mid * 1.6):
        #         print("fc")
        #         len_cos_start.insert(i,int((len_cos_end[i]+len_cos_start[i])/2) )
        #         len_cos_end.insert(i,int((len_cos_end[i]+len_cos_start[i])/2) )




        # len_cos_again_copy = len_cos_again.copy()
        # len_cos_again_copy.sort(reverse=True)
        # print(str(len_cos_again_copy))
        last_end = []
        last_start = []
        for i in range(char_num):
            pos = len_cos_again.index(len_cos_again_copy[i])
            len_cos_again[pos] = -1
            last_start.append(len_cos_start[pos])
            last_end.append(len_cos_end[pos])

        # print("last_start :" + str(last_start))
        # print("last_end :" + str(last_end))

        last_end.sort()
        last_start.sort()

        # print("七个最大的分隔last_start :" + str(last_start))
        # print("七个最大的分割last_end :" + str(last_end))


        if (double == 0):
            for i in range(len(last_start)):
                if (i == 0):
                    self.result.append(last_start[i])
                    self.result.append(last_start[i + 1])
                elif (i == 6):
                    self.result.append(last_end[i - 1])
                    self.result.append(last_end[i])
                else:
                    self.result.append(last_end[i - 1])
                    self.result.append(last_start[i + 1])
        if (double == 1):
            for i in range(len(last_start)):
                if (i == 0):
                    self.result_double.append(last_start[i])
                    self.result_double.append(last_start[i + 1])
                elif (i == 4):
                    self.result_double.append(last_end[i - 1])
                    self.result_double.append(last_end[i])
                else:
                    self.result_double.append(last_end[i - 1])
                    self.result_double.append(last_start[i + 1])


        if (double == 0):
            # print(self.result)
            return self.result,gray_or
        else:
           # print(self.result_double)
            return self.result_double,gray_or

def save_csv(image_name, predict_color, predict_chars):
    save_arr = np.empty((10000, 3), dtype=np.str)
    save_arr = pd.DataFrame(save_arr, columns=['车牌号', '车牌颜色','测试文件名'])
    predict_label = predict_chars
    for i in range(len(image_name)):
        filename = image_name[i]
        save_arr.values[i, 0] = predict_label[i]
        save_arr.values[i, 1] = predict_color[i]
        save_arr.values[i, 2] = filename
    save_arr.to_csv('submit_test.csv', decimal=',', encoding='gb2312', index=False, index_label=False)
    print('submit_test.csv have been write, locate is :', os.getcwd())

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

def color_rec(image_color):
    image = image_color
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    sp = H.shape
    rows = sp[0]
    colums = sp[1]

    yellow = 0
    blue = 0
    white = 0
    black = 0
    d = {'yellow': 0, 'blue': 0, 'white': 0,'black':0}
    for i in range(rows):
        for j in range(colums):
            if(10<=H[i][j]<=38):
                d['yellow']+=1
            elif(75<=H[i][j]<=130):
                d['blue']+=1
            # elif(H[i][j]<17 or H[i][j]>130):
            else:
                if(V[i][j]<80):
                    d['black']+=1
                if(V[i][j]>80):
                    d['white']+=1

    max = 0
    tag = 0
    for i in d:
        #print(i+"  "+str(d[i]))
        if(max<d[i]):
            max = d[i]
            tag = i
    if(tag == "yellow"):
        if(d['yellow'] < (int(d['white'])*1.5) ):
            tag="white"
    return tag

if __name__ == "__main__":

    chars_dic = {'0':'NumZero','1':'NumOne','2':'NumTwo','3':'NumThree','4':'NumFour','5':'NumFive','6':'NumSix','7':'NumSeven','8':'NumEight','9':'NumNine','A':'AlphaA','B':'AlphaB','C':'AlphaC','D':'AlphaD','E':'AlphaE','F':'ALphaF','G':'AlphaG','H':'AlphaH','J':'AlphaJ','K':'AlphaK','L':'AlphaL','M':'AlphaM','N':'AlphaN','P':'AlphaP','Q':'AlphaQ','R':'AlphaR','S':'AlphaS','T':'AlphaT','U':'AlphaU','V':'AlphaV','W':'AlphaW','X':'AlphaX','Y':'AlphaY','Z':'AlphaZ','澳':'HanZiAo','藏':'HanZiZang','川':'HanZiChuan','鄂':'HanZiE','甘':'HanZiGan1','赣':'HanZiGan2','港':'HanZiGang','挂':'HanZiGua','桂':'HanZiGui1','贵':'HanZiGui2','黑':'HanZiHei','沪':'HanZiHu','吉':'HanZiJi1','冀':'HanZiJi2','津':'HanZiJin1','晋':'HanZiJin2','京':'HanZiJing1','警':'HanZiJing2','辽':'HanZiLiao','领':'HanZiLing','鲁':'HanZiLu','蒙':'HanZiMeng','闽':'HanZiMing','宁':'HanZiNing','青':'HanZiQing','琼':'HanZiQiong','陕':'HanZiShan','使':'HanZiShi','苏':'HanZiSu','皖':'HanZiWan','湘':'HanZiXiang','新':'HanZiXin','学':'HanZiXue','渝':'HanZiYu1','豫':'HanZiYu2','粤':'HanZiYue','云':'HanZiYun','浙':'HanZiZhe'}
    nums_dic = {'0':'NumZero','1':'NumOne','2':'NumTwo','3':'NumThree','4':'NumFour','5':'NumFive','6':'NumSix','7':'NumSeven','8':'NumEight','9':'NumNine'}
    alpha_dic = {'A':'AlphaA','B':'AlphaB','C':'AlphaC','D':'AlphaD','E':'AlphaE','F':'ALphaF','G':'AlphaG','H':'AlphaH','J':'AlphaJ','K':'AlphaK','L':'AlphaL','M':'AlphaM','N':'AlphaN','P':'AlphaP','Q':'AlphaQ','R':'AlphaR','S':'AlphaS','T':'AlphaT','U':'AlphaU','V':'AlphaV','W':'AlphaW','X':'AlphaX','Y':'AlphaY','Z':'AlphaZ'}
    hanzi_dic = {'澳':'HanZiAo','藏':'HanZiZang','川':'HanZiChuan','鄂':'HanZiE','甘':'HanZiGan1','赣':'HanZiGan2','港':'HanZiGang','挂':'HanZiGua','桂':'HanZiGui1','贵':'HanZiGui2','黑':'HanZiHei','沪':'HanZiHu','吉':'HanZiJi1','冀':'HanZiJi2','津':'HanZiJin1','晋':'HanZiJin2','京':'HanZiJing1','警':'HanZiJing2','辽':'HanZiLiao','领':'HanZiLing','鲁':'HanZiLu','蒙':'HanZiMeng','闽':'HanZiMing','宁':'HanZiNing','青':'HanZiQing','琼':'HanZiQiong','陕':'HanZiShan','使':'HanZiShi','苏':'HanZiSu','皖':'HanZiWan','湘':'HanZiXiang','新':'HanZiXin','学':'HanZiXue','渝':'HanZiYu1','豫':'HanZiYu2','粤':'HanZiYue','云':'HanZiYun','浙':'HanZiZhe'}
    color_dic = {'yellow':'黄','blue':'蓝','white':'白','black':'黑'}
    #**********Net loading******************************
    PlateDetec_net_path = "./src/yolov2.cfg"
    PlateDetec_weight_path = "./src/yolov2.weights"
    PlateDetec_meta_path = "./src/yolov2.data"
    PlateDetec_darknet = Darknet(PlateDetec_meta_path,PlateDetec_net_path,PlateDetec_weight_path)
    CharsRec_net_path = "./src/darknet.cfg"
    CharsRec_weight_path = "./src/darknet_73000.weights"
    CharsRec_meta_path = "./src/darknet.data"
    CharsRec_darknet = Darknet(CharsRec_meta_path,CharsRec_net_path,CharsRec_weight_path)
    
    os.system("cls")
    print("**************************************")
    print("*************车牌识别 v1.0************")
    print("**************学校：HDU **************")
    print("***********指导老师：陈小雕***********")
    print("****参赛选手：刘一芃  马俊逸  王赛****")
    print("**************************************")
    print("\n")

    #**********Image loading*******************************1
    file_path = input("请输入图片或者文件夹路径：")
    while(not os.path.exists(file_path)):
        file_path = input("请输入正确的图片或者文件夹路径：")
    if os.path.isdir(file_path):
        image_list = []
        image_name = os.listdir(file_path)
        for i in range(len(image_name)):
            image_list.append(os.path.join(file_path,image_name[i]))
    else:
        image_list = []
        image_list.append(file_path)

    #**********PlateDetec and recognization******************************
    Output_chars = []
    image_name_list = []
    color_list = []
    for image_path in image_list:   
        Chars = []
        image_name = image_path.split("\\")[-1]
        #Detected
        image_cv = cv_imread(image_path)
        image, arr = PlateDetec_darknet.array_to_image(image_cv)
        detect = PlateDetec_darknet.detect(image)
        try:
            bounds = detect[0][2]
        except:
            print("fail detecting"+image_path)
            continue
        yExtent = int(bounds[3])
        xEntent = int(bounds[2])
        xCoord = int(bounds[0] - bounds[2]/2)
        yCoord = int(bounds[1] - bounds[3]/2)
        image_plate = image_cv[yCoord:yCoord+yExtent,xCoord:xCoord+xEntent]
        image_color_temp = color_rec(image_plate)
        color_list.append(color_dic[image_color_temp])
        #Croped
        print("processing.."+image_path)
        try:
            Chars_crop = CharsCroped(image_plate)
            Chars_bound, gray = Chars_crop.Croped()
        except:
            cv2.imwrite("./src/predic.jpg",image_plate)
            image_temp = cv2.imread("./src/predic.jpg")
            Chars_crop = CharsCroped(image_temp)
            Chars_bound, gray = Chars_crop.Croped()
        #Rec
        if(len(Chars_bound)==16):
            r_0 = Chars_bound[0]
            r_1 = Chars_bound[1]
            k = 0
            for i in range(7):
                image_char = gray[r_0:r_1,Chars_bound[2+k*2]:Chars_bound[3+k*2]]
                k = k + 1
                cv2.imwrite("./src/char%s.jpg"%k,image_char)
                image_char = cv2.imread("./src/char%s.jpg"%k)
                image_char_Image, arr = CharsRec_darknet.array_to_image(image_char)
                out = CharsRec_darknet.predict_image(CharsRec_darknet.net,image_char_Image)
                res = []
                for i in range(CharsRec_darknet.meta.classes):
                    res.append((CharsRec_darknet.altNames[i], out[i]))
                res = sorted(res, key=lambda x: -x[1])
                if(k == 1):
                    if res[0][0] in hanzi_dic.values():
                        normal = 1
                    else:
                        normal = 0
                temp_i = 0
                while(k == 2 and normal == 1):
                    if res[temp_i][0] in alpha_dic.values():
                        res[0] = res[temp_i]
                        break
                    else:
                        temp_i = temp_i + 1
                        continue
                if(res[0][0]=="AlphaS" and k>2):
                    res[0] = res[1]
                temp_j = 0
                while(k>2 and normal == 1):
                    if not (res[temp_j][0] in hanzi_dic.values()):
                        res[0] = res[temp_j]
                        break
                    else:
                        temp_j = temp_j + 1
                        continue
                Chars.append(res[0][0])
        else:
            r_0 = Chars_bound[0]
            r_1 = Chars_bound[1]
            k = 0
            for i in range(2):
                image_char = gray[r_0:r_1,Chars_bound[2+k*2]:Chars_bound[3+k*2]]
                k = k + 1
                cv2.imwrite("./src/char%s.jpg"%k,image_char)
                image_char = cv2.imread("./src/char%s.jpg"%k)
                image_char_Image, arr = CharsRec_darknet.array_to_image(image_char)
                out = CharsRec_darknet.predict_image(CharsRec_darknet.net,image_char_Image)
                res = []
                for i in range(CharsRec_darknet.meta.classes):
                    res.append((CharsRec_darknet.altNames[i], out[i]))
                res = sorted(res, key=lambda x: -x[1])
                if(k == 1):
                    if res[0][0] in hanzi_dic.values():
                        normal = 1
                    else:
                        normal = 0
                temp_i = 0
                while(k == 2 and normal == 1):
                    if res[temp_i][0] in alpha_dic.values():
                        res[0] = res[temp_i]
                        break;
                    else:
                        temp_i = temp_i + 1
                        continue
                if(res[0][0]=="AlphaS" and k!=2):
                    res[0] = res[1]
                Chars.append(res[0][0])

            r_2 = Chars_bound[6]
            r_3 = Chars_bound[7]
            for i in range(5):
                image_char = gray[r_0:r_1,Chars_bound[4+k*2]:Chars_bound[5+k*2]]
                k = k + 1
                cv2.imwrite("./src/char%s.jpg"%k,image_char)
                image_char = cv2.imread("./src/char%s.jpg"%k)
                image_char_Image, arr = CharsRec_darknet.array_to_image(image_char)
                out = CharsRec_darknet.predict_image(CharsRec_darknet.net,image_char_Image)
                res = []
                for i in range(CharsRec_darknet.meta.classes):
                    res.append((CharsRec_darknet.altNames[i], out[i]))
                res = sorted(res, key=lambda x: -x[1])
                Chars.append(res[0][0])
#        print(image_name)
#        print(Chars)
        for temp_num in range(len(Chars)):
            Chars[temp_num] = list(chars_dic.keys())[list(chars_dic.values()).index(Chars[temp_num])]
        Output_chars.append("".join(Chars))
        image_name_list.append(image_name)
    save_csv(image_name_list, color_list, Output_chars)
    os.system("pause")