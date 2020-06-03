import random
import numpy as np
import cv2 as cv
import os
from labelme import utils
from tqdm import tqdm
import yaml
import PIL


def gen_background(w, h):
    img = np.zeros([h, w, 3], np.uint8) + 255
    return img


def point_transform(point, M):
    px = M[0][0] * point[0] + M[0][1] * point[1] + M[0][2]
    py = M[1][0] * point[0] + M[1][1] * point[1] + M[1][2]
    return [int(px), int(py)]


def target_rotate(ftimg, rotate, scale, pts):
    h, w, _ = ftimg.shape
    matRotate = cv.getRotationMatrix2D((h * 0.5, w * 0.5), rotate, scale)
    dst = cv.warpAffine(ftimg, matRotate, (4000, 4000))
    # cv.imshow("backimage", dst)
    # cv.waitKey(0)
    pts_t = []
    for point in pts:
        dst_point = point_transform(point, matRotate)
        pts_t.append(dst_point)
    return dst, pts_t


def Gen_Segmentation_Datas(bgimg, ftimg, pts, location, rotate, scale):
    deltax = location[0]
    deltay = location[1]
    ftimg, dst_points = target_rotate(ftimg, rotate, scale, pts)

    poi_min = np.min(dst_points, 0) - 5
    poi_max = np.max(dst_points, 0) + 5
    for j in range(len(dst_points)):
        dst_points[j][0] = dst_points[j][0] - poi_min[0]
        dst_points[j][1] = dst_points[j][1] - poi_min[1]
    ftimg = ftimg[poi_min[1]:poi_max[1], poi_min[0]:poi_max[0]]
    # cv.imshow("backimage", ftimg)
    # cv.waitKey(0)
    rows, cols = ftimg.shape[:2]
    if (bgimg.shape[:2][0] > (rows + deltax)) & (bgimg.shape[:2][1] > (cols + deltay)):
        roi = bgimg[deltax:rows + deltax, deltay:cols + deltay]

        mask = np.zeros(ftimg.shape[:2], np.uint8)
        pts = np.array(dst_points, np.int32)
        mask = cv.polylines(mask, [pts], True, (255, 255, 255))
        mask = cv.fillPoly(mask, [pts], (255, 255, 255))
        notmask = cv.bitwise_not(mask)
        # cv.imshow("backimage", notmask)
        # cv.waitKey(0)
        backimage = cv.bitwise_and(roi, roi, mask=notmask)
        frontpic = cv.bitwise_and(ftimg, ftimg, mask=mask)

        result = cv.add(backimage, frontpic)

        resultimg = bgimg.copy()
        resultimg[deltax:rows + deltax, deltay:cols + deltay] = result
        for i in range(len(dst_points)):
            dst_points[i][0] = dst_points[i][0] + location[1]
            dst_points[i][1] = dst_points[i][1] + location[0]
        return resultimg, dst_points
    else:
        return bgimg, None


def save_label_info(img, name, dir_label, dir_info, dir_vis, shape_):
    label_name_to_value = {'_background_': 0, "1": 1, "2": 2, "3": 3}
    label_values, label_names = [], []
    for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
        # print(ln,lv)
        label_values.append(lv)
        label_names.append(ln)
    assert label_values == list(range(len(label_values)))

    lbl = utils.shapes_to_label(img.shape, shape_, label_name_to_value)  # 'instance'
    # print(ins)

    captions = ['{}: {}'.format(lv, ln)
                for ln, lv in label_name_to_value.items()]
    lbl_viz = utils.draw_label(lbl, img, captions)

    # ins_viz=utils.draw_label(lbl, img, captions)
    # PIL.Image.fromarray(img).save('img.png')
    # PIL.Image.fromarray(lbl).save(osp.join(out_dir, 'label.png'))
    utils.lblsave(dir_label + name + '.png', lbl)
    PIL.Image.fromarray(lbl_viz).save(dir_vis + name + '.png')
    info = dict(label_names=label_names)
    with open(dir_info + name + '.yaml', 'w') as f:
        yaml.safe_dump(info, f, default_flow_style=False)


if __name__ == '__main__':
    path="data"
    if not os.path.exists(path):
        os.mkdir(path)
    out_dir_img = path+"/img/"
    out_dir_label = path+"/label/"
    out_dir_info = path+"/info/"
    out_dir_vis = path+"/vis/"
    label_name = ["1", "2", "3"]
    if not os.path.exists(out_dir_img):
        os.mkdir(out_dir_img)
    if not os.path.exists(out_dir_label):
        os.mkdir(out_dir_label)
    if not os.path.exists(out_dir_info):
        os.mkdir(out_dir_info)
    if not os.path.exists(out_dir_vis):
        os.mkdir(out_dir_vis)
    # root="data/" #保存目录


    pts1=[[782, 1328], [1095, 1283], [1492, 1238], [1912, 1202], [2292, 1167], [2628, 1147], [2876, 1131], 
          [2953, 1147], [2957, 1193], [2957, 1234], [2931, 1267], [2915, 1296], [2924, 1512], [2960, 1528], 
          [2976, 1554], [2963, 1615], [2889, 1644], [2537, 1647], [1828, 1647], [1276, 1615], [763, 1593], 
          [715, 1589], [695, 1515], [695, 1428], [715, 1347]]
    pts2=[[782, 1328], [1095, 1283], [1492, 1238], [1912, 1202], [2292, 1167], [2628, 1147], [2876, 1131], 
      [2953, 1147], [2957, 1193], [2957, 1234], [2931, 1267], [2915, 1296], [2924, 1512], [2960, 1528], 
      [2976, 1554], [2963, 1615], [2889, 1644], [2537, 1647], [1828, 1647], [1276, 1615], [763, 1593], 
      [715, 1589], [695, 1515], [695, 1428], [715, 1347]]
    pts3=[[782, 1328], [1095, 1283], [1492, 1238], [1912, 1202], [2292, 1167], [2628, 1147], [2876, 1131], 
      [2953, 1147], [2957, 1193], [2957, 1234], [2931, 1267], [2915, 1296], [2924, 1512], [2960, 1528], 
      [2976, 1554], [2963, 1615], [2889, 1644], [2537, 1647], [1828, 1647], [1276, 1615], [763, 1593], 
      [715, 1589], [695, 1515], [695, 1428], [715, 1347]]
    
    frontimg1 = cv.imread('zz.jpg')  # target原图片1
    frontimg2 = cv.imread("def_1.jpg") # target原图片2
    frontimg3 = cv.imread("def_2.jpg") # # target原图片3
    pts = [pts1, pts2, pts3]
    frontimgs = [frontimg1, frontimg2, frontimg3]
    for i in tqdm(range(50)):  # 生成图片(生成失败的跳过)
        # Time=time.strftime('%Y-%m-%d-%H-%M-%S')
        Time = '%06d' % i
        bgimg = cv.imread('black_bg.jpg')  # 背景图片
        h, w, _ = bgimg.shape
        ran_tar_nums = random.randrange(1, 5)  # 随机生成目标数
        nums = 0
        shapes = []
        for j in range(ran_tar_nums):
            ran_x = random.randrange(0, h)
            ran_y = random.randrange(0, w)
            ran_id = random.randrange(0, 3)
            ran_rotate = random.randrange(0, 360)

            bgimg, dst = Gen_Segmentation_Datas(bgimg, frontimgs[ran_id], pts[ran_id], (ran_x, ran_y), ran_rotate,
                                          0.3)  # 生成合成图片
            shape = {}
            if dst != None:

                shape['label'] = label_name[ran_id]
                shape['points'] = dst
                shape['shape_type'] = 'polygon'
                nums += 1
            if shape != {}:
                shapes.append(shape)
        # print(nums,ran_tar_nums)
        if nums != 0:
            cv.imwrite(out_dir_img + Time + ".jpg", bgimg)  # 储存图片
            save_label_info(bgimg, Time, out_dir_label, out_dir_info, out_dir_vis, shapes)  # 储存label,info

        #print(nums )
            # cv.namedWindow("1", cv.WINDOW_NORMAL)
            # cv.imshow("1", bgimg)
            # cv.waitKey(30)

    """
    #Gen_Segmentation_Datas(bgimg,ftimg,pts,location,rotate,scale)
      bgimg:背景图片
      ftimg:target原图片
      pts:target在原图片中的轮廓点
      location:target在背景中的像素位置
      rotate:旋转角度
      scale:target缩放尺度
    :return:  (若没有加入target返回原背景和None)
      resultimg:合成后图片
      dst_points:合成后target轮廓坐标  
    """


