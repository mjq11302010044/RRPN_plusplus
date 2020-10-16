# coding=utf-8
import cv2 as cv
import os
import platform
import socket
import os
import threading
import time
import json
import numpy as np

# server_name = 'mjq'
# server_pswd = 'mjq'

IP = "10.141.209.43"
dir_name = './demo/'
file = 'demo_img.jpg'

_DEBUG = False

def draw_demo(frame, result_dict):

    polys = result_dict['polys']
    recs = result_dict['recs']
    id_cnt = 0

    for poly in polys:
        pt_num = len(poly)

        poly_np = np.array(poly).reshape(-1, 2)
        min_x = np.min(poly_np[:, 0])
        min_y = np.min(poly_np[:, 1])
        id_cnt += 1
        for i in range(pt_num):
            cv.line(frame, (poly[i][0], poly[i][1]), (poly[(i+1)%pt_num][0], poly[(i+1)%pt_num][1]), (0, 255, 0), 1)
        cv.putText(frame, str(id_cnt), (int(min_x), int(min_y)), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        print('id', id_cnt, ':', recs[id_cnt-1])
    return frame


def video_demo():

    capture=cv.VideoCapture(0) 
    while(True):
        ref,frame=capture.read()
        cv.imshow("1",frame)

        c = cv.waitKey(200) & 0xff
        if c==27:
            capture.release()
            break


def upload_img(file, dir_name, tcp_socket):
    path = os.path.join(dir_name, file)

    filename = os.path.basename(path)

    file_size = os.stat(path).st_size

    file_info = 'post|%s|%s' % (filename, file_size)  # split获取字符串的信息       以此方式打包，依次为   cmd/name/size
    tcp_socket.sendall(bytes(file_info, 'utf-8'))  # 第一次发送请求，不是具体内容，而是先发送数据信息

    f = open(path, 'rb')
    has_sent = 0
    while has_sent != file_size:
        data = f.read(1024)
        tcp_socket.sendall(data)  # 发送真实数据
        has_sent += len(data)

    f.close()
    print('上传成功')


def main():
    # 创建套接字
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 连接服务器
    tcp_socket.connect((IP, 9992))

    capture = cv.VideoCapture(0)

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    while (True):
        ref, frame = capture.read()
        if _DEBUG:
            print('frame meta:', frame.shape)
        cv.imwrite(os.path.join(dir_name, file), frame)

        upload_img(file, dir_name, tcp_socket)

        all_res_bytes = bytes()

        # return result size
        result_size = int(tcp_socket.recv(1024).decode(encoding='utf-8'))

        # last_len = 4096

        total_size = 0

        while result_size - total_size > 0:
            temp_str = tcp_socket.recv(4096)
            all_res_bytes += temp_str
            last_len = len(temp_str)
            total_size += last_len
            # print('last_len:', last_len)

        dump_str = all_res_bytes.decode(encoding='utf-8')

        if _DEBUG:
            print('dump_str:', dump_str)

        result_json = json.loads(dump_str)
        if _DEBUG:
            print('results:', result_json)
        cv.imshow("show", draw_demo(frame, result_json))
        c = cv.waitKey(2000) & 0xff
        if c == 27:
            capture.release()
            break



if __name__ == '__main__':
    main()


# video_demo()
# cv.waitKey()

# print(os.popen('pscp ').read())