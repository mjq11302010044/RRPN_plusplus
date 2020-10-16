# -*-coding=utf-8-*-
import os
import socket
import time
import json
import e2e_demo

# 定义全局列表用来存储子文件夹
list1 = []

_DEBUG = True

def recv_img(path_out, dir_name, dir_socket):

    out_file = os.path.join(dir_name, path_out)
    data = dir_socket.recv(1024)

    cmd, filename, filesize = str(data, 'utf8').split('|')  # 第一次提取请求信息，获取  post name size
    # filesize = os.stat(data)
    # path = os.path.join(BASE_DIR, 'MNIST_80', filename)

    if _DEBUG:
        print('********** frame data **********')
        print('cmd:', cmd)
        print('filename:', filename)
        print('filesize:', filesize)

    filesize = int(filesize)

    f = open(out_file, 'wb')
    has_receive = 0
    while has_receive != filesize:
        data = dir_socket.recv(1024)  # 第二次获取请求，这次获取的就是传递的具体内容了，1024为文件发送的单位
        f.write(data)
        has_receive += len(data)

    f.close()

def deal_dir():
    # 待完善用于处理子文件夹,需要利用递归完成
    pass


def main():
    # 创建套接字
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 固定端口号
    tcp_socket.bind(("", 9992))
    # 被动套接字转换为主动套接字
    tcp_socket.listen(128)
    # 将队列中的客户端取出

    # 　接受客户端消息

    demo_dir = './demo_img'
    if not os.path.isdir(demo_dir):
        print('create demo_dir..')
        os.makedirs(demo_dir)

    demo_imname = 'demo.jpg'

    print('Prepared to get client...')

    while True:

        client_socket, client_ip = tcp_socket.accept()
        try:
            while True:
                recv_img(demo_imname, demo_dir, client_socket)

                ###################### rec proc ######################

                result_dict = e2e_demo.improc(impath=os.path.join(demo_dir, demo_imname))

                ######################################################

                if _DEBUG:
                    print('result_dict:', result_dict)

                json_bytes = json.dumps(result_dict).encode(encoding='utf-8')

                result_size = len(json_bytes)

                client_socket.sendall(str(result_size).encode(encoding='utf-8'))

                # wait for produce
                time.sleep(0.1)

                client_socket.send(json_bytes)
        except Exception as e:
            print(e)
            client_socket.close()

if __name__ == '__main__':
    main()