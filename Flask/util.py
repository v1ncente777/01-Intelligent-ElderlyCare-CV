import os
import shutil
import csv
import oss2
import requests
import json
import time
import queue
from itertools import islice

message_queue = queue.Queue()


# http send post    aim to send data to spring and get result
def http_post(url, event_type, event_location, event_desc, old_people_id=-1):
    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    data_json = {'id': 0,  # id=0 means insert; id=1 means update;
                 'event_desc': event_desc,
                 'event_type': event_type,
                 'event_date': current_time,
                 'event_location': event_location,
                 'oldperson_id': old_people_id}
    message_queue.put(data_json)
    print(data_json)
    response = requests.post(url, data=json.dumps(data_json), headers=headers).text
    # print(response)
    return response


def oss_upload(path):
    list_path = []
    list_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            list_path.append(file_path)
            list_name.append(file)
    auth = oss2.Auth('LTAI5t8fCTtpyTorYRSHCtfi', 'XCuhZcuSbUXeCMF3wvdtMrMS6FMpkY')
    # Endpoint以杭州为例，其它Region请按实际情况填写。
    bucket = oss2.Bucket(auth, 'http://oss-cn-beijing.aliyuncs.com/', 'kioffk')
    print(list_name)
    for i in range(len(list_path)):
        with open(list_path[i], 'rb') as fileobj:
            # Seek方法用于指定从第1000个字节位置开始读写。上传时会从您指定的第1000个字节位置开始上传，直到文件结束。
            fileobj.seek(0, os.SEEK_SET)
            # Tell方法用于返回当前位置。
            current = fileobj.tell()
            bucket.put_object(path + '/' + list_name[i], fileobj)


def oss_upload_model(path, version_path):
    auth = oss2.Auth('LTAI5t8fCTtpyTorYRSHCtfi', 'XCuhZcuSbUXeCMF3wvdtMrMS6FMpkY')
    # Endpoint以杭州为例，其它Region请按实际情况填写。
    bucket = oss2.Bucket(auth, 'http://oss-cn-beijing.aliyuncs.com/', 'kioffk')
    with open(path, 'rb') as fileobj:
        # Seek方法用于指定从第1000个字节位置开始读写。上传时会从您指定的第1000个字节位置开始上传，直到文件结束。
        fileobj.seek(0, os.SEEK_SET)
        # Tell方法用于返回当前位置。
        current = fileobj.tell()
        bucket.put_object(path, fileobj)
    with open(version_path, 'rb') as fileobj:
        # Seek方法用于指定从第1000个字节位置开始读写。上传时会从您指定的第1000个字节位置开始上传，直到文件结束。
        fileobj.seek(0, os.SEEK_SET)
        # Tell方法用于返回当前位置。
        current = fileobj.tell()
        bucket.put_object(version_path, fileobj)


def oss_download(remote_file, local_file):
    auth = oss2.Auth('LTAI5t8fCTtpyTorYRSHCtfi', 'XCuhZcuSbUXeCMF3wvdtMrMS6FMpkY')
    # Endpoint以杭州为例，其它Region请按实际情况填写。
    bucket = oss2.Bucket(auth, 'http://oss-cn-beijing.aliyuncs.com/', 'kioffk')
    # if_access_bucket = oss2.Bucket(self.auth, config['OSS_CONFIG']['endpoint'],'if-access')
    local_path = os.path.dirname(local_file)
    if local_path and not os.path.exists(local_path):
        os.makedirs(local_path)

    res = bucket.get_object_to_file(remote_file, local_file)
    return res


def oss_download_image():
    auth = oss2.Auth('LTAI5t8fCTtpyTorYRSHCtfi', 'XCuhZcuSbUXeCMF3wvdtMrMS6FMpkY')
    # Endpoint以杭州为例，其它Region请按实际情况填写。
    bucket = oss2.Bucket(auth, 'http://oss-cn-beijing.aliyuncs.com/', 'kioffk')
    # if_access_bucket = oss2.Bucket(self.auth, config['OSS_CONFIG']['endpoint'],'if-access')
    if os.path.exists('images/oldpeople'):
        shutil.rmtree('images/oldpeople', True)
    os.mkdir('images/oldpeople')
    if os.path.exists('images/volunteer'):
        shutil.rmtree('images/volunteer', True)
    os.mkdir('images/volunteer')
    if os.path.exists('images/employee'):
        shutil.rmtree('images/employee', True)
    os.mkdir('images/employee')

    data = [
        ("id_card", "name", "type"),
        ("Unknown", "陌生人", ""),
    ]
    f = open('info/people_info.csv', 'w')
    writer = csv.writer(f)

    for b in islice(oss2.ObjectIterator(bucket, prefix='images/oldpeople'), 1000):
        if b.key.endswith('/'):
            print('directory: ', b.key)
            isExists = os.path.exists(b.key)
            if not isExists:
                os.makedirs(b.key)
                print(b.key, '创建成功')
        else:
            if not os.path.exists(os.path.split(b.key)[0]):
                path = os.path.split(b.key)[0]
                os.makedirs(path)
                str_temp = path.split('/')
                data_item = (str_temp.__getitem__(len(str_temp) - 1), str_temp.__getitem__(len(str_temp) - 1), "old_people")
                data.append(data_item)
                print(os.path.split(b.key)[0], '创建成功')
            print("downloadfile-->", b.key)
            bucket.get_object_to_file(b.key, b.key)
    for b in islice(oss2.ObjectIterator(bucket, prefix='images/volunteer'), 1000):
        if b.key.endswith('/'):
            print('directory: ', b.key)
            isExists = os.path.exists(b.key)
            if not isExists:
                os.makedirs(b.key)
                print(b.key, '创建成功')
        else:
            if not os.path.exists(os.path.split(b.key)[0]):
                path = os.path.split(b.key)[0]
                os.makedirs(path)
                str_temp = path.split('/')
                data_item = (str_temp.__getitem__(len(str_temp) - 1), str_temp.__getitem__(len(str_temp) - 1), "volunteer")
                data.append(data_item)
                print(os.path.split(b.key)[0], '创建成功')
            print("downloadfile-->", b.key)
            bucket.get_object_to_file(b.key, b.key)
    for b in islice(oss2.ObjectIterator(bucket, prefix='images/employee'), 1000):
        if b.key.endswith('/'):
            print('directory: ', b.key)
            isExists = os.path.exists(b.key)
            if not isExists:
                os.makedirs(b.key)
                print(b.key, '创建成功')
        else:
            if not os.path.exists(os.path.split(b.key)[0]):
                path = os.path.split(b.key)[0]
                os.makedirs(path)
                str_temp = path.split('/')
                data_item = (str_temp.__getitem__(len(str_temp) - 1), str_temp.__getitem__(len(str_temp) - 1), "employee")
                data.append(data_item)
                print(os.path.split(b.key)[0], '创建成功')
            print("downloadfile-->", b.key)
            bucket.get_object_to_file(b.key, b.key)

    for i in data:
        writer.writerow(i)
    f.close()

