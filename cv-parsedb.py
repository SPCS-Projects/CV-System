import sqlite3
import ast
import os
import shutil
from datetime import datetime


def createDirs(dirs):
    if type(dirs) is list:
        for dir in dirs:
            if not os.path.exists(dir):
                os.mkdir(dir)
    else:
        if not os.path.exists(dirs):
            os.mkdir(dirs)
    return True


def getPath(__file__):
    return os.path.dirname(os.path.abspath(__file__))


def parse(database, path):
    createDirs(path+"/results")
    file = open(path+"/results/db.csv", 'w+')
    file.write("ID, Name, First Seen, Last Seen, Dwell Time, Age, Gender, Race\n")
    file.close()
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    results = cursor.execute("SELECT IMG_NAME, IMG, FULL_IMG, FIRST_SEEN, LAST_SEEN, AGE, GENDER, RACE, ID FROM face_meta")
    for result in results:
        dir = "{}/results/{}".format(path, result[0])
        faces_dir = "{}/faces".format(dir)
        full_img_dir = "{}/full".format(dir)
        dirs = [dir, faces_dir, full_img_dir]
        createDirs(dirs)

        faces = result[1]
        faces_array = ast.literal_eval(faces)
        for img in faces_array:
            shutil.copyfile(path+img, faces_dir+img[6:])

        full_img = result[2]
        full_img_array = ast.literal_eval(full_img)
        for img in full_img_array:
            shutil.copyfile(path + img, full_img_dir+img[7:])

        first = datetime.strptime(result[3], "%Y-%m-%d %H:%M:%S.%f")
        if result[4] is not None:
            last = datetime.strptime(result[4], "%Y-%m-%d %H:%M:%S.%f")
        else:
            last = first

        info = "{}, {}, {}, {}, {}, {}, {}, {}\n".format(result[8], result[0], result[3], result[4], last - first, result[5], result[6], result[7])

        file = open(dir + "/info.csv", 'w+')
        file.write("ID, Name, First Seen, Last Seen, Dwell Time, Age, Gender, Race\n")
        file.write(info)
        file.close()

        file = open(path+"/results/db.csv", 'a')
        file.write(info)
        file.close()


def run(timestamp=None):
    path = getPath(__file__)
    if timestamp is None:
        database = "{}/database/{}.db".format(path, datetime.now().strftime("%Y-%m-%d"))
    else:
        database = "{}/database/{}.db".format(path, timestamp)
    if not os.path.isfile(database):
        print("database doesn't exist yet to parse")
    else:
        print("did exist")
        parse(database, path)


run()
