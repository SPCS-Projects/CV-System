from tqdm import tqdm
import sqlite3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import ast
from datetime import datetime
import random
import string
import shutil
from pathlib import Path


def gen_id(size):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size))


def sql(facial_data, target_img_path, database, time_stamp, stats, target_img, timestr, path):
    database_old = None
    for file in os.listdir(path + '/database/'):
        if file.endswith(".db"):
            database_old = (os.path.join((path + '/database/'), file))
    if database_old is None:
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute('''create table face_meta (ID INT primary key, uID VARCHAR(10), EMBEDDING BLOB, IMG TEXT, FULL_IMG TEXT, FIRST_SEEN TEXT, LAST_SEEN TEXT, TIME_RANGES TEXT, TOTAL_TIME REAL, AGE TEXT, GENDER TEXT, RACE TEXT)''')
        count = 0
    else:
        conn = sqlite3.connect(database_old)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM face_meta")
        count = (cursor.fetchone())[0]
    append_num = False
    value = 1

    if len(facial_data) > 1:
        append_num = True

    for i in range(len(facial_data)):
        dataframe = []
        data = []

        target_embedding = facial_data[i]

        if append_num:
            new_path = target_img_path + "_" + str(value)
            value += 1
            data.append(new_path)
        else:
            data.append(target_img_path)
        data.append(target_embedding)
        dataframe.append(data)

        select_statement = 'select uID, embedding from face_meta'
        results = cursor.execute(select_statement)

        instances = []
        for result in results:
            uID = result[0]
            embedding_bytes = result[1]
            embedding = np.frombuffer(embedding_bytes, dtype='float32')

            instance = []
            instance.append(uID)
            instance.append(embedding)
            instances.append(instance)

        result_df = pd.DataFrame(instances, columns=['uID', 'embedding'])
        add_df = pd.DataFrame(dataframe, columns=['img_name', 'embedding'])
        target_duplicated = np.array([target_embedding, ] * result_df.shape[0])
        result_df['target'] = target_duplicated.tolist()

        def findEuclideanDistance(row):
            source = np.array(row['embedding'])
            target = np.array(row['target'])
            distance = (source - target)
            return np.sqrt(np.sum(np.multiply(distance, distance)))

        try:
            result_df['distance'] = result_df.apply(findEuclideanDistance, axis=1)
            result_df = result_df[result_df['distance'] <= 10]
            result_df = result_df.sort_values(by=['distance']).reset_index(drop=True)
        except:
            pass
            # print("database was empty")
        result_df = result_df.drop(columns=["embedding", "target"])

        # print("Result", str(i) + ":\n", result_df.head(10))
        if len(result_df) == 0:
            for index, instance in tqdm(add_df.iterrows(), total=add_df.shape[0]):
                uID = gen_id(50)
                embeddings = instance['embedding']
                insert_statement = 'INSERT INTO face_meta (ID, uID, EMBEDDING, IMG, FULL_IMG, FIRST_SEEN, LAST_SEEN, AGE, GENDER, RACE, TOTAL_TIME) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
                #temporarily still storing a time string for img name since were saving pics
                image_name = instance['img_name'] + ".png"
                plt.imsave(path + "/faces/{}".format(image_name), target_img[i][0][:, :, ::-1])
                img_binary = str(["/faces/{}".format(image_name)])
                full_img_path = "/images/{}".format(timestr+".png")
                full_img = str([full_img_path])
                insert_args = (count, uID, embeddings.tobytes(), img_binary, full_img, time_stamp, time_stamp, stats[i][2], stats[i][1], stats[i][0], 0.0)
                cursor.execute(insert_statement, insert_args)

                count += 1
        elif len(result_df) > 0:
            comps = 11.0
            index = 0
            for idx in range(len(result_df)):
                comp = result_df.iat[idx, 1]
                if comps > comp:
                    comps = comp
                    index = idx
            name = result_df.iat[index, 0]
            insert_args = (name,)
            # Grabs the relavent time info at the same time
            insert_statement = "SELECT IMG, FULL_IMG, FIRST_SEEN, LAST_SEEN, TIME_RANGES, TOTAL_TIME FROM face_meta WHERE uID = ?"
            results = cursor.execute(insert_statement, insert_args)
            for result in results:
                face = ast.literal_eval(result[0])
                full_img = ast.literal_eval(result[1])
                # Added grabbing the extra time information
                first_seen = result[2]
                last_seen = result[3]
                if result[4] is not None:
                    time_ranges = ast.literal_eval(result[4])
                else:
                    time_ranges = None
                total_time = result[5]

            # .strip('][').replace('\"', '').split(',') could potentially use something like this since its way faster, and probably will

            # face = ast.literal_eval(face)
            # full_img = ast.literal_eval(full_img)

            #these leaves each entry wrapped in "" and the actual string containing ''
            # face = face.strip('[]').replace('"', '').replace(' ', '').split(',')
            # full_img = full_img.strip('][').replace('\"', '').replace('"', '').split(',')

            image_name = data[0] + ".png"
            full_img_path = "/images/{}".format(timestr + ".png")
            plt.imsave(path + "/faces/{}".format(image_name), target_img[i][0][:, :, ::-1])
            face.append("/faces/{}".format(image_name))
            full_img.append(full_img_path)

            # Checking if it should cache current time range, 15 being the second time period before the person is considered to have left
            cache_times = False
            dt_last_seen = datetime.strptime(last_seen, "%Y-%m-%d %H:%M:%S.%f")
            if (time_stamp - dt_last_seen).total_seconds() > 600:
                cache_times = True
                if time_ranges is None:
                    time_ranges = [[first_seen, last_seen]]
                else:
                    time_ranges.append([first_seen, last_seen])
                dt_first_seen = datetime.strptime(first_seen, "%Y-%m-%d %H:%M:%S.%f")
                total_time = total_time + (dt_last_seen - dt_first_seen).total_seconds()

            if cache_times:
                insert_args = (str(face), str(full_img), time_stamp, time_stamp, str(time_ranges), total_time, name)
                cursor.execute('UPDATE face_meta SET IMG = ?, FULL_IMG = ?, FIRST_SEEN = ?, LAST_SEEN = ?, TIME_RANGES = ?, TOTAL_TIME = ? WHERE uID = ?', insert_args)
            else:
                insert_args = (str(face), str(full_img), time_stamp, name)
                cursor.execute('UPDATE face_meta SET IMG = ?, FULL_IMG = ?, LAST_SEEN = ? WHERE uID = ?', insert_args)
    conn.commit()
    if not database_old:
        i = 0
        database_name = Path(database).stem
        new_file = path+"/db_backup/{}_{}.db".format(database_name, i)
        while os.path.isfile(new_file):
            i += 1
            new_file = path + "/db_backup/{}_{}.db".format(database_name, i)
        shutil.copyfile(database, new_file)
    else:
        i = 0
        database_name = Path(database_old).stem
        new_file = path + "/db_backup/{}_{}.db".format(database_name, i)
        while os.path.isfile(new_file):
            i += 1
            new_file = path + "/db_backup/{}_{}.db".format(database_name, i)
        shutil.copyfile(database_old, new_file)
