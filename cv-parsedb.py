import sqlite3
import ast
import os
import shutil
from datetime import datetime


def createDirs(dirs):
    if isinstance(dirs, list):
        for d in dirs:
            if not os.path.exists(d):
                os.mkdir(d)
    else:
        if not os.path.exists(dirs):
            os.mkdir(dirs)
    return True


def getPath(__file__):
    return os.path.dirname(os.path.abspath(__file__))


def parse(database, path):
    # Make top-level results dir
    createDirs(path + "/results")

    # Create master CSV with new header
    master_csv = os.path.join(path, "results", "db.csv")
    with open(master_csv, "w+") as file:
        file.write("ID, uID, First Seen, Last Seen, Dwell Time, Age, Gender, Race\n")

    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    results = cursor.execute(
        """
        SELECT ID, uID, IMG, FULL_IMG, FIRST_SEEN, LAST_SEEN,
               AGE, GENDER, RACE, TOTAL_TIME
        FROM face_meta
        """
    )

    for result in results:
        # Unpack row with clear names
        person_id      = result[0]
        uid            = result[1]
        faces          = result[2]
        full_img       = result[3]
        first_seen_str = result[4]
        last_seen_str  = result[5]
        age            = result[6]
        gender         = result[7]
        race           = result[8]
        total_time     = result[9]

        # Directories per person
        dir_path      = f"{path}/results/{person_id}"
        faces_dir     = f"{dir_path}/faces"
        full_img_dir  = f"{dir_path}/full"
        createDirs([dir_path, faces_dir, full_img_dir])

        # Copy face crops
        if faces:
            faces_array = ast.literal_eval(faces)
            for img in faces_array:
                shutil.copyfile(path + img, faces_dir + img[6:])

        # Copy full images
        if full_img:
            full_img_array = ast.literal_eval(full_img)
            for img in full_img_array:
                shutil.copyfile(path + img, full_img_dir + img[7:])

        # Handle timestamps (for consistency / sanity)
        first = datetime.strptime(first_seen_str, "%Y-%m-%d %H:%M:%S.%f")
        if last_seen_str is not None:
            last = datetime.strptime(last_seen_str, "%Y-%m-%d %H:%M:%S.%f")
        else:
            last = first
            last_seen_str = first_seen_str  # if NULL, mirror first seen

        if total_time is not None:
            dwell_time = total_time
        else:
            dwell_time = str(last - first)

        # Row for CSVs
        info = "{}, {}, {}, {}, {}, {}, {}, {}\n".format(
            person_id,        # ID
            uid,              # uID
            first_seen_str,   # First Seen
            last_seen_str,    # Last Seen
            dwell_time,       # Dwell Time (TOTAL_TIME)
            age,
            gender,
            race
        )

        # Per-person info.csv
        info_csv = os.path.join(dir_path, "info.csv")
        with open(info_csv, "w+") as f:
            f.write("ID, uID, First Seen, Last Seen, Dwell Time, Age, Gender, Race\n")
            f.write(info)

        # Append to master db.csv
        with open(master_csv, "a") as f:
            f.write(info)


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
