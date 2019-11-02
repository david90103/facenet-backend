# 引用必要套件
import firebase_admin
import flask
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage

from flask_apscheduler import APScheduler
from flask import Flask

import os, sys
#https://www.cnblogs.com/huchong/p/9088611.html#_lab2_6_1

#cron 定時
#year (int|str) – 4-digit year
#month (int|str) – month (1-12)
#day (int|str) – day of the (1-31)
#week (int|str) – ISO week (1-53)
#day_of_week (int|str) – number or name of weekday (0-6 or mon,tue,wed,thu,fri,sat,sun)
#hour (int|str) – hour (0-23)
#minute (int|str) – minute (0-59)
#second (int|str) – second (0-59)
#start_date (datetime|str) – earliest possible date/time to trigger on (inclusive)
#end_date (datetime|str) – latest possible date/time to trigger on (inclusive)
#timezone (datetime.tzinfo|str) – time zone to use for the date/time calculations (defaults to scheduler timezone)

#interval 間隔
#weeks (int) – number of weeks to wait
#days (int) – number of days to wait
#hours (int) – number of hours to wait
#minutes (int) – number of minutes to wait
#seconds (int) – number of seconds to wait
#start_date (datetime|str) – starting point for the interval calculation
#end_date (datetime|str) – latest possible date/time to trigger on
#timezone (datetime.tzinfo|str) – time zone to use for the date/time calculations

class Config(object):
    JOBS=[
        {
            'id':'job1',
            'func':'__main__:job_1',
            'args':('shi',),
            'trigger':'interval',
            'minutes': 5
            #'trigger':'cron',
            #'hour':17,
            #'minute':8
        }
#        {
#            'id':'job2',
#            'func':'__main__:getPictures',
#            'args':(),
#            'trigger':'interval',
#            'minutes':5
#        }
    ]

def job_1(arg1):   # 一個函式，用來做定時任務的任務。
    os.system("python " + arg1 + ".py")

def getPictures():
    # 初始化firestore
    db = firestore.client()

    bucket = storage.bucket()
    Pictures_dir = '/home/chengzu/facenetServer/Pictures/'

    blobs = bucket.list_blobs()

    for blob in blobs:
        #print(blob.name)
        #print(Pictures_dir + blob.name.split("/",1)[0])
        if not os.path.exists(Pictures_dir + blob.name.split("/",1)[0]):
            os.mkdir(Pictures_dir + blob.name.split("/",1)[0])

        print(Pictures_dir + blob.name)
        blob.download_to_filename(Pictures_dir + blob.name)  # Download

    print('--------Done--------')

# 引用私密金鑰
# path/to/serviceAccount.json 請用自己存放的路徑
cred = credentials.Certificate('/home/chengzu/facenetServer/shi.json')

firebase_admin.initialize_app(cred, {
    'storageBucket': 'flutter-firebase-ffdb2.appspot.com'
})

app = flask.Flask(__name__) # 例項化flask

app.config.from_object(Config())# 為例項化的flask引入配置

@app.route('/')  # 首頁路由
def hello_world():
    return 'hello'


if __name__=='__main__':
    scheduler=APScheduler()  # 例項化APScheduler
    scheduler.init_app(app)  # 把任務列表放進flask
    scheduler.start() # 啟動任務列表
    app.run(host='0.0.0.0', threaded=True)  # 啟動flask