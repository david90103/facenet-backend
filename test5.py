# 引用必要套件
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from firebase_admin import db
import os, sys, json

# 引用私密金鑰
# path/to/serviceAccount.json 請用自己存放的路徑
cred = credentials.Certificate('/home/chengzu/facenetServer/serviceUse.json')

firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://quickstart-1553344776386.firebaseio.com/',
    'storageBucket': 'quickstart-1553344776386.appspot.com'
})

ref = db.reference('classno')

for key in ref.get(shallow=True):
    print(key+':')
    for j in ref.child(key).get(shallow=True):
        if j != 0:
            print(ref.child(key).child(j).get())
    print('----------------------')


print('--------Done--------')


#shutil.copyfile("old","new") #複製檔案，只能是檔案
#shutile.copytree('old','new') #複製資料夾，都只能是目錄，且new不存在
#shutile.copy('old','new') #複製檔案/資料夾，複製 old 為 new（new是檔案，若不存在，即新建），複製 old 為至 new 資料夾（資料夾已存在）
#shutil.move('old','new') #移動檔案/資料夾至new