# 引用必要套件
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from firebase_admin import db
import os, sys, json, shutil

# path/to/serviceAccount.json 請用自己存放的路徑
cred = credentials.Certificate('/home/chengzu/facenetServer/shi.json')

firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://flutter-firebase-ffdb2.firebaseio.com/',
    'storageBucket': 'flutter-firebase-ffdb2.appspot.com'
})

ref = db.reference('classes')

#for key in ref.get(shallow=True):#shallow=True
#    if(ref.child(key).child("students").get(shallow=True)):
#        print(key+':')
#        for j in ref.child(key).child("students").get(shallow=True):
#            print(ref.child(key).child("students").child(j).child("nid").get())
#        print('----------------------')
#print('-----------DONE-----------')

dir = '/home/chengzu/facenetServer/image/'
src = '/home/chengzu/facenetServer/Pictures/'

for key in ref.get(shallow=True):#shallow=True
    if(ref.child(key).child("students").get(shallow=True)):
        print(key+':')
        dl_dir = dir + key
        print(dl_dir)
        if os.path.exists(dl_dir):
            shutil.rmtree(dl_dir)#刪除資料夾
        os.mkdir(dl_dir)
        for j in ref.child(key).child("students").get(shallow=True):
            #print(ref.child(key).child("students").child(j).child("nid").get())
            tempNid = ref.child(key).child("students").child(j).child("nid").get()
            tempDir = dl_dir + '/' + tempNid
            print(tempDir)
            if os.path.exists(tempDir):
                shutil.rmtree(tempDir)#刪除資料夾
            shutil.copytree(src + tempNid, tempDir)
        print('-----' + key + '-----')

#old
#for key in ref.get(shallow=True):
#    print(key+':')
#    dl_dir = dir + key
#    print(dl_dir)
#    if os.path.exists(dl_dir):
#        shutil.rmtree(dl_dir)#刪除資料夾
#    os.mkdir(dl_dir)
#    for j in ref.child(key).get(shallow=True):
#        if j != 0:
#            print(ref.child(key).child(j).get())
#            tempNid = ref.child(key).child(j).get()
#            tempDir = dl_dir + '/' + tempNid
#            shutil.copytree(src + tempNid, tempDir)
#    print('-----' + key + '-----')