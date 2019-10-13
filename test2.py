# 引用必要套件
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from firebase_admin import db
import os, sys, json
#import importlib
#PYTHONIOENCODING=utf-8
#importlib.reload(sys)
#export PYTHONIOENCODING=utf-8
#unset PYTHONIOENCODING
print(sys.stdout.encoding)
# 引用私密金鑰
# path/to/serviceAccount.json 請用自己存放的路徑
cred = credentials.Certificate('/home/chengzu/facenetServer/serviceUse.json')

firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://quickstart-1553344776386.firebaseio.com/',
    'storageBucket': 'quickstart-1553344776386.appspot.com'
})

#ref = db.reference('stu')
#print(ref.get())
#ref = db.reference('stu')
#for i in range(len(ref.get())):
#    if i > 0:
#        print(ref.child(str(i)).child('nid').get())
#snapshot = ref.child('1').child('nid').get()

#ref = db.reference('thr')
#for i in range(len(ref.get())):
#    if i > 0:
#        for j in range(len(ref.child(str(i)).child('class').get())):
#            if j > 0:
#                print(ref.child(str(i)).child('class').child(str(j)).child('number').get())

#print(ref.child('1').child('class').get())

ref = db.reference('thr')
print(ref.get())

ref = db.reference('classno')
#for i in range(len(ref.get())):

print(ref.get(shallow=True))
for key in ref.get(shallow=True):
    print(key+':')
    for j in ref.child(key).get(shallow=True):
        if j != 0:
            print(ref.child(key).child(j).get())
    print('----------------------')

json_object = json.dumps(ref.get(shallow=True))
print(json_object + str(type(json_object)))

#print(snapshot)
#for key in snapshot:
#    print(key)

#print(snapshot)

print('--------Done--------')