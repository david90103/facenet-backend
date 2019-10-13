# 引用必要套件
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import os, sys

# 引用私密金鑰
# path/to/serviceAccount.json 請用自己存放的路徑
cred = credentials.Certificate('/home/chengzu/facenetServer/shi.json')

firebase_admin.initialize_app(cred, {
    'storageBucket': 'flutter-firebase-ffdb2.appspot.com'
})

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

    blob.download_to_filename(Pictures_dir + blob.name)  # Download



print('--------Done--------')