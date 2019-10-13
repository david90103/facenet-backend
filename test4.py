# 引用必要套件
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import os, sys

# 引用私密金鑰
# path/to/serviceAccount.json 請用自己存放的路徑
cred = credentials.Certificate('/home/chengzu/facenetServer/serviceUse.json')

firebase_admin.initialize_app(cred, {
    'storageBucket': 'quickstart-1553344776386.appspot.com'
})

# 初始化firestore
db = firestore.client()

bucket = storage.bucket()
prefix = 'd0542001@mail.fcu.edu.tw/'
dl_dir = '/home/chengzu/facenetServer/'

blobs = bucket.list_blobs()
#blobs = bucket.list_blobs(prefix=prefix)  # Get list of files

for blob in blobs:
    print(blob.name)
    if blob.name[-1] == '/':
        if not os.path.exists(dl_dir + blob.name):
            os.mkdir(dl_dir + blob.name)
    else:
        blob.download_to_filename(dl_dir + blob.name)  # Download

#    filename = blob.name.replace('/', '_') 
#    blob.download_to_filename(dl_dir + filename)  # Download

#for blob in blobs:
#    blob_Name.append(blob.name)
#    print(blob.name)
#    blob = bucket.blob(blob.name)
#    blob.download_to_filename(blob.name)

#blob = bucket.blob()

#blob.download_to_filename('test1')

print('--------Done--------')