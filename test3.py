# 引用必要套件
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage

# 引用私密金鑰
# path/to/serviceAccount.json 請用自己存放的路徑
cred = credentials.Certificate('C:/Users/User/Desktop/facenet/serviceUse.json')

firebase_admin.initialize_app(cred, {
    'storageBucket': 'quickstart-1553344776386.appspot.com'
})

# 初始化firestore
db = firestore.client()



bucket = storage.bucket()

blobs = bucket.list_blobs()
for blob in blobs:
   print(blob.name)

blob = bucket.blob('d0542001@mail.fcu.edu.tw/')

blob.download_to_filename('test.png')

print('--------Done--------')