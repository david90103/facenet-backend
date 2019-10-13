# coding:utf-8
import os, sys, json, shutil



# 需要複製檔案的資料夾位置
#give_pos = "/home/chengzu/facenetServer/d0542001@mail.fcu.edu.tw"
# 需要複製到的位置
#des_pos = "/home/chengzu/test"

#if not os.path.exists(unicode(des_pos, 'utf-8')):
#    os.mkdir(unicode(des_pos, "utf-8"))
# 移動檔案
#move_all_files(unicode(give_pos, "utf-8"))


#A資料夾裡都是xml文件，B資料夾是一个空資料夾，C資料夾裡面都是jpg檔，现在要在Ａ資料夾中挑選出含有C資料夾里jpg檔名字的xml檔保存到B中。（例如Ｃ中含有888.jpg文件，要在Ａ中挑选出888.xml保存到B）
#dir1 = ""#C資料夾
#dir2 = "/home/chengzu/facenetServer/"#A資料夾
#for root1, dirs1, file1 in os.walk(dir1):
#    for a in file1:
#        for root2, dirs2, file2 in os.walk(dir2):
#            for b in file2:
#                if os.path.join(a).split('.')[0] == os.path.join(b).split('.')[0]:
#                    print os.path.join(dir2,b)
#                    shutil.copy(os.path.join(dir2,b), r'B文件夹路径')

#dir1 = '/home/chengzu/facenetServer/d0542001@mail.fcu.edu.tw'
#shutil.copytree(dir1,'/home/chengzu/facenetServer/test/d0542001@mail.fcu.edu.tw')＃移動資料夾

#shutil.rmtree('/home/chengzu/facenetServer/test/')#刪除資料夾