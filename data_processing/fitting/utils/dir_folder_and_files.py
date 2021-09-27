import os
import shutil
from shutil import ignore_patterns

def make_dir(path):
    if(os.path.exists(path) == False):
        os.makedirs(path)

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def recursively_copy_folder(sourceDir,destDir,with_git=False,fromSafe=False):
    if not fromSafe:
        print("This is unsafe!!! Please use safely_recursively_copy_folder!!!")
        input()
    try:
        shutil.copytree(sourceDir, destDir,ignore=ignore_patterns('.git'))
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        print('Directory not copied. Error: %s' % e)
        # if e.errno == errno.ENOTDIR:
        #     shutil.copy(sourceDir, destDir)
        # else:
        #     print('Directory not copied. Error: %s' % e)
        return False
    return True
def safely_recursively_copy_folder(source_folder,traget_folder):
    if os.path.exists(traget_folder):
        print("[COPY] please delete config folder:",traget_folder)
        input()
        print("ok")
    if not recursively_copy_folder(source_folder,traget_folder,fromSafe=True):
        print("[COPY]warning!!! still cannot copy:",source_folder)