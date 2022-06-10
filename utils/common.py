import os
import json
import re


def get_results(dir_: str)->list:
  image_list = list()

  if os.path.exists(dir_) is False:
    return image_list
    
  with open(os.path.join(dir_, "image_list.txt"), "r") as f:
    raw = json.loads(f.read())
  image_list = raw["images"]
  return image_list


def remove_exception(list_):
  return [ item for item in list_ if re.search(r'^thumbnail+?', item) is None ]


def get_files(dir_: str, type:str="path") -> list:
  if type == "name":
    list_ = [ x for _, _, files in os.walk(dir_) for x in files \
      if x.endswith("jpg") or x.endswith("png") ]
    list_ = remove_exception(list_)
  elif type == "path":
    list_ = [os.path.join(d, x) for d, _, files in os.walk(dir_) for x in files \
      if x.endswith("jpg") or x.endswith("png") ]
    list_ = remove_exception(list_)
  elif type == "both":
    list_ = list()
    for d, _, files in os.walk(dir_):
      for x in files:
        if x.endswith("jpg") or x.endswith("png"):
          if re.search(r'^thumbnail+?', x) is None:
            element = {"name": x, "path": os.path.join(d, x)}
            list_.append(element)
  return list_


def get_dirs(dir_: str) -> list:
  dirs = [s for _, s, _ in os.walk(dir_)]
  if 0 < len(dirs):
    return dirs[0]
  else:
    return []


def create_output_directory(dir_:str):
  if not os.path.isdir(dir_):
    os.makedirs(dir_)


if __name__=="__main__":
  # dir_ = "/data/kwkim/dataset/bladder/testset_v1.0"
  dir_ = "/data/kwkim/dataset/bladder/test_patches"
  
  files = get_files(dir_, "path")
  print(f'[D] length: {len(files)}\n\tfiles: {files}')
  # dirs = get_dirs(dir_)
  # print(dirs)
