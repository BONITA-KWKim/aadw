import os


def get_files(dir_: str, type:str="path") -> list:
  if type == "name":
    return [ x for _, _, files in os.walk(dir_) for x in files \
      if x.endswith("jpg") or x.endswith("png") ]
  elif type == "path":
    return [os.path.join(d, x) for d, _, files in os.walk(dir_) for x in files \
      if x.endswith("jpg") or x.endswith("png") ]


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
  dir_ = "/data/kwkim/dataset/bladder/testset_v1.0"

  files = get_files(dir_)
  print(len(files))
  dirs = get_dirs(dir_)
  print(dirs)

