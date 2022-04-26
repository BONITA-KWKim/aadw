import os

dir_ = '/data/kwkim/aadw/dataset/colpo/train'
for (dirpath, dirnames, filenames) in os.walk(dir_):
  for idx, x in enumerate(filenames):
    no = '{0:04d}'.format(idx)
    dirname = dirpath.split('/')[-1]
    print(f'dirname: {dirname}')
    os.rename(os.path.join(dirpath, x), os.path.join(dirpath, f'{dirname}_{no}.png'))

