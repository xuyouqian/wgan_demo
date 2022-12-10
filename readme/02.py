import glob

root_path = '../data/faces'

pathes = glob.glob(root_path + '/*')
print(pathes[:3])
