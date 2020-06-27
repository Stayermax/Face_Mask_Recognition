import os
import shutil
# print(len(os.listdir('train/')))
def sort_data():
    for i, img_name in enumerate(os.listdir('train/')):
        if(i%100==0):
            print(f'DONE {i} out of {len(os.listdir("train/"))}')
        if('_0.' in img_name):
            shutil.copy(f'train/{img_name}', 'train_sorted/no_mask/')
        else:
            shutil.copy(f'train/{img_name}', 'train_sorted/mask/')

sort_data()

print(f"MASK:  {len(os.listdir('train_sorted/mask/'))}")
print(f"NO MASK:  {len(os.listdir('train_sorted/no_mask/'))}")