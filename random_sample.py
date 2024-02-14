import glob 
import numpy as np
import os 
import shutil 

source = glob.glob('./images/*')

labels_dict = {
    'alb_id':0,
    'aze_passport':1,
    'esp_id':2,
    'est_id':3,
    'fin_id':4,
    'grc_passport':5,
    'lva_passport':6,
    'rus_internalpassport':7,
    'srb_passport':8,
    'svk_id':9,
}
dataset = 'datasets/classify_initial'
for classes in source:
    images = glob.glob(classes + '/*')
    label = labels_dict[os.path.basename(classes)]
    rnd = np.random.choice(images, 50, replace=False)
    print(classes)
    print(f'train images: {len(rnd)}')
    os.makedirs(f'./train/{os.path.basename(classes)}', exist_ok=True)
    for r in rnd:
        if not os.path.exists(f'./{dataset}/train/{os.path.basename(classes)}/images'):
            os.makedirs(f'./{dataset}/train/{os.path.basename(classes)}/images')
        if not os.path.exists(f'./{dataset}/train/{os.path.basename(classes)}/labels'):
            os.makedirs(f'./{dataset}/train/{os.path.basename(classes)}/labels')
        shutil.copy(r, f'./{dataset}/train/{os.path.basename(classes)}/images/{os.path.basename(r)}')
        with open(f'./{dataset}/train/{os.path.basename(classes)}/labels/{os.path.basename(r).split(".")[0]}.txt', 'w') as file:
            file.write(str(label))

    remaining_images = list(set(images) - set(rnd))
    rnd = np.random.choice(remaining_images, 25, replace=False)
    print(f'val images: {len(rnd)}')
    for r in rnd:
        if not os.path.exists(f'./{dataset}/val/{os.path.basename(classes)}/images'):
            os.makedirs(f'./{dataset}/val/{os.path.basename(classes)}/images')
        if not os.path.exists(f'./{dataset}/val/{os.path.basename(classes)}/labels'):
            os.makedirs(f'./{dataset}/val/{os.path.basename(classes)}/labels')
        shutil.copy(r, f'./{dataset}/val/{os.path.basename(classes)}/images/{os.path.basename(r)}')
        with open(f'./{dataset}/val/{os.path.basename(classes)}/labels/{os.path.basename(r).split(".")[0]}.txt', 'w') as file:
            file.write(str(label))
    
    remaining_images = list(set(remaining_images) - set(rnd))
    rnd = remaining_images
    print(f'test images: {len(rnd)}')
    for r in rnd:
        if not os.path.exists(f'./{dataset}/test/{os.path.basename(classes)}/images'):
            os.makedirs(f'./{dataset}/test/{os.path.basename(classes)}/images')
        if not os.path.exists(f'./{dataset}/test/{os.path.basename(classes)}/labels'):
            os.makedirs(f'./{dataset}/test/{os.path.basename(classes)}/labels')
        shutil.copy(r, f'./{dataset}/test/{os.path.basename(classes)}/images/{os.path.basename(r)}')
        with open(f'./{dataset}/test/{os.path.basename(classes)}/labels/{os.path.basename(r).split(".")[0]}.txt', 'w') as file:
            file.write(str(label))    
