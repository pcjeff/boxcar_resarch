

unwant_list = ['./crop1.txt', './crop2.txt', './crop3.txt', './crop4.txt']
unwant_imgs = []
crop_imgs = []

for filename in unwant_list:
    with open(filename, 'r') as f:
        for line in f:
            unwant_imgs.append(line.strip())

with open('/tmp3/pcjeff/boxcar/crop_all.txt', 'r') as f:
    for line in f:
        crop_imgs.append(line.strip())

for img in set(crop_imgs) - set(unwant_imgs):
    print img
