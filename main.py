################################
#
# RUN
# cmd + I  (run the entire python)
# shift + Enter (run hydogen)
#
# VIRTUAL ENV
# virtualenv remi_bioimage_package
# source activate remi_insight
# source deactivate
#
# INSTALLATION (main dir)
# python setup.py sdist
# pip install . --upgrade
#
# SPHINX
# sphinx-build -b html . ../docs
#
# TEST
# 1.  go to the folder image_features_extraction or tests
# 2.  run  pytest  on the terminel
#
################################



#!py.test tests


import image_features_extraction.Images as im

imgs = im.Images('./images')
print('numberf of images:{}'.format(imgs.count()))
img1 = imgs.item(1)

print('file names:')
for img in imgs:
    print(img.file_name())


regs = img1.regions()
print('numberf of regions:{}'.format(regs.count()))

print('features:')
features = regs.get_features(['label', 'area','perimeter', 'centroid'], class_value=1)
print(features.get_dataframe())


#for reg in regs:
#    areas.append(reg.perimeter)

#import matplotlib.pyplot as plt
#plt.plot(areas)
#plt.show()
