require 'image'

new_path = 'svhn_image.png'

img = image.load('cifar10_image.png')
new_img = image.load(new_path)

print(img:size())
print(new_img:size())

trans_img = image.scale(new_img, img:size(2), img:size(3), 'bicubic')
image.save(new_path, trans_img)

print("Image Convert Completed!")
print(img:size())
print(trans_img:size())
