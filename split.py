import os
import shutil
import random
import numpy as np
import math



os.makedirs('./data')
os.makedirs('./data/train')
os.makedirs('./data/test')

os.listdir('./data')

root_dir = './animals'

classes = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat", 
    "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog", 
    "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox", 
    "goat", "goldfish", "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog", 
    "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish", "kangaroo", 
    "koala", "ladybugs", "leopard", "lion", "lizard", "lobster", "mosquito", "moth", "mouse", 
    "octopus", "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda", "parrot", 
    "pelecaniformes", "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat", 
    "reindeer", "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake", 
    "sparrow", "squid", "squirrel", "starfish", "swan", "tiger", "turkey", "turtle", "whale", 
    "wolf", "wombat", "woodpecker", "zebra"
]

for clss in classes:
    print('------------' + clss + '-------------')
    dirtry = root_dir + '/' + clss
    files = os.listdir(dirtry)
    np.random.shuffle(files)

    base_outdir = './data/'

    for folder in ['train', 'test']:
        target_dir = base_outdir + folder
        os.makedirs(target_dir + '/' + clss)
        target_class = target_dir + '/' + clss

        if folder == 'train':
            images_to_pass = files[: math.floor(0.8*len(files))]
            for img in images_to_pass:
                img = dirtry + '/' + img
                shutil.copy(img, target_class)
        else:
            images_to_pass = files[math.floor(0.8*len(files)):]
            for img in images_to_pass:
                img = dirtry + '/' + img
                shutil.copy(img, target_class)

train_sum = 0
for animal in os.listdir('./data/train'):
    lnk = './data/train/' + animal
    train_sum += len(os.listdir(lnk))

test_sum = 0
for animal in os.listdir('./data/test'):
    lnk = './data/test/' + animal
    test_sum += len(os.listdir(lnk))

print(train_sum)
print(test_sum)