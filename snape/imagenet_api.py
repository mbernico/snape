
import imghdr
import os
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


def get_synset_image_links(wnid):
    """
    gets images from imagenet for a given synset
    :return:
    """
    url = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="
    url += wnid

    request = requests.get(url)

    link_list = request.text.split('\r\n')
    return link_list


def download_image(image_url, file_out):

    img_data = requests.get(image_url).content
    with open(file_out, 'wb') as handler:
        handler.write(img_data)

    # check if file is an image
    file_type = imghdr.what(file_out)

    # delete file if something is wrong
    if file_type is None:
        os.remove(file_out)


def retrieve_class_counts():

    request = requests.get("http://www.image-net.org/api/xml/ReleaseStatus.xml")

    soup = BeautifulSoup(request.text, "xml")

    row_list = []

    for synset in soup.findAll('synset'):
        row_list.append(synset.attrs)

    df = pd.DataFrame(row_list)
    df['numImages'] = pd.to_numeric(df['numImages'])

    # check = df[df['numImages'] > 100]
    return df


def get_ilsvrc_1000_synsets():

    request = requests.get("http://image-net.org/challenges/LSVRC/2014/browse-synsets")

    soup = BeautifulSoup(request.text)

    html_list = soup.findAll('a')

    wnid_list = []

    for h in html_list:
        url = h.attrs['href']
        if 'wnid=' in url:
            wnid_list.append(url[-9:])

    return wnid_list


def sample_synset_links(wnid, n, img_dir):

    img_links = get_synset_image_links(wnid)
    i = 0

    sub_dir = img_dir + wnid
    os.mkdir(sub_dir)

    while i < n:

        # pop a random sample
        pop_ix = np.random.choice(len(img_links), 1)[0]
        sam = img_links.pop(pop_ix)

        # try to download it
        file_name = img_dir + wnid + '/' + str(i) + '.jpg'

        try:
            download_image(sam, file_name)
        except:
            pass

        # repeat until # downloaded = n
        i = len(os.listdir(sub_dir))

        # need to add functionality for exiting if stuck in while loop

    #np.random.choice(im_links, n, replace = False)


def catch_unavailable_flicker_img():
    pass


##############################

synsets = get_ilsvrc_1000_synsets()

chosen_synsets = np.random.choice(synsets, 2, replace = False)

for syn in chosen_synsets:
    print(syn)
    print('sample:', sample_synset_links(syn, 5, '/Users/home/Desktop/'))

# need to do some EDA on the synset frequency counts
