
import imghdr
import os
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from snape import flicker


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

    if catch_unavailable_img(img_data):
        pass

    else:
        with open(file_out, 'wb') as handler:
            handler.write(img_data)

        file_type = imghdr.what(file_out)

        if file_type is None:
            os.remove(file_out)
        else:
            print(image_url)


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

    soup = BeautifulSoup(request.text, "html.parser")

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

    # np.random.choice(im_links, n, replace = False)


def catch_unavailable_img(img_data):

    im1_check = img_data == flicker.junk_image1
    im2_check = img_data == flicker.junk_image2

    is_it_junk = im1_check or im2_check
    return is_it_junk


def sample_imagenet():
    pass
