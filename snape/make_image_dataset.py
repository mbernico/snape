
import imghdr
import os
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from snape import flicker


class ImageNet:

    def __init__(self, n_classes):
        self.ilsvrc_synsets = self.get_ilsvrc_1000_synsets()
        self.chosen_synsets = np.random.choice(self.ilsvrc_synsets, n_classes, replace=False)

    def sample_synset_links(self, wnid, n, img_dir):
        img_links = self.get_synset_image_links(wnid)
        i = 0
        sub_dir = img_dir + wnid
        os.mkdir(sub_dir)
        while i < n:
            pop_ix = np.random.choice(len(img_links), 1)[0]
            sam = img_links.pop(pop_ix)
            file_name = img_dir + wnid + '/' + str(i) + '.jpg'
            try:
                ImageGrabber().download_image(sam, file_name)
            except:
                pass
            i = len(os.listdir(sub_dir))
            if len(img_links) == 0:
                break
            # need to add functionality for exiting if stuck in while loop

    def get_images(self, n_samples, output_dir):
        for syn in self.chosen_synsets:
            print(syn)
            self.sample_synset_links(syn, n_samples, output_dir)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def retrieve_class_counts():
        request = requests.get("http://www.image-net.org/api/xml/ReleaseStatus.xml")
        soup = BeautifulSoup(request.text, "xml")
        row_list = []
        for synset in soup.findAll('synset'):
            row_list.append(synset.attrs)
        df = pd.DataFrame(row_list)
        df['numImages'] = pd.to_numeric(df['numImages'])
        return df


class ImageGrabber:

    def download_image(self, image_url, file_out):
        img_data = requests.get(image_url)
        if self.catch_unavailable_image(img_data):
            pass
        else:
            with open(file_out, 'wb') as handler:
                handler.write(img_data.content)
            file_type = imghdr.what(file_out)
            if file_type is None:
                os.remove(file_out)
            else:
                print(image_url)

    @staticmethod
    def catch_unavailable_image(img_data):
        not_an_image = 'image' not in img_data.headers['Content-Type']
        im1_check = img_data.content == flicker.junk_image1
        im2_check = img_data.content == flicker.junk_image2
        is_it_junk = not_an_image or im1_check or im2_check
        return is_it_junk


class OpenImages:
    pass


class GoogleSearch:
    pass
