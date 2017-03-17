#############################################################################
#
#
# The snape.make_image_dataset module provides functionality for downloading
# a unique image dataset. It adapts the same interface as snape.make_dataset,
# such that a user inputs their desired configuration in either the form of a
# dictionary -for calling from python, or json -for calling via command line.
#
#
#############################################################################

import imghdr
import os
import requests
import pandas as pd
import sys
from bs4 import BeautifulSoup
from snape import flicker
from snape.make_dataset import parse_args, load_config
from snape.utils import get_random_state


def make_image_dataset(config=None):
    if config is None:
        # called from the command line so parse configuration
        args = parse_args(sys.argv[1:])
        config = load_config(args['config'])

    random_state = get_random_state(config["random_seed"])

    if config["image_source"] == "imagenet":
        _ImageNet(n_classes=config["n_classes"],
                  weights=config["weights"],
                  n_samples=config["n_samples"],
                  output_dir=config["out_path"],
                  random_state=random_state).get_images()

    elif config["image_source"] == "openimages":
        print("Not yet supported. The only image_source currently supported is 'imagenet'")

    elif config["image_source"] == "googlesearch":
        print("Not yet supported. The only image_source currently supported is 'imagenet'")

    else:
        print(config["image_source"], "is not a supported image_source")
        print("The only image_source currently supported is 'imagenet'")


def check_configuration(conf):
    # todo: check values assigned to each config key
    expected_conf_args = ["n_classes", "n_samples", "out_path", "weights", "image_source", "random_seed"]
    for key in conf.keys():
        assert key in expected_conf_args, key + " is not an allowed configuration argument"
    for key in expected_conf_args:
        assert key in conf.keys(), key + " was not specified in the configuration"


class _ImageNet:

    # todo: prescreen image links for junk
    # todo: precompute available synsets
    # todo: return class labels
    def __init__(self, n_classes, weights, n_samples, output_dir, random_state=None):
        self.ilsvrc_synsets = self.get_ilsvrc_1000_synsets()
        self.random_state = get_random_state(random_state)
        self.chosen_synsets = self.random_state.choice(self.ilsvrc_synsets, n_classes, replace=False)
        self.n_samples = n_samples
        self.output_dir = output_dir
        self.weights = weights

    def get_images(self):
        for i, syn in enumerate(self.chosen_synsets):
            print(syn)
            n = int(self.n_samples * self.weights[i])
            self.sample_synset_links(syn, n, self.output_dir)

    def sample_synset_links(self, wnid, n, img_dir):
        img_links = self.get_synset_image_links(wnid)
        i = 0
        sub_dir = img_dir + wnid
        os.mkdir(sub_dir)
        while i < n:
            pop_ix = self.random_state.choice(len(img_links), 1)[0]
            sam = img_links.pop(pop_ix)
            file_name = img_dir + wnid + '/' + str(i) + '.jpg'
            try:
                _ImageGrabber().download_image(sam, file_name)
            except:
                pass
            i = len(os.listdir(sub_dir))
            if len(img_links) == 0:
                break
            # todo: add more functionality for exiting if stuck in while loop

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


class _ImageGrabber:

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


class _OpenImages:
    # todo: build this class for scraping the OpenImages dataset
    pass


class _GoogleSearch:
    # todo: build this class for scraping the google api
    pass

# todo: embed each set of class labels in word vectors & pre-compute similarity matrix

if __name__ == "__main__":
    make_image_dataset()
