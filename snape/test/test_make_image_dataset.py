
import shutil
from snape.make_image_dataset import *
from snape.make_image_dataset import _ImageNet, _ImageGrabber
from snape.utils import get_random_state
import glob
import pytest

conf = {
    "n_classes": 2,
    "n_samples": 11,
    "out_path": "./test_images/",
    "weights": [.8, .2],
    "image_source": "imagenet",
    "random_seed": 42
}

random_state = get_random_state(conf["random_seed"])


def test_make_image_dataset():
    os.mkdir(conf["out_path"])
    try:
        make_image_dataset(conf)

        subdirs = glob.glob(f'{conf["out_path"]}/*')
        print(f"Subdirs: {subdirs}")

        assert len(subdirs) == conf["n_classes"], \
            f'Expected {conf["n_classes"]} classes, but got {len(subdirs)}'

        n_samples = conf["n_samples"]
        weights = conf["weights"]
        exp_class_counts = sorted([int(w * n_samples) for w in weights])

        # sort subdir names by present images under each
        subdir_counts = sorted([len(glob.glob(f'{s}/*')) for s in subdirs])
        assert exp_class_counts == subdir_counts, \
            f"\nExpected class counts: {exp_class_counts}" \
            f"\nActual class counts: {subdir_counts}"

    finally:
        shutil.rmtree(conf["out_path"])


@pytest.mark.parametrize(
    'cfg', [

        # missing an arg
        {
            "n_samples": 11,
            "out_path": "./test_images/",
            "weights": [.8, .2],
            "image_source": "imagenet",
            "random_seed": 42
        },

        # wrong arg
        {
            "nclasses": 2,
            "n_samples": 11,
            "out_path": "./test_images/",
            "weights": [.8, .2],
            "image_source": "imagenet",
            "random_seed": 42
        }
    ]
)
def test_check_configuration(cfg):
    with pytest.raises(AssertionError):
        check_configuration(cfg)


class TestImageNet:
    def __init__(self):
        self.image_net = _ImageNet(n_classes=conf["n_classes"],
                                   weights=conf["weights"],
                                   n_samples=conf["n_samples"],
                                   output_dir=conf["out_path"],
                                   random_state=random_state)

    def test_get_images(self):
        os.mkdir(conf["out_path"])
        try:
            self.image_net.get_images()
            sub_dir1 = conf["out_path"] + os.listdir(conf["out_path"])[0]
            sub_dir2 = conf["out_path"] + os.listdir(conf["out_path"])[1]
            n_images1 = len(os.listdir(sub_dir1))
            n_images2 = len(os.listdir(sub_dir2))
            class1_size = int(conf["n_samples"] * conf["weights"][0])
            assert (class1_size == n_images1) or (class1_size == n_images2), "Did not download n images"
        except:
            raise
        finally:
            shutil.rmtree(conf["out_path"])

    def test_sample_synset_links(self):
        n = 5
        wnid = 'n02114855'
        os.mkdir(conf["out_path"])
        try:
            self.image_net.sample_synset_links(wnid, n, conf["out_path"])
            n_images = len(os.listdir(conf["out_path"] + '/' + wnid))
            assert n == n_images, "Did not download n images"
            assert wnid in os.listdir(conf["out_path"]), "Did not get the requested synset"
        except:
            raise
        finally:
            shutil.rmtree(conf["out_path"])

    def test_get_ilsvrc_1000_synsets(self):
        synsets = self.image_net.get_ilsvrc_1000_synsets()
        assert len(synsets) == 1000, "ILSVRC page parsed incorrectly"

    def test_get_synset_image_links(self):
        wnid = 'n02114855'
        links = self.image_net.get_synset_image_links(wnid)
        assert len(links) > 0, "Did not return any image links"

    def test_retrieve_class_counts(self):
        class_counts = self.image_net.retrieve_class_counts()
        assert isinstance(class_counts, pd.core.frame.DataFrame), "Class counts not returned in a dataframe"


class TestImageGrabber:

    def test_download_image(self):
        good_url = "http://farm4.static.flickr.com/3290/2998414960_01dd35d094.jpg"
        good_im_path = "ducky.jpg"
        _ImageGrabber().download_image(good_url, good_im_path)
        good_im_type = imghdr.what(good_im_path)
        os.remove(good_im_path)
        assert good_im_type is not None
        bad_url = "https://mckinleyleather.com/image/130963084.jpg"
        bad_im_path = "no_ducky.jpg"
        _ImageGrabber().download_image(bad_url, bad_im_path)
        is_file = os.path.isfile(bad_im_path)
        assert not is_file

    def test_catch_unavailable_image(self):
        good_url = "http://farm4.static.flickr.com/3290/2998414960_01dd35d094.jpg"
        good_img_data = requests.get(good_url)
        assert not _ImageGrabber.catch_unavailable_image(good_img_data), "The good image tested was found to be bad"
        stale_url = "https://mckinleyleather.com/image/130963084.jpg"
        stale_img_data = requests.get(stale_url)
        assert _ImageGrabber.catch_unavailable_image(stale_img_data), "The stale image tested was found to be good"
        junk_url = "http://farm4.static.flickr.com/3225/2806850016_9bf939037e.jpg"
        junk_img_data = requests.get(junk_url)
        assert _ImageGrabber.catch_unavailable_image(junk_img_data),  "The junk image tested was found to be good"


class TestOpenImages:
    pass


class TestGoogleSearch:
    pass
