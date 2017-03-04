import urllib2  # the lib that handles the url stuff

# mammals
parent_synset = 'n01861778'

target_url = 'http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=' + parent_synset

data = urllib2.urlopen(target_url)  # it's a file like object and works just like a file
for line in data:  # files are iterable
    print line
