# PiWiSe

Pixel-wise segmentation on the [VOC2012][dataset] dataset using
[pytorch][pytorch].

## Setup

See dataset examples [here][dataset_example].

### Download

Download [image archive][dataset_download] and extract and do:

```
mkdir data
mv VOCdevkit/VOC2012/JPEGImages data/images
mv VOCdevkit/VOC2012/SegmentationClass data/classes
rm -rf VOCdevkit
```

### Install

We recommend using [pyenv][pyenv]:

```
pyenv virtualenv 3.6.0 piwise 
pyenv activate piwise
```

then install requirements with `pip install -r requirements.txt`.

[pyenv]: https://github.com/pyenv/pyenv
[pytorch]: http://pytorch.org
[dataset]: http://host.robots.ox.ac.uk/pascal/VOC/
[dataset_example]: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples
[dataset_download]: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
