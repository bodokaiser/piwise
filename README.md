# PiWiSe

Pixel-wise segmentation on the [VOC2012][dataset] dataset using
[pytorch][pytorch].

- [x] [FCN](https://arxiv.org/abs/1605.06211)
- [x] [SegNet](https://arxiv.org/abs/1511.00561)
- [x] [PSPNet](https://arxiv.org/abs/1612.01105)
- [x] [UNet](https://arxiv.org/abs/1505.04597)
- [ ] [RefineNet](https://arxiv.org/abs/1611.06612)

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

## Usage

If you want to have visualization open an extra tab with:

```
python -m visdom.server -port 5000
```

Then start training:

```
python main.py --model basic --visualize --port 5000
```

or if you have a large CUDA card:

```
python main.py --model unet --cuda --visualize --port 5000
```

[pyenv]: https://github.com/pyenv/pyenv
[pytorch]: http://pytorch.org
[dataset]: http://host.robots.ox.ac.uk/pascal/VOC/
[dataset_example]: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples
[dataset_download]: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
