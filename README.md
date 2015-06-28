```
ffmpeg -i test.mp4 -t 10 -vcodec png "images/test/%04d.png"
python train_2to1.py --gpu=2 conv3layer images/test
python convert.py --gpu=0 output/conv3layer_700001.dump  images/test result 1 10
```
