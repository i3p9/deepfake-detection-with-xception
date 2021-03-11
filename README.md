# deepfake-detection-with-xception

Steps:
- Grab the required packages from requirements.txt using pip

prepare and train Dataset:
- We have used the Kaggle Deepfake Challange dataset, Link: https://www.kaggle.com/c/deepfake-detection-challenge/data

- Download the dataset and from `train_sample_videos` folder, extract faces from the videos. Put them in the corresponding folders inside `dataset`. Classes are predefined already.

- Run `train_dataset.py` to train and generate models.

```bash
python3.py train_dataset.py dataset/ classes.txt  result/
```

Predefined settings:
```bash
Epoch: 10 / 30 [First/Final stage]
Learning rate: 5e-3 / 5e-4
Batch size: 32 / 64
```

- Then take the best model from examining the graph and run `app.py` to detect videos. It can take a video file or a youtube-dl supported video link as a input. Note that we've tested online links only with Youtube so your results may vary.


Note:

There's also a basic image predictor which takes a LOT less time compared to a video. Use `image_prediction.py`

```bash
python3 image_prediction.py path_to_model.p classes.txt input_image.jpg
```


Credits:
```
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
```
