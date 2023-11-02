### The code of "CAMF: An Interpretable Infrared and Visible Image Fusion Network Based on Class Activation Mapping"

#### requirement:
tensorflow-gpu 1.15  
opencv-python  
Pillow  
scipy 1.2.1  

#### Test:
Download the pre-trained checkpoint from [here](https://drive.google.com/file/d/1eECkhVdIoJoSEbGkx1-YNo6_XatGS01-/view?usp=sharing) and put them in ./checkpoint/

Run `python test_one_image.py` to test. The test_dataset can be set as 'tno', 'roadscene' or 'medical'.

#### Train:
The auto-encoder is trained using the MS-COCO dataset.  

You can download my training set from [here](https://drive.google.com/file/d/1BJ2UAE1_eS2xnFE0a-JAvxPhO_7Oe_3H/view?usp=sharing) or download the original data from the [official website](https://cocodataset.org/#home).

Run `python train_auto_encoder.py` to train the auto-encoder network.

Run `python train_classifier.py` to train the classifier network.


If this work is helpful to you, please cite it as: 
```
@article{tang2023camf,
  title={CAMF: An Interpretable Infrared and Visible Image Fusion Network Based on Class Activation Mapping},
  author={Tang, Linfeng and Chen, Ziang and Huang, Jun and Ma, Jiayi},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}
```