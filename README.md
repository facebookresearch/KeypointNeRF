# KeypointNeRF
The official public release of KeypointNeRF: Generalizing Image-based Volumetric Avatars using Relative Spatial Encoding of Keypoints

## Installation 
Please install python dependencies specified in `environment.yml`:
```bash
conda env create -f environment.yml
conda activate KeypointNeRF
```

## Data preparation
Please follow the instructions provided by the  [**NHP**](https://github.com/markomih/NHP_fork_data) to download the preprocessed dataset and store it under the `${ZJU_DATA}` such that it follows the structure:
```bash
${ZJU_DATA}
├── CoreView_313
├── CoreView_315
├── CoreView_377
├── CoreView_386
├── CoreView_387
├── CoreView_390
├── CoreView_392
├── CoreView_393
├── CoreView_394
└── CoreView_396
```

## Train your own model
Execute `train.py` script to train the model on the ZJU dataset.
```shell script
python train.py --config ./configs/zju.json --data_root $ZJU_DATA
```
To extract the evaluation images execute:
```shell script
python train.py --config ./configs/zju.json --data_root $ZJU_DATA --run_val
```
Then, to compute the evaluation metrics run:
```shell script
python eval_zju.py --src_dir ./EXPERIMENTS/zju/images_v3
```


## Publication
If you find our code or paper useful, please consider citing:
```bibtex
@InProceedings{Mihajlovic:ECCV:22,
  title = {{KeypointNeRF}: Generalizing Image-based Volumetric Avatars using Relative Spatial Encoding of Keypoints},
  author = {Mihajlovic, Marko and Bansal, Aayush and Zollhoefer, Michael and Tang, Siyu and Saito, Shunsuke},
  booktitle = {ECCV},
  year = {2022},
}
```
