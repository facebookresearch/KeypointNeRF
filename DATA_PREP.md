## Data preparation

1. Setup the ZJU-Mocap dataset: Fill in the [agreement](https://zjueducn-my.sharepoint.com/:b:/g/personal/pengsida_zju_edu_cn/EUPiybrcFeNEhdQROx4-LNEBm4lzLxDwkk1SBcNWFgeplA?e=BGDiQh) and email Peng Sida (pengsida@zju.edu.cn) and cc Xiaowei Zhou (xwzhou@zju.edu.cn) to request the download link.

2. Create a soft link:
```bash
mkdir ./data
ln -s /path/to/zju_mocap ./data/zju_mocap
```

3. Download the precalculated 3D joints from [here](https://drive.google.com/file/d/1ZA_5KcprqHAgyT5r73EX4JF6W9GHGi07/view?usp=sharing) and extract them into the respective directories:
```bash
unzip zju_joints3d.zip
cp -r ./zju_joints3d/* ./data/zju_mocap/
```

4. Execute the following script to make the image names consistent for all subjects:
```bash
python preprocess/rename_zju.py --data_dir ./data/zju_mocap
```

After this step the data directory follows the structure:
```bash
${./data/zju_mocap}
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
