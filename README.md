# Layered-Garment Net

This repository contains a pytorch implementation (training and testing) of LGN

## Demo
This demo runs for sample dataset given in `sample/data/` folder. Refer the folder and source code to run on custom data.
1. Download trained model from () and save in home folder of the project.
2. run the following script:
```
python -m apps.update
```
The output will be saved in `sample/data/` folder. The resulting mesh are watertight and can be visualized using meshlab.
3. To trim the generated mesh, run the following script:
```
python -m apps.trim
```