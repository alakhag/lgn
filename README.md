# Layered-Garment Net

![](lgn.jpg)

This repository contains a pytorch implementation (training and testing) of
> **Layered-Garment Net: Generating Multiple Implicit Garment Layers from a Single Image**
>
> Alakh Aggarwal, 
> Jikai Wang, 
> Steven Hogue, 
> Saifeng Ni, 
> Madhukar Budagavi,
> Xiaohu Guo
>
> ACCV 2022
>
> [[Paper]](https://personal.utdallas.edu/~xguo/ACCV2022.pdf)

## Demo
This demo runs for sample dataset given in `sample/data/` folder. Refer the folder and source code to run on custom data.
1. Download trained model from [checkpoints](https://utdallas.box.com/s/s5wm6uwh648xw0lq3f2wddng1gfa8lo8) and save in home folder of the project by name `checkpoints`.
2. run the following script:
```
python -m apps.update
```
The output will be saved in `sample/data/` folder. The resulting mesh are watertight and can be visualized using meshlab.
3. To trim the generated mesh, run the following script:
```
python -m apps.trim
```

## Citation
```
@inproceedings{aggarwal2022layered,
  title={Layered-Garment Net: Generating Multiple Implicit Garment Layers from a Single Image},
  author={Aggarwal, Alakh and Wang, Jikai and Hogue, Steven and Ni, Saifeng and Budagavi, Madhukar and Guo, Xiaohu},
  booktitle={Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year={2022}
}
```
