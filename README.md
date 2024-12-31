# Identification of Nanocomposites Agglomerates in Scanning Electron Microscopy Images

This repository contains the code and dataset used in the paper:

> Bai, Yu, Wang, Yan, Qiang, Dayuan, Yuan, Xin, Wu, Jiehui, Chen, Weilong, Zhang, Sai, Zhang, Yanru, and Chen, George.  
> "Identification of nanocomposites agglomerates in scanning electron microscopy images based on semantic segmentation."  
> *IET Nanodielectrics*, vol. 5, no. 2, pp. 93–103, 2022.  
> [DOI Link](https://doi.org/10.1049/nde2.12034) <!-- Replace with actual DOI if available -->

## Project Overview

This project focuses on identifying nanocomposite agglomerates in scanning electron microscopy (SEM) images using semantic segmentation techniques. The repository includes:

- A dataset of SEM images.
- Implementations of three semantic segmentation methods described in the paper.

## Repository Structure

```
Nano/
│
├── fcn/               # Implementation of Fully Convolutional Networks (FCN)
├── pixel/             # Pixel-based semantic segmentation method
├── SEM dataset/       # SEM image dataset
├── unsupervised/      # Unsupervised semantic segmentation method
│
├── .gitignore         # Specifies files to be ignored by Git
├── LICENSE            # License file for the project
└── README.md          # Project documentation
```

## Methods

#### Fully Convolutional Networks (FCN)
This folder contains the implementation of Fully Convolutional Networks (FCN) used for semantic segmentation in SEM images. The code is based on the implementation from [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch) with basic modifications for this project.

#### Pixel-based Method
[//]: # (#### Execution Steps for `pixel/`)
1. **Cutting Image Patches**:
   Execute `cutpatches/cut.py` to split the original training images into smaller pixel patches. Customizable parameters include the image path and save path. **Note**: This step generates a large amount of data (tens of gigabytes).

2. **Training the Model**:
   Run the training program `train721` to train the model. Training parameters can be modified in `config.param.py`.

3. **Prediction**:
   Use `predictor` to predict results. Parameters include the path to the prediction images, along with corresponding settings modified in the `config` file.

#### Unsupervised Method
### 3. `unsupervised/`
This folder provides the code for the unsupervised semantic segmentation method. The code is adapted from [Unsupervised-Segmentation](https://github.com/Yonv1943/Unsupervised-Segmentation/tree/master) with modifications to fit the specific requirements of this project.
```bash
cd unsupervised
python unsupervised_segmentation.py
```

[//]: # ()
[//]: # (### Results)

[//]: # (The output for each method will be saved in their respective directories. Evaluation metrics and visualized segmentations are provided.)

## Citation

If you find this code or dataset helpful, please cite our work:
```bibtex
@article{bai2022identification,
  title={Identification of nanocomposites agglomerates in scanning electron microscopy images based on semantic segmentation},
  author={Bai, Yu and Wang, Yan and Qiang, Dayuan and Yuan, Xin and Wu, Jiehui and Chen, Weilong and Zhang, Sai and Zhang, Yanru and Chen, George},
  journal={IET Nanodielectrics},
  volume={5},
  number={2},
  pages={93--103},
  year={2022},
  publisher={Wiley Online Library}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

[//]: # (## Contact)

[//]: # ()
[//]: # (For questions or issues, please contact [Your Email or GitHub Issues Page].)

[//]: # ()
[//]: # (---)

We hope this repository serves as a valuable resource for researchers and practitioners working on SEM image analysis and semantic segmentation.
