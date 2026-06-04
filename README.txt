Title: Expert-annotated optical microscopy images of human sperm for detection and DNA fragmentation assessment

Dataset Description
This dataset consists of optical bright-field images of human sperm prepared using the sperm chromatin dispersion (SCD) method. It is divided into three main parts:

Binary Classification – cropped images labeled as sperm or non-sperm.

Halo Classification – cropped sperm images categorized by halo size (large halo, medium halo, small halo, without halo) and a separate non-sperm category.

Raw Images – high-resolution full-field images containing multiple sperm cells and non-sperm entities.

All annotations were performed independently by five experienced embryologists, with final labels assigned using majority voting to ensure quality and consistency.

File Naming Convention

All files are named sequentially with zero-padded integers (e.g., 001.png, 002.png, 003.png).

Each subfolder restarts numbering independently.

Metadata File
The Metadata.csv file provides structured information about each image, with the following columns:

filename – image file name (e.g., 001.png)

subset – Binary_Classification, Halo_Classification, or Raw_Images

class – sperm / non-sperm

halo_category – large, medium, small, without halo (if applicable; “-” otherwise)

Usage Notes

The Binary_Classification subset is intended for sperm detection tasks.

The Halo_Classification subset supports DNA fragmentation studies using halo size as a proxy for chromatin integrity.

The Raw_Images subset captures natural variation in sperm concentration and morphology, suitable for segmentation, unsupervised clustering, or preprocessing.

Users may need to resize or normalize images for compatibility with specific machine learning models.

Citation and Referencing
UDataset DOI: Saadat H, Torkashvand H, Borna MR. Expert-annotated optical microscopy images of human sperm for detection and DNA fragmentation assessment. Figshare, 2025. https://doi.org/10.6084/m9.figshare.30120811
An associated manuscript describing this dataset is under review at Scientific Data. Once published, please cite both the dataset DOI and the article.

License
This dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Users may share and adapt the material provided proper attribution is given.