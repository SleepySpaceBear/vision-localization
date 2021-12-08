# Vision Localization
This is an implementation of the paper [Global Localization and Relative Positioning Based on Scale-Invariant Keypoints](https://cs.gmu.edu/~kosecka/Publications/ras05.pdf) that uses the [Active Vision Dataset](https://www.cs.unc.edu/~ammirato/active_vision_dataset_website/index.html). This is my final project for my autonomous robotics class. There are two scripts here, one for creating the database (`setup-database.py`) and one for querying it (`vision-localization.py`).

## `setup-database.py`
Usage: `python setup-database.py [database dir] [Active Vision instance]`

This script takes two parameters: the path to the directory of the database to be created and the path to the active vision dataset instance to be used. The script will then feature match all the images in the dataset against each other and produce an optimal subset for feature-matching queries. 

## `vision-localization.py`

