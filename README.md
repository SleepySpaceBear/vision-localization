# Vision Localization
This is an implementation of the paper [Global Localization and Relative Positioning Based on Scale-Invariant Keypoints](https://cs.gmu.edu/~kosecka/Publications/ras05.pdf) that uses the [Active Vision Dataset](https://www.cs.unc.edu/~ammirato/active_vision_dataset_website/index.html). There are two scripts here, one for creating the database (`setup-database.py`) and one for querying it (`vision-localization.py`).

## `setup-database.py`
Usage: `python setup-database.py [database directory] [Active Vision dataset instance] [path to text file list of database images]`

This script takes three parameters: the path to the directory of the database to be created, the path to the active vision dataset instance, and the path to the list of images to be used for the database. The list of images used for testing is `database_images.txt`. It moves the database images into the database and creates a `image_params.json` file, which has the world position(`pos` attribute) and viewing direction (`dir` attribute) for each image in the database.

## `vision-localization.py`

