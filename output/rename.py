import os


PATH = "image/"
def rename():
    """
    the function renames the images in output/image folder
    """
    for count, file_name in enumerate(os.listdir(PATH)):
        dst = "output" + str(count) + ".jpg"
        src = PATH + file_name
        dst = PATH + dst

        os.rename(src, dst)


if __name__ == '__main__':
    rename()