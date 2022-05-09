from data_load_golf_swings import data_loader_golf_swings
import matplotlib.pyplot as plt
import os
CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR_PATH = os.path.dirname(CURRENT_DIR_PATH)

if __name__ == "__main__":
    for img in data_loader_golf_swings:
        img = img.squeeze()
        plt.imshow(img.permute(1,2,0))
        plt.show()