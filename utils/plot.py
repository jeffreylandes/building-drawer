import matplotlib.pyplot as plt


def plot_sample(sample):
    site = sample["site"]
    plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
    plt.subplot(131)
    plt.title("Buildings")
    plt.imshow(site[0])
    plt.axis("off")
    plt.subplot(132)
    plt.title("Initial Building Drawing")
    plt.imshow(site[1])
    plt.axis("off")
    plt.subplot(133)
    plt.title("Starting Point")
    plt.imshow(site[2])
    plt.axis("off")
    plt.show()
