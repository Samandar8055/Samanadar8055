import numpy as np
from matplotlib import pyplot as plt, image as mpimg, patches

from swarm import Swarm
from utils import TEXT, ANIMATION, get_out_mode, gen_data, animate, save


def dissimilarity(x, y, model, target):
    i = int(y)
    j = int(x)
    return np.linalg.norm(target[i: i + model.shape[0], j: j + model.shape[1]] - model)


def plot(model, target, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, num=name + ".png", figsize=(12, 7), gridspec_kw={'width_ratios': [1, 6]})

    ax1.imshow(model, cmap="gray", vmin=0, vmax=1)
    ax2.imshow(target, cmap="gray", vmin=0, vmax=1)

    ax1.set_title("Model image")
    ax2.set_title("Target image")

    ax1.axis("off")
    ax2.axis("off")

    return fig, ax2


def display(model, target, name):
    plot(model, target, name)
    plt.show()


def init(model, target, name):
    fig, ax2 = plot(model, target, name)

    title = ax2.text(0, 0, "", bbox={"facecolor": "w", "pad": 5}, fontsize=12)
    boxes = [patches.Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='r', facecolor='none')
             for _ in range(swarm.n_particles)]
    for box in boxes:
        ax2.add_patch(box)

    return fig, title, boxes


def update(data):
    x, y, _, count = data

    title.set_text("Epochs: {}".format(count))
    for k in range(swarm.n_particles):
        boxes[k].set_bounds(x[k], y[k], model.shape[1], model.shape[0])

    return *boxes, title


INFO = {  # parameters for the animation
    "quarto_stato": {
        "fps": 10
    },


}

if __name__ == "__main__":
    out_mode = get_out_mode()

    for name in INFO.keys():
        print("\n" + name + ".png")
        target = mpimg.imread("./" + name + "/target3.png")
        model = mpimg.imread("./" + name + "/model3.png")

        swarm = Swarm([(0, target.shape[1] - model.shape[1]), (0, target.shape[0] - model.shape[0])],
                      lambda x, y: dissimilarity(x, y, model, target), 0.5)

        if out_mode == ANIMATION:
            display(model, target, name)
            fig, title, boxes = init(model, target, name)
            anim = animate(fig, update, lambda: gen_data(swarm), INFO[name]["fps"])
#            save(anim, name + "/" + name, INFO[name]["fps"], ".mp4")
        elif out_mode == TEXT:
            swarm.minimize()

    print()
