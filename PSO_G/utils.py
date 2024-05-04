from matplotlib import pyplot as plt, animation as anim

TEXT = "t"
ANIMATION = "a"


def get_out_mode():
    mode = input("\nHow do you want the output to be displayed? Type \"t\" for text-only, \"a\" for animation: ")
    while mode != TEXT and mode != ANIMATION:
        mode = input("Please type either \"t\" or \"a\": ")
    return mode


def gen_data(swarm):
    while swarm.has_not_converged():
        swarm.update_swarm()

        x = swarm.positions[:, 0]
        y = swarm.positions[:, 1]
        z = swarm.get_f_values()
        count = swarm.epoch_count

        yield x, y, z, count


def animate(fig, update, gen_data, fps):
    animation = anim.FuncAnimation(fig, update, gen_data, interval=1e3/fps, blit=True, repeat=False, save_count=1500)
    plt.show()
    return animation


def save(animation, name, fps, ext=".gif"):
    writer = anim.PillowWriter(fps) if ext == ".gif" else anim.FFMpegWriter(fps)
    animation.save("images/" + name + ext, writer)
