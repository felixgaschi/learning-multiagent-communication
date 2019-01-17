import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def decode_pos(pos, size, offset=1):
    """
    transforms integer position on grid into i, j
    """
    return (pos - offset) // size, (pos - offset) % size

def encode_pos(i, j, size, offset=1):
    """
    encode i, j position into integer (with offset for additionnal information
    e.g. total absence)
    """
    pos = offset + j + i * size
    if pos > size ** 2:
        return 0
    else:
        return pos


def move_pos(coord, dir, size):
    if dir == 1:
        new_coord = max(0, coord[0] - 1), coord[1]
    elif dir == 2:
        new_coord = min(size - 1, 1 + coord[0]), coord[1]
    elif dir == 3:
        new_coord = coord[0], min(size - 1, coord[1] + 1)
    elif dir == 4:
        new_coord = coord[0], max(0, coord[1] - 1)
    else:
        new_coord = coord
    return new_coord, (new_coord == coord and dir != 0)

def plot_indiv(points, ax, title, grid_size, tt, prey = False):
    
    
    ticks = np.arange(grid_size)
    norm = plt.Normalize(tt.min(), tt.max())
    
    if not (points == points[0]).all():
        points = points.reshape(-1, 1 , 2)

        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if not prey:
            lc = LineCollection(segments, cmap = 'winter' ,norm=norm)
        else:
            lc = LineCollection(segments, cmap = 'autumn' ,norm=norm)
        lc.set_array(tt)
        lc.set_linewidth(10)
        line = ax.add_collection(lc)
        plt.colorbar(line, ax=ax)
    else:
        if not prey:
            ax.plot(points[0,0], points[0,1], 's', color = 'b', markersize = 10)
        else:
            ax.plot(points[0,0], points[0,1], 's', color = 'r', markersize = 10)
            
    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid()
    ax.set_title(title)
    
def plot_episode(pred_coord_history, prey_coord_history, grid_size, figsize = (20, 10), n_cols = 3):
    
    n_agents = pred_coord_history.shape[1]
    n_cols = n_cols
    n_rows = np.int(np.ceil((n_agents + 1)  / n_cols))

    tt = np.arange(pred_coord_history.shape[0])

    fig = plt.figure(figsize = figsize)

    # pred
    for i in range(pred_coord_history.shape[1]):
        plot_coord = np.unravel_index(i, (n_rows,n_cols))
        ax = plt.subplot2grid((n_rows, n_cols),plot_coord, fig = fig)
        points = pred_coord_history[:,i]

        plot_indiv(points, ax, "Predator %d" % i, grid_size, tt, prey = False)

    # prey
    plot_coord = np.unravel_index(n_agents, (n_rows,n_cols))
    ax = plt.subplot2grid((n_rows, n_cols), plot_coord, fig = fig)
    points = prey_coord_history

    plot_indiv(points, ax, "Prey", grid_size, tt, prey = True)
    plt.show()
