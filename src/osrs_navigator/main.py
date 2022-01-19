from helpers import create_grid, plot_path, rescale_pos
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

import cv2

if __name__ == '__main__':
    # TODO make into function - then use as Pathfinder inside Navigator
    maze = create_grid(
        r'C:\Users\husar\PycharmProjects\OSRS\OSRS_automation\src\osrs_navigator\images\maps\converted_bw_map_cropped'
        r'.png')
    start_pos = (164, 316)
    goal_pos = (47, 110)  # [172, 119]# #[422, 287] # [612, 790]  #  #
    # start = rescale_pos(start, [211, 365], [200, 200])
    # goal = rescale_pos(goal, [211, 365], [200, 200])
    print("Positions: ", start_pos, goal_pos)

    image = cv2.circle(maze, start_pos, radius=5, color=(255, 0, 0), thickness=-1)
    image = cv2.circle(image, goal_pos, radius=5, color=(255, 0, 0), thickness=-1)
    cv2.imshow("water", image)
    cv2.waitKey(5000)
    # A-star pathfinder
    grid = Grid(matrix=maze)
    start = grid.node(164, 316)
    end = grid.node(47, 110)
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, runs = finder.find_path(start, end, grid)

    plot_path(path, list(start_pos), list(goal_pos), maze)
    print(path)
