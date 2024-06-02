import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator


def read_data(filename, depth=0.0):
    with open(filename, mode="r", encoding="utf-8") as file:
        data = json.load(file)
        size = data["size"]
        num_of_boreholes = data["num_of_boreholes"]
        boreholes = data["boreholes"]
        points, values = [], []

        for i in range(num_of_boreholes):
            coordinates = boreholes[i]["coordinates"]
            num_of_sensors = boreholes[i]["num_of_sensors"]
            temperatures = boreholes[i]["temperatures"]

            for j in range(num_of_sensors):
                if temperatures[j][1] == depth:
                    points.append(coordinates)
                    values.append(temperatures[j][0])

    return size, np.array(points), np.array(values)


def idw_interpolate(points, values, grid_x, grid_y, power=2):
    if power < 0:
        raise ValueError("Invalid power argument")

    x_coord, y_coord = points[:, 0], points[:, 1]
    x_axis, y_axis = grid_x.ravel(), grid_y.ravel()

    result = np.zeros_like(x_axis)

    for i, (x0, y0) in enumerate(zip(x_axis, y_axis)):
        dists = np.sqrt((x_coord - x0)**2 + (y_coord - y0)**2)

        if np.any(dists == 0):
            result[i] = values[np.argmin(dists)]
        else:
            weights = 1 / (dists ** power)
            result[i] = np.sum(weights * values) / np.sum(weights)

    return result.reshape(grid_x.shape)


def tin_interpolate(points, values, grid_x, grid_y):
    triangulation = Delaunay(points)
    interpolator = LinearNDInterpolator(triangulation, values)

    grid_z_linear = interpolator(grid_x, grid_y)
    grid_z_nearest = idw_interpolate(points, values, grid_x, grid_y)

    return np.where(np.isnan(grid_z_linear), grid_z_nearest, grid_z_linear)


def rbf_interpolate(points, values, grid_x, grid_y):
    x_axis, y_axis = grid_x.ravel(), grid_y.ravel()
    plane_points = np.vstack((x_axis, y_axis)).T

    interpolator = RBFInterpolator(
        points, values, kernel="multiquadric", epsilon=.375)

    return interpolator(plane_points).reshape(grid_x.shape)


def show_plot(size, points, values, grid):
    size_x, size_y = size
    x_coord, y_coord = points[:, 0], points[:, 1]

    plt.imshow(grid, extent=[0, size_x, 0, size_y],
               origin='lower', cmap="rainbow")
    plt.scatter(x_coord, y_coord, c=values,
                cmap="rainbow", edgecolors="black")
    plt.colorbar(label='Temperature')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Temperature distribution')
    plt.show()


def main():
    size, points, values = read_data("data.json", depth=0)
    size_x, size_y = size

    grid_x, grid_y = np.meshgrid(np.linspace(0, size_x, 100),
                                 np.linspace(0, size_y, 100))

    # grid_z = idw_interpolate(points, values, grid_x, grid_y)
    # grid_z = tin_interpolate(points, values, grid_x, grid_y)
    grid_z = rbf_interpolate(points, values, grid_x, grid_y)

    show_plot(size, points, values, grid_z)


if __name__ == "__main__":
    main()
