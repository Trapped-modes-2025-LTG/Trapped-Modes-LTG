import matplotlib.pyplot as plt
import numpy as np


class HeightMap:
    def __init__(self, height_map, angles, calibration_factor):
        self.values = height_map
        self.angles = angles
        self.calibration_factor = calibration_factor
        self.coordinates = self.generate_coordinates()

    def generate_coordinates(self):
        xs = np.linspace(0, self.values.shape[1], self.values.shape[1]) * self.calibration_factor
        ys = np.linspace(0, self.values.shape[0], self.values.shape[0]) * self.calibration_factor
        return Coordinates(xs, ys)

    def show_angles(self):
        fig, axs = plt.subplots(1, 2)
        for i, angles in enumerate(self.angles):
            im = axs[i].contourf(self.coordinates.x_mesh*1e3, self.coordinates.y_mesh*1e3, self.angles[i], 100)
            for c in im.collections: # Esto soluciona el problema de aliasing al guardar como .pdf.
                c.set_edgecolor("face")
            cbar = fig.colorbar(im, ax=axs[i])
            cbar.set_label('Ángulo [rad]', rotation=270, labelpad=15)
            axs[i].set_xlabel('Posición x [mm]')
            axs[i].set_ylabel('Posición y [mm]')
            axs[i].set_aspect("equal")
            axs[i].set_title(f"Phi_{i+1}.")
        plt.tight_layout()
        plt.show()

    def show(self, fig=None, axis=None, show_plot=True):  # TODO: Tal vez centrar por defecto.
        if fig is None and axis is None:
            fig, axis = plt.subplots()
        im = axis.contourf(self.coordinates.x_mesh * 1e3, self.coordinates.y_mesh * 1e3, self.values * 1e3, 100)
        for c in im.collections: # Esto soluciona el problema de aliasing al guardar como .pdf.
            c.set_edgecolor("face")

        cbar = fig.colorbar(im, ax=axis)
        cbar.set_label('Altura [mm]', rotation=270, labelpad=15)
        axis.set_xlabel('Posición x [mm]')
        axis.set_ylabel('Posición y [mm]')
        axis.set_aspect("equal")
        if show_plot:
            plt.show()
        return fig

    # TODO: agregar un radial_mean(self, center, max_radius) y devuelve el promedio y std. O se llamaría ¿angular_mean()?.
    def show_slice(self, x_index=None, y_index=None, show_plot=True):  # TODO: Vertical u horizontal (¿o ángulo arbitrario, radial?), máximo o mínimo.
        if x_index is None and y_index is None:
            max_indexs = np.argwhere(np.max(self.values) == self.values)[0][::-1]
        elif x_index is not None and y_index is None:
            max_indexs = [x_index, 0]
        elif x_index is None and y_index is not None:
            max_indexs = [0, y_index]
        else:
            max_indexs = [x_index, y_index]
        self.coordinates.recenter(max_indexs)

        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(6, 8))
        self.show(fig, axs[0], show_plot=False)

        axs[0].hlines(0, self.coordinates.x[0] * 1e3, self.coordinates.x[-1] * 1e3, linestyle='--', linewidth=2, color='white')
        axs[0].scatter([0], [0], color="white")

        sliced = self.values[max_indexs[1], :]
        axs[1].plot(self.coordinates.x * 1e3, sliced * 1e3)
        axs[1].set_ylabel("Altura [mm]")
        axs[1].set_xlabel("Posición [mm]")
        plt.tight_layout()

        if show_plot:
            plt.show()
        return fig


class Coordinates:
    def __init__(self, xs, ys):
        self.x = xs
        self.y = ys
        self._x_mesh = None
        self._y_mesh = None

    def _calculate_meshgrid(self):
        self._x_mesh, self._y_mesh = np.meshgrid(self.x, self.y)

    @property
    def x_mesh(self):
        if self._x_mesh is None:
            self._calculate_meshgrid()
        return self._x_mesh

    @property
    def y_mesh(self):
        if self._y_mesh is None:
            self._calculate_meshgrid()
        return self._y_mesh

    def recenter(self, new_center_indexs):
        self.x -= self.x[new_center_indexs[0]]
        self.y -= self.y[new_center_indexs[1]]

        if self._x_mesh is not None or self._y_mesh is not None:
            self._calculate_meshgrid()
