# Import modules
import plotly.graph_objects as go
import numpy as np


# Define plane equations
def plane1(x: float, y: float) -> float:
    return 2*x + 3*y + 1


def plane2(x: float, y: float) -> float:
    return -x + 17*y - 3


def plane3(x: float, y: float) -> float:
    return 40*x - 16


# Set x and y ranges
x_values = [0, 1]
y_values = [0, 1]

# Calculate z values
z_values1 = np.zeros((len(x_values), len(y_values)))
z_values2 = np.copy(z_values1)
z_values3 = np.copy(z_values1)
for x, i_x in enumerate(x_values):
    for y, i_y in enumerate(y_values):
        z_values1[i_x, i_y] = plane1(x, y)
        z_values2[i_x, i_y] = plane2(x, y)
        z_values3[i_x, i_y] = plane3(x, y)

# Create plotly surface objects
colorscale = "Viridis"

surface1 = go.Surface(
    x=x_values,
    y=y_values,
    z=z_values1,
    # ^An array with x values as rows and y values as columns
    surfacecolor=np.ones(shape=z_values1.shape),
    # ^Has to be the same shape as the z values array
    # - determines which value from color scale corresponds to each z value point
    colorscale=colorscale,
    cmin=0,
    cmax=10,
    showscale=False)

surface2 = go.Surface(
    x=x_values,
    y=y_values,
    z=z_values2,
    surfacecolor=3*np.ones(shape=z_values2.shape),
    colorscale=colorscale,
    cmin=0,
    cmax=10,
    showscale=False)

surface3 = go.Surface(
    x=x_values,
    y=y_values,
    z=z_values3,
    surfacecolor=5*np.ones(shape=z_values3.shape),
    colorscale=colorscale,
    cmin=0,
    cmax=10,
    showscale=False)

# Create plotly figure
figure = go.Figure(
    data=[surface1, surface2, surface3])

figure.update_layout(
    scene=dict(
        # Set ranges
        xaxis=dict(range=x_values),
        yaxis=dict(range=y_values),
        zaxis=dict(range=[-10, 10])))

# Display plot
figure.show()
