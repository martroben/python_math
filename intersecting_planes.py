# Import modules
import plotly.graph_objects as go
import numpy as np


# Define plane equations
def plane1(x: float, y: float) -> float:
    return 1*x + 2*y + 3


def plane2(x: float, y: float) -> float:
    return 1*x - 6*y


def plane3(x: float, y: float) -> float:
    return 1*x + 12*y - 15


# Set x and y values and z range
x_values = [-3, 3]
y_values = [-3, 3]
z_range = [-10, 15]

# Calculate z values
z_values1 = np.zeros((len(x_values), len(y_values)))
z_values2 = np.copy(z_values1)
z_values3 = np.copy(z_values1)
for i_x, x in enumerate(x_values):
    for i_y, y in enumerate(y_values):
        z_values1[i_y, i_x] = plane1(x, y)
        z_values2[i_y, i_x] = plane2(x, y)
        z_values3[i_y, i_x] = plane3(x, y)

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
        zaxis=dict(range=z_range)))

# Display plot
figure.show()
