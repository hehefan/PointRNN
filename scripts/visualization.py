"""
3D Point Cloud Visualization
Original Author: https://github.com/argoai/argoverse-api
Modified by Hehe Fan
Date October 2019
"""

import os
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

from mayavi import mlab
from typing import Any, Iterable, List, Optional, Tuple, Union, cast

#: A stub representing mayavi_wrapper.mlab figure types
Figure = Any

#: A 3D Point
Point = np.ndarray

#: An array of 3D points
PointCloud = np.ndarray

#: Any numeric type
Number = Union[int, float]

#: RGB color created from 0.0 to 1.0 values
Color = Tuple[float, float, float]

FigSize = Tuple[float, float]
Coordinate = Tuple[float, float, float]

def plot_points_3D_mayavi(
    points: np.ndarray,
    bird: bool,
    fig: Figure,
    per_pt_color_strengths: np.ndarray = None,
    fixed_color: Optional[Color] = (1, 0, 0),
    colormap: str = "spectral",
) -> Figure:
    """Visualize points with Mayavi. Scale factor has no influence on point size rendering
    when calling `points3d()` with the mode="point" argument, so we ignore it altogether.
    The parameter "line_width" also has no effect on points, so we ignore it also.
    Args:
       points: The points to visualize
       fig: A Mayavi figure
       per_pt_color_strengths: An array of scalar values the same size as `points`
       fixed_color: Use a fixed color instead of a colormap
       colormap: different green to red jet for 'spectral' or 'gnuplot'
    Returns:
       Updated Mayavi figure
    """
    if len(points) == 0:
        return None

    if per_pt_color_strengths is None or len(per_pt_color_strengths) != len(points):
        # Height data used for shading
        if bird:
            per_pt_color_strengths = points[:, 2]
        else:
            per_pt_color_strengths = points[:, 0]

    mlab.points3d(
        points[:, 0],  # x
        points[:, 1],  # y
        points[:, 2],  # z
        per_pt_color_strengths,
        mode="point",  # Render each point as a 'point', not as a 'sphere' or 'cube'
        colormap=colormap,
        color=fixed_color,  # Used a fixed (r,g,b) color instead of colormap
        figure=fig,
    )

    return fig

def draw_coordinate_frame_at_origin(fig: Figure) -> Figure:
    """
    Draw the origin and 3 vectors representing standard basis vectors to express
    a coordinate reference frame.
    Args:
       fig: Mayavi figure
    Returns:
       Updated Mayavi figure
    Based on
    --------
    https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/kitti_data/draw.py
    https://github.com/charlesq34/frustum-pointnets/blob/master/mayavi/viz_util.py
    """
    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)

    # Form standard basis vectors e_1, e_2, e_3
    axes = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64)
    # e_1 in red
    mlab.plot3d(
        [0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig
    )
    # e_2 in green
    mlab.plot3d(
        [0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig
    )
    # e_3 in blue
    mlab.plot3d(
        [0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig
    )

    return fig

def draw_lidar(
    point_cloud: np.ndarray, bird: bool = True, colormap: str = "jet", fig: Optional[Figure] = None, bgcolor: Color = (0, 0, 0), fig_size: FigSize = (200, 200), focalpoint: Coordinate = (0, 0, 0), elevation: int = 0, distance: float = 62.0
) -> Figure:
    """Render a :ref:`PointCloud` with a 45 degree viewing frustum from worm-vehicle.
    Creates a Mayavi figure, draws a point cloud. Since the majority of interesting objects and
    scenarios are found closeby to the ground, we want to see the objects near the ground expressed
    in the full range of the colormap. Since returns on power lines, trees, and buildings
    will dominate and dilute the colormap otherwise, we clip the colors so that all points
    beyond a certain z-elevation (height) share the same color at the edge of the colormap.
    We choose anything beyond the 90th percentile as a height outlier.
    Args:
       point_cloud: The pointcloud to render
       fig: A pre-existing Mayavi figure to render to
       bgcolor: The background color
       colormap: "spectral" or "gnuplot" or "jet" are best
    Returns:
       Updated or created Mayavi figure
    """
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=fig_size)

    '''
    z_thresh = np.percentile(point_cloud[:, 2], 90)
    thresholded_heights = point_cloud[:, 2].copy()
    # Colors of highest points will be clipped to all lie at edge of colormap
    thresholded_heights[thresholded_heights > z_thresh] = 5
    '''
    tmp = [point_cloud]
    for i in range(1,201):
        tmp.append(point_cloud+0.0001*i)
        tmp.append(point_cloud-0.0001*i)
    point_cloud = np.concatenate(tmp, 0)

    # draw points
    fig = plot_points_3D_mayavi(
        #points=point_cloud, fig=fig, per_pt_color_strengths=thresholded_heights, fixed_color=None, colormap=colormap
        points=point_cloud, bird=bird, fig=fig, per_pt_color_strengths=None, fixed_color=None, colormap=colormap
    )
    fig = draw_coordinate_frame_at_origin(fig)
    mlab.view(
        azimuth=180, elevation=elevation, focalpoint=focalpoint, distance=distance, figure=fig
    )
    return fig

def mkdirs(name):
    if not os.path.exists(name):
        os.makedirs(name)

if __name__ == '__main__':

    R = 5

    gths = np.load('test-argo-5m-1024point-10step.npy')
    frames = np.load('test-predicted-frames.npy')

    bird_dir = 'bird'
    worm_dir = 'worm'

    mkdirs(bird_dir)
    mkdirs(worm_dir)

    distance = 2*np.sqrt(3*R*R)

    point_size = 5
    axes_limits = [[-R, R], [-R, R], [-R, R]]  # X axis range  # Y axis range  # Z axis range
    axes_str = ["X", "Y", "Z"]
    axes = [1, 0, 2]

    for i in range(gths.shape[0]):
        gth = gths[i]
        flow = flows[i]
        frame = frames[i]

        # bird’s-eye view
        curr_bird = os.path.join(bird_dir, '%04d'%(i+1))
        mkdirs(curr_bird)
        for j in range(5):
            fig = draw_lidar(gth[j], bird=True, focalpoint=(0, 0, 0), elevation=0, distance=distance)
            mlab.savefig(os.path.join(curr_bird, 'ctx-%02d.png'%(j+1)))
            mlab.close()
            fig = draw_lidar(gth[j+5], bird=True, focalpoint=(0, 0, 0), elevation=0, distance=distance)
            mlab.savefig(os.path.join(curr_bird, 'gth-%02d.png'%(j+1)))
            mlab.close()
            fig = draw_lidar(frame[j], bird=True, focalpoint=(0, 0, 0), elevation=0, distance=distance)
            mlab.savefig(os.path.join(curr_bird, 'prd-%02d.png'%(j+1)))
            mlab.close()
        os.system('convert -delay 20 -loop 0 %s/ctx-*.png %s/ctx.gif'%(curr_bird, curr_bird))
        os.system('convert -delay 20 -loop 0 %s/gth-*.png %s/gth.gif'%(curr_bird, curr_bird))
        os.system('convert -delay 20 -loop 0 %s/prd-*.png %s/prd.gif'%(curr_bird, curr_bird))

        # worm’s-eye view
        curr_worm = os.path.join(worm_dir, '%04d'%(i+1))
        mkdirs(curr_worm)
        for j in range(5):
            fig = draw_lidar(gth[j], bird=False, focalpoint=(0, 0, 0), elevation=90, distance=distance)
            mlab.savefig(os.path.join(curr_worm, 'ctx-%02d.png'%(j+1)))
            mlab.close()
            fig = draw_lidar(gth[j+5], bird=False, focalpoint=(0, 0, 0), elevation=90, distance=distance)
            mlab.savefig(os.path.join(curr_worm, 'gth-%02d.png'%(j+1)))
            mlab.close()
            fig = draw_lidar(frame[j], bird=False, focalpoint=(0, 0, 0), elevation=90, distance=distance)
            mlab.savefig(os.path.join(curr_worm, 'prd-%02d.png'%(j+1)))
            mlab.close()

        os.system('convert -delay 20 -loop 0 %s/ctx-*.png %s/ctx.gif'%(curr_worm, curr_worm))
        os.system('convert -delay 20 -loop 0 %s/gth-*.png %s/gth.gif'%(curr_worm, curr_worm))
        os.system('convert -delay 20 -loop 0 %s/prd-*.png %s/prd.gif'%(curr_worm, curr_worm))
