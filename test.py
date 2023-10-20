"""
this module is for testing and debugging purposes.
that means it is for personal usage only and is not part of the project.
It contains functions that are not imported in any other part of the project.
"""

import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduced_plot(
        df: pd.DataFrame | str = None,
        color: str = 'Label',
        n_components=3, method: str = 'PCA'):
    """
    Plot a 2D or 3D scatter plot of reduced version of the given dataframe.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Dataframe to plot. If not provided, reads from 'data/train.csv'.
    color : str, optional
        Column name of the column to color the points by.
    method : str, optional
        The method to use for dimensionality reduction. Must be
        either 'PCA' or 'TSNE'.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If color is not a column in df.

    Returns
    -------
    None
    """
    if isinstance(df, str):
        df = pd.read_csv(df, index_col=0)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame or str")
    if color not in df.columns:
        raise ValueError(f"color must be a column in df, but {color} is not")
    if n_components not in [2, 3]:
        raise ValueError("n_components must be either 2 or 3")

    if method == 'PCA':
        reducer = PCA(n_components=n_components)
    elif method == 'TSNE':
        reducer = TSNE(n_components=n_components)
    else:
        raise ValueError("method must be either PCA or TSNE")

    if n_components == 2:

        reduced = pd.DataFrame(
            reducer.fit_transform(df[list(set(df.columns) - {color})]),
            columns=['Component 1', 'Component 2'])
        reduced[color] = df[color].astype('str')

        fig = px.scatter(
            reduced, x='Component 1', y='Component 2',
            color=color, title=f'{method} plot')

    else:

        reduced = pd.DataFrame(
            reducer.fit_transform(df[list(set(df.columns) - {color})]),
            columns=['Component 1', 'Component 2', 'Component 3'])
        reduced[color] = df[color].astype('str')

        fig = px.scatter_3d(
            reduced, x='Component 1', y='Component 2', z='Component 3',
            color=color, title=f'{method} plot')

    fig.update_traces(
        marker=dict(size=4),
        selector=dict(mode='markers'))
    fig.show()
