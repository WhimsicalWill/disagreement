import os
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import distrax
import cv2

treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)


class OneHotDist(distrax.OneHotCategorical):

    def __init__(self, logits=None, probs=None):
        super().__init__(logits, probs)

    def sample(self, sample_shape=(), seed=None):
        sample = super().sample(sample_shape=sample_shape, seed=seed)
        # Straight-through estimator for sampling
        sample = sg(sample) + (self.probs - sg(self.probs))
        return sample


def make_video_comparison(videos_start, videos_end, pred_start, pred_end, config):
    '''
    Creates a video grid to compare the data and its reconstructions. The grid
    is of size (2, B) and each cell contains a video. The top row is the actual
    data and the bottom row is the reconstructions.

	Args:
		videos_start (jnp.array): The first 10 frames of the video data (B, 10, 64, 64, 1)
		videos_end (jnp.array): The last 10 frames of the video data (B, 10, 64, 64, 1)
		pred_start (jnp.array): The first 10 frames reconstructed by the posterior (B, 10, 64, 64, 1)
		pred_end (jnp.array): The last 10 frames predicted by the prior (B, 10, 64, 64, 1)
        config: The config
	Returns:
		jnp.ndarray: An ndarray containing the video data
    '''
    # Slice the input data to only consider the first K samples of the minibatch
    videos_start, videos_end, pred_start, pred_end = (
        videos_start[:config['COMPARISON_SAMPLES']],
        videos_end[:config['COMPARISON_SAMPLES']],
        pred_start[:config['COMPARISON_SAMPLES']],
        pred_end[:config['COMPARISON_SAMPLES']]
    )

    # Concatenate start and end to form the full ground truth and predictions
    true_video = jnp.concatenate([videos_start, videos_end], axis=1)  # shape (K, 20, 64, 64, 1)
    pred_video = jnp.concatenate([pred_start, pred_end], axis=1)  # shape (K, 20, 64, 64, 1)
    
    # Create a grid of 2 rows and K columns, each cell is a video of shape (20, 64, 64, 1)
    num_samples = true_video.shape[0]
    grid = jnp.stack([true_video, pred_video])  # (2, K, 20, 64, 64, 1)

    # Convert the grid to (T, H, W) where T is the number of time steps, H the height, and W the width
    # We need to stack the videos into a grid format
    T = 20  # Number of frames
    H = 64 * 2  # 2 rows: ground truth on top, prediction on bottom
    W = 64 * num_samples  # K columns for K samples
    
    video_frames = jnp.zeros((T, H, W), dtype=jnp.uint8)
    
    for t in range(T):
        for row in range(2):  # ground truth and prediction
            for col in range(num_samples):  # for each sample
                # Extract a frame from the input to be used as a cell in the video grid
                cell = grid[row, col, t].squeeze()  # (64, 64)
                # Place it in the correct location in the grid
                video_frames = (video_frames
                    .at[t, row * 64:(row + 1) * 64, col * 64:(col + 1) * 64]
                    .set((cell * 255).astype(jnp.uint8))
                )

    return video_frames


def save_video(video, filename):
    '''
    Writes data to an mp4 video file.

	Args:
		video (jnp.array): The video data of shape (T, H, W)
		filename (str): The filename to write the video to
    '''
    if not os.path.exists('viz'):
        os.makedirs('viz')
    filepath = os.path.join('viz', filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    FPS = 2.0
    height, width = video.shape[1:]
    video_writer = cv2.VideoWriter(filepath, fourcc, FPS, (width, height), isColor=False)
    video_np = np.array(video)

    for frame in video_np:
        video_writer.write(frame)
    
    video_writer.release()