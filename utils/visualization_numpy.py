import matplotlib.pyplot as plt
import numpy as np


def show_as_grid_contour(ax, phi, linewidth=1.0, stride=8, flip=False):
    H, W = phi.shape[1:]
    plot_phi = phi - 0.5
    N = phi.shape[-1]

    # Use linspace exactly as in original code
    levels = np.linspace(0, N, int(N / stride))

    ax.contour(plot_phi[1], levels=levels, linewidths=linewidth, alpha=0.8)
    ax.contour(plot_phi[0], levels=levels, linewidths=linewidth, alpha=0.8)

    if flip:
        ax.set_ylim([0, H])


def visualize_one_case(check_data, pred_image, phi):
    def prep_slice(vol):
        return np.transpose(vol[0], (1, 0, 2))  # (1, D, H, W) -> (H, D, W)

    fixed_image = prep_slice(check_data["fixed_image"])
    moving_image = prep_slice(check_data["moving_image"])
    pred_image = pred_image[0, 0].transpose(1, 0, 2)  # → [D, H, W]

    print(fixed_image.shape, moving_image.shape, pred_image.shape, phi.shape)

    phi = phi[0]  # [3, D, H, W]
    D, H, W = phi.shape[1:]

    grid_d = np.arange(D).reshape(D, 1, 1).repeat(H, axis=1).repeat(W, axis=2)
    grid_h = np.arange(H).reshape(1, H, 1).repeat(D, axis=0).repeat(W, axis=2)
    grid_w = np.arange(W).reshape(1, 1, W).repeat(D, axis=0).repeat(H, axis=1)

    identity = np.stack([grid_d, grid_h, grid_w], axis=0)  # [3, D, H, W]
    phi = phi + identity  # deformation field + identity

    for slice_idx in range(30, 120, 10):
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        axs[0].imshow(1 - fixed_image[slice_idx, :, :], cmap="Greys", origin="lower")
        axs[0].set_title("Fixed Image")

        axs[1].imshow(1 - moving_image[slice_idx, :, :], cmap="Greys", origin="lower")
        axs[1].set_title("Moving Image")

        axs[2].imshow(1 - pred_image[slice_idx, :, :], cmap="Greys", origin="lower")
        axs[2].set_title("Predicted Image")

        axs[3].imshow(1 - pred_image[slice_idx, :, :], cmap="Greys", origin="lower")
        axs[3].set_title("Predicted + Grid")

        # 用 H/W 来可视化 deformation field (h, w)
        show_as_grid_contour(axs[3], phi[[1, 2], slice_idx], linewidth=1, stride=5, flip=False)

        for ax in axs.ravel():
            ax.set_axis_off()

        plt.suptitle(f"Slice {slice_idx} - Pre/Post Registration")
        plt.tight_layout()
        plt.show()
