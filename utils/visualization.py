import matplotlib.pyplot as plt
import numpy as np
import torch


def show_img(img, slice_idx, ax, title=None):
    ax.imshow(1 - img[slice_idx, :, :].T.cpu().numpy(), cmap="Greys", origin="lower")
    if title is not None:
        ax.set_title(title)


def overlay_img(img1, img2, slice_idx, ax, title=None):
    ax.imshow(1 - img1[:, slice_idx, :].T.cpu().numpy(), cmap="Greys", origin="lower")
    ax.imshow(1 - img2[:, slice_idx, :].T.cpu().numpy(), cmap="Greys", origin="lower", alpha=0.5)
    if title is not None:
        ax.set_title(title)


def visualize_registration(check_data, pred_image, pred_label, ddf_keypoints, target_res):
    def prep_slice(vol):
        return vol[0][0].permute(1, 0, 2).cpu()

    fixed_image = prep_slice(check_data["fixed_image"])
    fixed_label = prep_slice(check_data["fixed_label"])
    moving_image = prep_slice(check_data["moving_image"])
    moving_label = prep_slice(check_data["moving_label"])
    pred_image = pred_image[0][0].permute(1, 0, 2).cpu()
    pred_label = pred_label[0][0].permute(1, 0, 2).cpu()

    # slice_idx = int(target_res[0] * 95.0 / 224)  # Slice equivalent to 95 in 224-depth
    for slice_idx in range(10, target_res[0] - 10, 20):
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        overlay_img(fixed_image, moving_image, slice_idx, axs[0, 0], "Before registration")
        overlay_img(fixed_image, pred_image, slice_idx, axs[0, 1], "After registration")
        overlay_img(fixed_label, moving_label, slice_idx, axs[1, 0])
        overlay_img(fixed_label, pred_label, slice_idx, axs[1, 1])
        for ax in axs.ravel():
            ax.set_axis_off()
        plt.suptitle("Image and label visualizations pre-/post-registration")
        plt.tight_layout()
        plt.show()

        # Keypoint visualization
        fixed_kp = check_data["fixed_keypoints"][0].cpu()
        moving_kp = check_data["moving_keypoints"][0].cpu()
        moved_kp = fixed_kp + ddf_keypoints[0].cpu()

        fig = plt.figure(figsize=(8, 4))
        for i, title, fkp in zip(
            [1, 2], ["Before registration", "After registration"], [fixed_kp, moved_kp]
        ):
            ax = fig.add_subplot(1, 2, i, projection="3d")
            ax.scatter(fkp[:, 0], fkp[:, 1], fkp[:, 2], s=2.0, color="lightblue")
            ax.scatter(moving_kp[:, 0], moving_kp[:, 1], moving_kp[:, 2], s=2.0, color="orange")
            ax.set_title(title)
            ax.view_init(-10, 80)
            ax.set_aspect("auto")

        plt.show()


def visualize_one_case(check_data, pred_image, phi):
    def prep_slice(vol):
        return vol[0][0].permute(1, 0, 2).cpu()

    fixed_image = prep_slice(check_data["fixed_image"])
    moving_image = prep_slice(check_data["moving_image"])
    pred_image = pred_image[0][0].permute(1, 0, 2).cpu()
    phi = phi.permute(0, 1, 3, 2, 4).cpu()

    print(fixed_image.shape, moving_image.shape, pred_image.shape, phi.shape)
    # print(phi.max(), phi.min())

    D, H, W = phi.shape[2:]
    grid_d = torch.arange(D).view(D, 1, 1).expand(D, H, W)
    grid_h = torch.arange(H).view(1, H, 1).expand(D, H, W)
    grid_w = torch.arange(W).view(1, 1, W).expand(D, H, W)
    identity = torch.stack((grid_d, grid_h, grid_w), dim=0).unsqueeze(0).float()  # [1, 3, D, H, W]

    phi = phi + identity

    for slice_idx in range(30, 120, 10):
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        show_img(fixed_image, slice_idx, axs[0], "Fixed Image")
        show_img(moving_image, slice_idx, axs[1], "Moving Image")
        show_img(pred_image, slice_idx, axs[2], "Predicted Image")
        show_img(pred_image, slice_idx, axs[3], "Predicted Image")
        show_as_grid_contour(axs[3], phi[0, [1, 2], slice_idx], linewidth=1, stride=5, flip=False)

        for ax in axs.ravel():
            ax.set_axis_off()

        plt.suptitle(f"Slice {slice_idx} - Pre/Post Registration")
        plt.tight_layout()
        plt.show()


def show_as_grid_contour(ax, phi, linewidth=1., stride=8, flip=False):
    data_size = phi.size()[1:]
    plot_phi = phi.cpu() - 0.5
    N = plot_phi.size()[-1]
    ax.contour(plot_phi[1], np.linspace(0, N, int(N / stride)), linewidths=linewidth, alpha=0.8)
    ax.contour(plot_phi[0], np.linspace(0, N, int(N / stride)), linewidths=linewidth, alpha=0.8)
    if flip:
        ax.set_ylim([0, data_size[0]])
