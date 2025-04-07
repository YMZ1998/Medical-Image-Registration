import matplotlib.pyplot as plt


def overlay_img(img1, img2, slice_idx, ax, title=None):
    ax.imshow(1 - img1[:, slice_idx, :].T, cmap="Blues", origin="lower")
    ax.imshow(1 - img2[:, slice_idx, :].T, cmap="Oranges", origin="lower", alpha=0.5)
    if title is not None:
        ax.title.set_text(title)


def visualize_registration(check_data, pred_image, pred_label, ddf_keypoints, target_res):
    # Permute to get correct axis for visualization
    def prep_slice(vol):
        return vol[0][0].permute(1, 0, 2).cpu()

    fixed_image = prep_slice(check_data["fixed_image"])
    fixed_label = prep_slice(check_data["fixed_label"])
    moving_image = prep_slice(check_data["moving_image"])
    moving_label = prep_slice(check_data["moving_label"])
    pred_image = pred_image[0][0].permute(1, 0, 2).cpu()
    pred_label = pred_label[0][0].permute(1, 0, 2).cpu()

    slice_idx = int(target_res[0] * 95.0 / 224)  # Slice equivalent to 95 in 224-depth

    fig, axs = plt.subplots(2, 2)
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

    fig = plt.figure()
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
