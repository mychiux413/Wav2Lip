from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

ms_ssim_module = MS_SSIM(data_range=1.0, size_average=True, channel=3)


def ms_ssim_loss(batch_img1, batch_img2):
    batch_img1 = batch_img1.permute((0, 2, 1, 3, 4))
    batch_img2 = batch_img2.permute((0, 2, 1, 3, 4))

    size = batch_img1.size()
    batch_img1 = batch_img1.reshape((-1, size[2], size[3], size[4]))
    batch_img2 = batch_img2.reshape((-1, size[2], size[3], size[4]))
    return 1.0 - ms_ssim_module(batch_img1, batch_img2)
