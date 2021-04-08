import pytorch_ssim

ssim = pytorch_ssim.SSIM()


def ssim_loss(img1, img2):
    img1 = img1.permute((0, 2, 1, 3, 4))
    img2 = img2.permute((0, 2, 1, 3, 4))

    size = img1.size()
    img1 = img1.reshape((-1, size[2], size[3], size[4]))
    img2 = img2.reshape((-1, size[2], size[3], size[4]))
    return ssim(img1, img2)
