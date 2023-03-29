import torch
import torch.nn.functional as F
import numpy as np
import absl.flags as flags

FLAGS = flags.FLAGS

def PC_sample(obj_mask, Depth, camK, coor2d):
    '''
    :param Depth: bs x 1 x h x w
    :param camK:
    :param coor2d:
    :return:
    '''
    # handle obj_mask
    if obj_mask.shape[1] == 2:   # predicted mask
        obj_mask = F.softmax(obj_mask, dim=1)
        _, obj_mask = torch.max(obj_mask, dim=1)
    '''
    import matplotlib.pyplot as plt
    plt.imshow(obj_mask[0, ...].detach().cpu().numpy())
    plt.show()
    '''
    bs, H, W = Depth.shape[0], Depth.shape[2], Depth.shape[3]
    x_label = coor2d[:, 0, :, :]
    y_label = coor2d[:, 1, :, :]

    rand_num = FLAGS.random_points
    samplenum = rand_num

    PC = torch.zeros([bs, samplenum, 3], dtype=torch.float32, device=Depth.device)

    for i in range(bs):
        # separate instance images and fuse the invalid zone whose depth values is small than 0,
        # caused by the depth camera
        dp_now = Depth[i, ...].squeeze()   # 256 x 256
        x_now = x_label[i, ...]   # 256 x 256
        y_now = y_label[i, ...]
        obj_mask_now = obj_mask[i, ...].squeeze()  # 256 x 256
        dp_mask = (dp_now > 0.0)
        fuse_mask = obj_mask_now.float() * dp_mask.float()

        camK_now = camK[i, ...]

        # analyze camK
        # fx and fy are the focal lengths of the camera in the x and y directions,
        # respectively. ux and uy are the x and y coordinates of the principal point
        # (the point where the optical axis intersects the image plane) in the image plane.
        fx = camK_now[0, 0]
        fy = camK_now[1, 1]
        ux = camK_now[0, 2]
        uy = camK_now[1, 2]
        #
        x_now = (x_now - ux) * dp_now / fx
        y_now = (y_now - uy) * dp_now / fy
        # In this situation,
        # the x-coordinate, y-coordinate, and z-coordinate refer to the 3D spatial coordinates of the points in the camera frame.
        # The x-coordinate and y-coordinate correspond to the pixel coordinates of the points in the 2D image plane,
        # while the z-coordinate represents the distance of each point from the camera,
        # expressed in metric units such as meters. #
        # Once these 3D coordinates have been obtained,
        # they can be used for further processing or analysis,
        # such as object tracking, 3D reconstruction, or augmented reality applications.
        p_n_now = torch.cat([x_now[fuse_mask > 0].view(-1, 1),
                             y_now[fuse_mask > 0].view(-1, 1),
                             dp_now[fuse_mask > 0].view(-1, 1)], dim=1)

        # basic sampling
        if FLAGS.sample_method == 'basic':
            l_all = p_n_now.shape[0]
            if l_all <= 1.0:
                return None, None
            if l_all >= samplenum:
                replace_rnd = False
            else:
                replace_rnd = True

            choose = np.random.choice(l_all, samplenum, replace=replace_rnd)  # can selected more than one times
            p_select = p_n_now[choose, :]
        else:
            p_select = None
            raise NotImplementedError

        # reprojection
        if p_select.shape[0] > samplenum:
            p_select = p_select[p_select.shape[0]-samplenum:p_select.shape[0], :]

        PC[i, ...] = p_select[:, :3]

    return PC / 1000.0