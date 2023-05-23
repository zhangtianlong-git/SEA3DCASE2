import numpy as np
import math
from matplotlib import pyplot as plt

# from casalNumerical import hxy


def get_cross_fill_cut(
    x, y, z, theta_xy, terrain, roadhalf, off_road, slop_g_c, slope_g_f
):
    """
    more函数包含更多的参数输出，用于作图，但计算效率更低。总体思路为把横截面分成三部分，左、中
    、右，然后分别计算对应的高差，最后合并矩阵求挖方和填方量
    """
    deltax, deltay = math.cos(theta_xy - math.pi / 2), math.sin(theta_xy - math.pi / 2)
    try:
        d_r = terrain[round(x + roadhalf * deltax), round(y + roadhalf * deltay)] - z
        d_l = terrain[round(x - roadhalf * deltax), round(y - roadhalf * deltay)] - z
    except:
        return (0, 0, 0, True)
    sign_d_r, sign_d_l = np.sign(d_r), np.sign(d_l)
    ## 判断左右挖方还是填方确定坡度
    slop_g_r = slop_g_c if d_r > 0 else slope_g_f
    slop_g_l = slop_g_c if d_l > 0 else slope_g_f
    tmp = np.arange(1, int(off_road) + 1)
    tmp_arr = tmp + roadhalf
    xs_r, xs_l = x + tmp_arr * deltax, x - tmp_arr * deltax
    ys_r, ys_l = y + tmp_arr * deltay, y - tmp_arr * deltay
    ## zs为横截面左右坡上的纵坐标
    zs_r, zs_l = z + sign_d_r * tmp * slop_g_r, z + sign_d_l * tmp * slop_g_l
    try:
        zs_r_delta = terrain[np.round(xs_r).astype(int), np.round(ys_r).astype(int)] - zs_r
        zs_l_delta = terrain[np.round(xs_l).astype(int), np.round(ys_l).astype(int)] - zs_l
    except:
        return (0, 0, 0, True)
    ## 这里index表征地面线和横截面的交点，必须用大于0，防止出现索引错误
    r_index, l_index = zs_r_delta * sign_d_r > 0, zs_l_delta * sign_d_l > 0
    zs_r_delta_final, zs_l_delta_final = zs_r_delta[r_index], zs_l_delta[l_index]
    is_exceed = False
    if len(zs_r_delta_final) == int(off_road) or len(zs_l_delta_final) == int(off_road):
        is_exceed = True
    ## 上述算的是左右坡面，这里算道路宽度范围内对应的高差，全部有效
    delta_int_road = roadhalf - int(roadhalf)
    tmp1 = np.arange(int(-roadhalf), int(roadhalf) + 1)
    xs_c, ys_c = x + tmp1 * deltax, y + tmp1 * deltay
    zs_c_delta_final = (
        terrain[np.round(xs_c).astype(int), np.round(ys_c).astype(int)] - z
    )
    ## 有效高差的矩阵进行拼接，但左坡数据的顺序未调整，不影响最终计算
    z_delta_final = np.concatenate(
        (zs_l_delta_final, zs_c_delta_final, zs_r_delta_final)
    )
    fill, excave = sum(z_delta_final[z_delta_final < 0]), sum(
        z_delta_final[z_delta_final >= 0]
    )
    ## road的取整部分更新挖填方和用地界
    tmp_list = delta_int_road * np.array([zs_c_delta_final[0], zs_c_delta_final[-1]])
    fill += sum(tmp_list[tmp_list < 0])
    excave += sum(tmp_list[tmp_list > 0])
    width = len(zs_l_delta_final) + roadhalf * 2 + len(zs_r_delta_final)
    tmp_list1 = []
    tmp_list1 = (
        tmp_list1 + [zs_r_delta_final[-1]]
        if len(zs_r_delta_final) > 0
        else tmp_list1 + [zs_c_delta_final[-1]]
    )
    tmp_list1 = (
        tmp_list1 + [zs_l_delta_final[-1]]
        if len(zs_l_delta_final) > 0
        else tmp_list1 + [zs_c_delta_final[0]]
    )
    tmp_list2 = np.array([tmp_list1])
    width += abs(sum(tmp_list2[tmp_list2 < 0] / slope_g_f))
    width += abs(sum(tmp_list2[tmp_list2 > 0] / slop_g_c))
    if __name__ == "__main__":
        xs_r_final, ys_r_final, zs_r_final = xs_r[r_index], ys_r[r_index], zs_r[r_index]
        xs_l_final, ys_l_final, zs_l_final = xs_l[l_index], ys_l[l_index], zs_l[l_index]
        z_final = np.concatenate((zs_l_final, [z] * len(tmp1), zs_r_final))
        x_final = np.concatenate((xs_l_final, xs_c, xs_r_final))
        y_final = np.concatenate((ys_l_final, ys_c, ys_r_final))
        t_final = terrain[x_final.astype(int), y_final.astype(int)]
        return (
            abs(fill),
            abs(excave),
            [x_final, y_final, z_final, t_final],
            width,
            is_exceed,
        )
    return (abs(fill), abs(excave), width, is_exceed)


def get_multi_layer_cost(cut1, cut2, fill2):
    """
    这里主要参考Casal等人的论文，只考虑两层土，计算各类挖填方相关的参数指标。注意，
    这里求出的参数还需要乘以线路段的长度才获得最终的cost
    """
    p1, p2, pr, pb, pw, c1, c2, r1, r2, s1, s2 = (
        5,
        2,
        1.1,
        4.5,
        2.0,
        0.97,
        0.95,
        1,
        0.8,
        1.25,
        1.25,
    )
    vc1, vc2, vf = cut1, cut2 - cut1, fill2
    vr = min(vf, vc1 * c1 * r1 + vc2 * c2 * r2)
    vb = vf - vr
    vw = (
        (1 - c1) * s1 * vc1
        + (1 - c2) * s2 * vc2
        + (1.25 / 0.9) * (vc1 * c1 * r1 + vc2 * c2 * r2 - vr)
    )
    # total_cost = p1*vc1+p2*vc2+pr*vr+pb*vb+pw*vw
    return [p1 * vc1 + p2 * vc2, pr * vr, pb * vb, pw * vw]


# if __name__ == "__main__":
#     fig = plt.figure()
#     ax = plt.axes(projection="3d")
#     # X, Y = np.meshgrid(np.arange(150), np.arange(160))
#     # Z = (X + Y)*0.5
#     Z = np.loadtxt('test.txt')
#     Z1 = Z*1000
#     Z2 = Z1-3
#     # ax.plot_surface(X, Y, Z, alpha=0.4, cstride=1, rstride=1, cmap='rainbow')

#     fill2, excave2, [x_final, y_final, z_final, t_final], width, is_exceed = get_cross_fill_cut(
#         657.4279, 854.3994, hxy(657.4279/1000, 854.3994/1000)*1000+0.98, math.atan(-2.731622), Z1, 7.58, 25, math.tan(0.78), math.tan(0.59))
#     fill1, excave1, _, _, _ = get_cross_fill_cut(
#         3657.4279, 854.3994, hxy(657.4279/1000, 854.3994/1000)*1000+0.98, math.atan(-2.731622), Z2, 7.58, 25, math.tan(0.78), math.tan(0.59))
#     total_cost = get_multi_layer_cost(excave1, excave2, fill2)
#     # total_cost = get_multi_layer_cost(60, 60, 60, math.pi, Z, 9.5, 25, 2, 1)
#     print('fill is {}, and cut is {}, width is {}, is exceed is {}'.format(fill2, excave2, width, is_exceed))
#     print(total_cost)
#     ax.plot(x_final, y_final, z_final, '.r')
#     ax.plot(x_final, y_final, t_final, '.k')
#     plt.show()
