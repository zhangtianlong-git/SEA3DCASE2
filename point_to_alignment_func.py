import sympy
import math
import tool_functions as tf
import matplotlib.pyplot as plt
# from casalNumerical import hxy
import numpy as np
from multiprocessing import Pool
from generalcross import get_multi_layer_cost
from generalcross import get_cross_fill_cut
import time
x = sympy.symbols('x')

# print(sympy.integrate(sympy.cos(x), (x, 0, math.pi/4)))


def get_geometry(three_points):
    """
    通过平面交点参数确定线路各个元素的长度，公式参考自Casal等人的论文
    每次输入三个交点点信息，求得中间交点对应的单侧直线段长度，缓和曲线长度和圆曲线长度
    """
    [xim1, yim1, rim1, wim1, xi, yi, ri, wi, xip1, yip1, rip1, wip1] = three_points
    disti = math.sqrt((xi-xim1)**2+(yi-yim1)**2)
    distip1 = math.sqrt((xip1-xi)**2+(yip1-yi)**2)
    # if xi-xim1 >= 0:
    #     phyi = -math.acos((yi-yim1)/disti)
    # else:
    #     phyi = 2*math.pi - math.acos((yi-yim1)/disti)
    # if xip1-xi >= 0:
    #     phyip1 = -math.acos((yip1-yi)/distip1)
    # else:
    #     phyip1 = 2*math.pi - math.acos((yip1-yi)/distip1)
    thetai = abs(math.pi-math.acos(((xim1-xi)*(xip1-xi)+(yim1-yi)*(yip1-yi))/disti/distip1))
    lcci = abs(ri)*wi
    lci = abs(ri)*(thetai-wi)
    tpi = float(sympy.integrate(sympy.cos(x*x/(2*abs(ri)*lci)), (x, 0, lci)))
    pfi = float(sympy.integrate(sympy.sin(x*x/(2*abs(ri)*lci)), (x, 0, lci)))
    phi = pfi*math.tan((thetai-wi)/2)
    hvi = (abs(ri)+pfi/math.cos((thetai-wi)/2)) * \
        math.sin(wi/2)/math.sin((math.pi-thetai)/2)
    len_seg_other = tpi + phi + hvi
    return [disti, lcci, lci, len_seg_other, ri]


def get_geo_lens_and_type(points):
    dist_tmp, len_set, total_len = 0, [], 0
    for i in range(round(len(points)/4-2)):
        [disti, lcci, lci, len_seg_other, ri] = get_geometry(
            points[i*4:i*4+3*4])
        len_set += [[disti-len_seg_other-dist_tmp, None, 'Z'],
                    [lci, ri, 'ZY'], [lcci, ri, 'Y'], [lci, ri, 'YZ']]
        total_len = total_len + (disti-len_seg_other-dist_tmp + lci*2 + lcci)
        dist_tmp = len_seg_other
    dist_final = math.sqrt(
        (points[-4]-points[-8])**2+(points[-3]-points[-7])**2)
    line_final = dist_final - dist_tmp
    total_len = total_len + line_final
    len_set.append([line_final, None, 'Z'])
    return (total_len, len_set)


def get_points_by_len_and_interval(len, interval):
    output, step_lens, iter = [], [], int(len//interval)
    for i in range(iter):
        output.append(interval*(i+1))
        step_lens.append(interval)
    if len % interval != 0:
        output.append(len)
        step_lens.append(len-interval*iter)
    return output, step_lens


def get_type_of_interval(len_set, intervals):
    tmp_id, cur_len, tmp_list, output = 0, 0, [], []
    for i in intervals[:-1]:
        if i - cur_len <= len_set[tmp_id][0]:
            tmp_list += [i - cur_len]
        else:
            tmp_list.append(len_set[tmp_id][2])
            output.append(tmp_list)

            tmp_list = []
            cur_len += len_set[tmp_id][0]
            tmp_list += [i - cur_len]
            tmp_id += 1
    tmp_list += [intervals[-1] - cur_len]
    tmp_list.append(len_set[-1][2])
    output.append(tmp_list)
    return output


def get_start_points(len_set, points):
    output = []
    x_old, y_old = points[0], points[1]
    theta_xy_old = math.atan((points[5]-points[1])/(points[4]-points[0]))
    output.append([x_old, y_old, theta_xy_old, len_set[0][1]])
    for i in len_set:
        arc_l, t = i[0], i[2]
        if t == 'Z' or t == 'Y':   # 计算下一个节点的坐标
            x, y, z, theta_xy, gradient_z = tf.z_y_move(
                x_old, y_old, 0, theta_xy_old, 0, arc_l, radi=i[1])
        elif t == 'ZY':
            x, y, z, theta_xy, gradient_z = tf.spr_move1(
                x_old, y_old, 0, theta_xy_old, 0, arc_l, arc_l, radi=i[1])
        elif t == 'YZ':
            x, y, z, theta_xy, gradient_z = tf.spr_move2(
                x_old, y_old, 0, theta_xy_old, 0, arc_l, arc_l, radi=i[1])
        else:
            raise
        x_old, y_old, theta_xy_old = x, y, theta_xy
        output.append([x_old, y_old, theta_xy_old, i[1]])
    return output


def get_all_xytheta(lens_and_types, start_points, len_set):
    xs, ys, thetas, types = [], [], [], []
    for i in range(len(lens_and_types)):
        lens = lens_and_types[i][:-1]
        t = lens_and_types[i][-1]
        x_old, y_old = start_points[i][0], start_points[i][1]
        theta_xy_old, r = start_points[i][2], len_set[i][1]
        arc_total = len_set[i][0]
        for arc_l in lens:
            if t == 'Z' or t == 'Y':   # 计算下一个节点的坐标
                x, y, z, theta_xy, gradient_z = tf.z_y_move(
                    x_old, y_old, 0, theta_xy_old, 0, arc_l, radi=r)
            elif t == 'ZY':
                x, y, z, theta_xy, gradient_z = tf.spr_move1(
                    x_old, y_old, 0, theta_xy_old, 0, arc_l, arc_total, radi=r)
            elif t == 'YZ':
                x, y, z, theta_xy, gradient_z = tf.spr_move2(
                    x_old, y_old, 0, theta_xy_old, 0, arc_l, arc_total, radi=r)
            else:
                raise
            xs.append(x), ys.append(y), thetas.append(
                theta_xy), types.append(t)
    return (xs, ys, thetas, types)


def show_xy_and_z(xs, ys, types, thetas, intervals, zs_final, zs_line, new_gs):
    terrains = []
    plt.plot(points[0], points[1], '.r')
    for i, j, t, a in zip(xs, ys, types, thetas):
        terrains.append(terrain[round(i), round(j)])
        if t == 'Z':
            plt.plot(i, j, '.r')
        elif t == 'Y':
            plt.plot(i, j, '.g')
        else:
            plt.plot(i, j, '.b')
        # plt.plot([i, i+resolution], [j, j+resolution*math.tan(a)], '-y')
    plt.xlim([0, 620]), plt.ylim([0, 410])
    plt.show()
    plt.figure()
    plt.plot([0], [z_start], '.r')
    plt.plot(intervals, zs_final, '.-r')
    plt.plot(intervals, terrains, '.-k')
    # plt.plot(intervals, zs_line, '.g')
    # for x, z, g in zip(intervals, zs_final, new_gs):
    #     plt.plot([x, x+resolution*0.5], [z, z+g*resolution*0.5], '-c')
    # plt.xlim([0, 4.5]), plt.ylim([1.56, 1.66])
    plt.show()


def get_zs_and_Ts(bpd_lens, slopes, ks, z_start, z_end, total_len):
    zs, Ts, max_delta_zs, z_old, lens = [z_start], [], [], z_start, [0]
    new_lens = [0] + bpd_lens + [1]
    for i in range(1, len(bpd_lens)+1):
        z_new = z_old + \
            (new_lens[i]-new_lens[i-1])*slopes[i-1]
        zs.append(z_new), lens.append(new_lens[i])
        z_old = z_new
    zs.append(z_end), lens.append(total_len)
    new_slopes = slopes + [(zs[-1]-zs[-2])/(lens[-1]-lens[-2])]
    for i in range(1, len(slopes)+1):
        t_tmp = abs(new_slopes[i]-new_slopes[i-1])*ks[i-1]/2
        Ts.append(t_tmp)
        max_delta_zs.append((t_tmp**2)/(2*ks[i-1]))
    Ts = [0] + Ts + [0]
    max_delta_zs = [0] + max_delta_zs + [0]
    return (zs, Ts, max_delta_zs, lens, new_slopes)


def get_final_zs(intervals, new_slopes, lens, max_delta_zs):
    tmp_id, zs_line, zs_offset, signs, new_gs = 0, [], [], [], []
    for i in range(len(new_slopes)-1):
        signs.append(np.sign(new_slopes[i+1]-new_slopes[i]))
    signs = [0] + signs + [0]

    for i in intervals:
        if lens[tmp_id] < i <= lens[tmp_id + 1]:
            zs_line.append((i-lens[tmp_id])*new_slopes[tmp_id]+zs[tmp_id])
            if i < lens[tmp_id] + Ts[tmp_id]:
                zs_offset.append(
                    signs[tmp_id]*((1-(i-lens[tmp_id])/Ts[tmp_id])**2)*max_delta_zs[tmp_id])
                tmp_ratio = (i-lens[tmp_id]+Ts[tmp_id])/(Ts[tmp_id]*2)
                tmp_g = tmp_ratio * \
                    (new_slopes[tmp_id]-new_slopes[tmp_id-1]) + \
                    new_slopes[tmp_id-1]
                new_gs.append(tmp_g)
            elif i > lens[tmp_id + 1] - Ts[tmp_id + 1]:
                zs_offset.append(
                    signs[tmp_id+1]*((1-(lens[tmp_id+1]-i)/Ts[tmp_id+1])**2)*max_delta_zs[tmp_id+1])
                tmp_ratio = (i-lens[tmp_id+1]+Ts[tmp_id+1])/(Ts[tmp_id+1]*2)
                tmp_g = tmp_ratio * \
                    (new_slopes[tmp_id+1]-new_slopes[tmp_id]) + \
                    new_slopes[tmp_id]
                new_gs.append(tmp_g)
            else:
                zs_offset.append(0)
                new_gs.append(new_slopes[tmp_id])
        else:
            tmp_id += 1
            zs_line.append((i-lens[tmp_id])*new_slopes[tmp_id]+zs[tmp_id])
            if i < lens[tmp_id] + Ts[tmp_id]:
                zs_offset.append(
                    signs[tmp_id]*((1-(i-lens[tmp_id])/Ts[tmp_id])**2)*max_delta_zs[tmp_id])
                tmp_ratio = (i-lens[tmp_id]+Ts[tmp_id])/(Ts[tmp_id]*2)
                tmp_g = tmp_ratio * \
                    (new_slopes[tmp_id]-new_slopes[tmp_id-1]) + \
                    new_slopes[tmp_id-1]
                new_gs.append(tmp_g)
            else:
                zs_offset.append(0)
                new_gs.append(new_slopes[tmp_id])
    zs_final = []
    for i, j in zip(zs_line, zs_offset):
        zs_final.append(i+j)
    return (zs_final, zs_line, new_gs)


if __name__ == '__main__':
    ZTMP = np.loadtxt("dem.txt")
    xx = np.arange(ZTMP.shape[0])
    yy = np.arange(ZTMP.shape[1])
    X, Y = np.meshgrid(xx, yy)
    terrain = (np.flip(ZTMP, axis=0)).T
    points = list(np.loadtxt('points.txt'))
    resolution = 1
    bpd_lens = list(np.loadtxt('bpd_lens.txt'))
    slopes = list(np.loadtxt('slopes.txt'))
    ks = [5800] * len(slopes)
    z_start, z_end = terrain[round(points[0]), round(points[1])], terrain[round(points[-4]), round(points[-3])]
    # todo 总长度计算错误，检查
    total_len, len_set = get_geo_lens_and_type(points)
    intervals, step_lens = get_points_by_len_and_interval(
        total_len, resolution)
    lens_and_types = get_type_of_interval(len_set, intervals)
    start_points = get_start_points(len_set, points)
    xs, ys, thetas, types = get_all_xytheta(
        lens_and_types, start_points, len_set)
    zs, Ts, max_delta_zs, lens, new_slopes = get_zs_and_Ts(
        bpd_lens, slopes, ks, z_start, z_end, total_len)
    zs_final, zs_line, new_gs = get_final_zs(
        intervals, new_slopes, lens, max_delta_zs)
    show_xy_and_z(xs, ys, types, thetas, intervals, zs_final, zs_line, new_gs)

    start_time = time.time()
    t_fill1, t_fill2, t_cut1, t_cut2, land_cost = 0, 0, 0, 0, 0
    clear_cost, pave_cost, maintain_cost = 0, 0, 0
    for x, y, z_final, a, g, s, interv in zip(xs, ys, zs_final, thetas, new_gs, step_lens, intervals):
        if round(interv*1000) == 1000:
            print('pause!!')
        fill2, cut2, width, _ = get_cross_fill_cut(
            x, y, z_final, a, terrain, 4.58, 20, math.tan(0.78), math.tan(0.59))
        fill1, cut1, _, _ = 0, 0, 0, 0
        t_fill1, t_fill2, t_cut1, t_cut2 = t_fill1 + fill1 * \
            s, t_fill2 + fill2*s, t_cut1 + cut1*s, t_cut2 + cut2*s
        land_cost += (width+6)*7*s
        clear_cost += 0.6*width*s*(1+g**2)**0.5
        pave_cost += 327*s*(1+g**2)**0.5
        maintain_cost += 60*s*(1+g**2)**0.5
    total_soil_cost = np.array(get_multi_layer_cost(t_cut1, t_cut2, t_fill2)) 
    all_cost = sum(total_soil_cost) + land_cost + clear_cost + pave_cost + maintain_cost
    end_time = time.time()
    print('soil cost is {}, land cost is {}, clear_cost is {}, pave cost is {}, maintain cost is {}'.format(total_soil_cost, land_cost, clear_cost, pave_cost, maintain_cost))
    print('all cost is {}'.format(all_cost))
    print("Time elapsed: ", end_time - start_time)
