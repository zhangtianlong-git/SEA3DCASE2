import math
import tool_functions as tf
import matplotlib.pyplot as plt
import numpy as np
import heapq
import copy
from generalcross import get_cross_fill_cut, get_multi_layer_cost

"""输入约束和定义参数"""
MAX_GRADIENT = 0.07                    # 最大坡度值
MIN_GRADIENT = 0.005                   # 最小坡度
GRADIENT_Z_RESOLUTION = 0.005          # 坡度分辨率
N_STEER = 10                           # 平面最大探索方向数
LEN_SPIRAL = 20                        # 缓和曲线长度
EXPLORE_RES = 10                       # 直线和圆曲线探索步长
MIN_LEN_SLOPE = 110                    # 最小坡长
MIN_R = 15                            # 最小曲线半径
MIN_LEN_CURV = 15                     # 最短曲线长度
MIN_LEN_TAN = 20                      # 最短直线长度
THETA_XY_RESOLUTION = np.deg2rad(1)    # 平面角度分辨率
K_MIN = 5800                          # Cacal论文中的K，K*坡度差=竖曲线长度,即竖曲线半径
MAX_LEN_TAN = 300                     # 最大直线长度
ZMAX = 400                            # 最大高程差
"""----------------"""

# 平面角度补偿值，避免出现负值
THETA_XY_IND = round(math.pi/THETA_XY_RESOLUTION)
# 竖向角度补偿值，避免出现负值
GRADIENT_Z_IND = round(MAX_GRADIENT/GRADIENT_Z_RESOLUTION)
# 最小曲线长度的线元段数，其减一才为实际的的圆曲线段数
MIN_SEG = MIN_LEN_CURV/EXPLORE_RES
# 最小曲线长度的线元段数，其减一才为实际的的圆曲线段数
MIN_SEG_T = MIN_LEN_TAN/EXPLORE_RES
# 最小坡长对应的段数
MIN_SEG_S = MIN_LEN_SLOPE/EXPLORE_RES
# 行进探索步长的转角为最大转角，主要用来确定角度分辨率
MAX_ANGLE_CHANGE = EXPLORE_RES / MIN_R
# 所有的转角
ANGLES = list(np.linspace(-MAX_ANGLE_CHANGE,
              MAX_ANGLE_CHANGE, N_STEER))
# 所有的探索半径
RADIUS = EXPLORE_RES / np.linspace(-MAX_ANGLE_CHANGE,
                                   MAX_ANGLE_CHANGE, N_STEER)
RADIUS = list(((RADIUS/10).round(0))*10)
# 所有的坡度值
GRADIENTS = np.linspace(-MAX_GRADIENT, MAX_GRADIENT,
                        int(2*MAX_GRADIENT/GRADIENT_Z_RESOLUTION+1))
GRADIENTS = list(GRADIENTS[abs(GRADIENTS) >= MIN_GRADIENT-0.00001])

# 地形参数等超大型的矩阵先存为txt，再读取，提升计算效率
ZTMP = np.loadtxt('dem.txt')
x = np.arange(ZTMP.shape[0])
y = np.arange(ZTMP.shape[1])
X, Y = np.meshgrid(x, y)
terrain = (np.flip(ZTMP, axis=0)).T
terrain1 = terrain - 3
heuristic = np.loadtxt('heuristic.txt')


class Node:

    def __init__(self, xind, yind, zind, theta_xy_ind, gradient_z_ind,
                 xlist, ylist, zlist, theta_xy_list, gradient_z_list=[0],
                 pind=None, cost=0, line_type=None, radius=None,
                 seqZ=0, seqY=0, seqS=0, lens=0, start=False, seqSS=0,
                 pre_ver_cur_len=0, fill_cut_volunm=[], loc_and_len_cost=[],
                 cross_section_cost=[]):
        self.xind = xind    # x坐标的整数部分，用于检查重复探索，提升探索效率
        self.yind = yind
        self.zind = zind
        self.theta_xy_ind = theta_xy_ind      # 探索节点水平方位角（取整）
        self.gradient_z_ind = gradient_z_ind  # 探索节点坡度（取整）
        self.xlist = xlist  # x坐标，是一个list
        self.ylist = ylist
        self.zlist = zlist
        self.theta_xy_list = theta_xy_list
        self.gradient_z_list = gradient_z_list
        self.pind = pind    # 上一个探索节点（父节点）的指针
        self.cost = cost    # 总成本
        self.radius = radius                   # 当前探索节点半径
        self.lens = lens  # 当前节点至起点的线路总长度（水平方向）
        self.line_type = line_type             # 'Z'表示直线，‘Y’表示圆，‘ZH’和‘HZ’为缓和曲线
        self.seqZ = seqZ    # 连续直线段的长度
        self.seqY = seqY    # 连续圆曲线的长度
        self.seqS = seqS    # 连续坡的长度
        self.seqSS = seqSS  # 连续坡的长度，但会被缓和曲线重置，用以考虑平包纵
        self.start = start  # 是否为起始点，起始点有一些特殊处理
        self.pre_ver_cur_len = pre_ver_cur_len  # 上一个竖曲线的长度的一半
        self.fill_cut_volunm = fill_cut_volunm  # 用于记录挖填方量，每次探索重新计算挖填方成本
        self.loc_and_len_cost = loc_and_len_cost  # 距离与占地相关的成本，定义见casal等的论文
        self.cross_section_cost = cross_section_cost  # 挖填方成本，包含挖方成本，回填成本等
        self.get_pre_constrain_inf()            # 获取哪些几何约束得到了满足，用以确定接下来的探索方式
        # potential explorations, [[线类型，半径，是否线形变化, 坡度，是否坡度变化, 是否被缓和曲线重置],...]
        self.pe = self.get_pe()
        # 若当前节点是竖曲线后的第一个节点，获取当前节点之前竖曲线上需要调整z坐标的节点和z的调整量
        self.lists_of_ver_curv_nodes_deltazs = self.get_nodes_deltazs()

    def get_nodes_deltazs(self):
        """
        若是竖曲线后第一个节点，向前回溯获取竖曲线内的节点信息，计算delta_z和新的坡度new_gs
        """
        if self.is_first_out_slope:
            grd_current = self.gradient_z_list[-1]
            # 判断是否仅有一个探索节点在竖曲线内
            not_only_one = False if (
                self.pre_ver_cur_len <= EXPLORE_RES) else True
            # 竖曲线最大外距，即变坡点与竖曲线的z方向差值
            max_deltaz = (self.pre_ver_cur_len**2)/(2*K_MIN)
            tmp, iter, nodes, deltazs, new_g_tmp = self.pind, 0, [], [], []
            if not_only_one:
                while True:
                    iter = iter + 1
                    # 当前节点到上个变坡点之间距离与半坡长的比率，类似相似三角形
                    tmp_ratio = tmp.seqS/self.pre_ver_cur_len
                    # 记录比率，在下个for循环中求新的坡度gs，思路为坡度沿竖曲线方向呈线形关系
                    new_g_tmp.append(0.5*tmp_ratio)
                    ratio = 1 - tmp_ratio
                    # 相似三角形求delta_z
                    deltaz = max_deltaz*ratio**2
                    nodes.append(tmp), deltazs.append(deltaz)
                    if tmp.seqS == EXPLORE_RES:
                        # delta_z的符号，凹为正，凸为负
                        sign_z = 1 if tmp.gradient_z_list[-1] - \
                            tmp.pind.gradient_z_list[-1] > 0 else -1
                        tmp = tmp.pind
                        break
                    tmp = tmp.pind
                # 变坡点节点，其坡度对应上个坡度
                nodes.append(tmp)
                grd_old = tmp.gradient_z_list[-1]
                # delta_z是对称的
                deltazs = deltazs + [max_deltaz] + deltazs[-1::-1]
                new_g_l, new_g_r = [], []
                for i in new_g_tmp:
                    # 坡度类似delta_z，具有对称性
                    new_g_l.append((0.5-i)*(grd_current-grd_old)+grd_old)
                    new_g_r.insert(0, (0.5+i)*(grd_current-grd_old)+grd_old)
                    tmp = tmp.pind
                    nodes.append(tmp)
                # 变坡点出的坡度为左右坡度的一半
                new_gs = new_g_l + [0.5*(grd_current+grd_old)] + new_g_r
                new_gs.reverse()
            else:
                # 只有一个探索节点位于竖曲线内的情况，是上述情况的简化版
                grd_l, grd_r = tmp.gradient_z_list[-1], self.gradient_z_list[-1]
                sign_z = 1 if grd_r - grd_l > 0 else -1
                nodes.append(tmp), deltazs.append(max_deltaz)
                new_gs = [0.5*(grd_l+grd_r)]
            return [nodes, deltazs, sign_z, new_gs]
        else:
            return None

    # 当前节点之前的约束，由于考虑竖曲线，还需考虑当前节点之后的约束
    def get_pre_constrain_inf(self):
        # 连续坡长约束
        self.enough_pre_slope_len_vertical = (
            self.seqS >= (MIN_LEN_SLOPE+self.pre_ver_cur_len))
        # 考虑水平竖缓不重合约束，由于SeqSS<=seqS，所以这里只用判断一个seqSS即可
        self.enough_pre_slope_len_horizontal = (
            self.seqSS >= self.pre_ver_cur_len)
        # 水平直线长度约束
        self.enough_tangent_len = (self.seqZ >= MIN_LEN_TAN)
        # 水平曲线长度约束
        self.enough_curve_len = (self.seqY >= MIN_LEN_CURV)
        # 为了计算竖曲线上的纵坐标，判断当前节点是否是曲线和直线交界处的那个点
        self.is_first_out_slope = True if (
            (0 <= (self.seqS-self.pre_ver_cur_len) < EXPLORE_RES) and
            (self.pre_ver_cur_len != 0)) else False

    # 为了考虑竖曲线，考虑当前节点之后的约束
    def get_next_constrain_inf(self, g_old, g_new):
        # 确定下个变坡的坡度后的当前坡长，需要和上个坡长合并起来确定最终的夹直线长度约束
        next_ver_cur_len = abs(K_MIN*(g_old-g_new))/2
        # 总体的坡长约束
        self.enough_next_slope_len_vertical = (
            self.seqS >= (MIN_LEN_SLOPE+self.pre_ver_cur_len+next_ver_cur_len))
        # 防止新坡度产生的坡长与缓和曲线重合
        self.enough_next_slope_len_horizontal = (
            self.seqSS >= next_ver_cur_len)

    def get_pe(self):
        lt = self.line_type
        pe = []  # potential explorations
        grd = self.gradient_z_list[-1]
        if lt == 'Z':
            # [[线类型，半径，是否线形变化, 坡度，是否坡度变化, 是否被缓和曲线重置],...]
            if self.seqZ <= MAX_LEN_TAN:
                pe.append(['Z', None, False, grd, False, False])
            # 在直线段上进行变坡，满足坡长要求和竖缓不重合
            if (self.enough_pre_slope_len_vertical) or (self.start):
                for j in GRADIENTS:
                    change_grd = not (grd == j)
                    if change_grd:
                        self.get_next_constrain_inf(grd, j)
                        if self.start or (self.enough_next_slope_len_vertical and self.enough_next_slope_len_horizontal):
                            pe.append(['Z', None, False, j, change_grd, False])
            # 如果满足平面几何约束，并且考虑竖缓不重合，还可增添圆曲线的缓和曲线，但坡度不发生变化
            if (self.enough_tangent_len and self.enough_pre_slope_len_horizontal) or (self.start):
                for i in RADIUS:
                    pe.append(['ZY', i, True, grd, False, False])
        elif lt == 'Y':
            # 防止出现回头曲线,判断条件为偏角不超过PI
            if (self.seqY) >= abs(self.radius)*3.14:
                return pe
            # [[线类型，半径，是否线形变化, 坡度，是否坡度变化, 是否被缓和曲线重置],...]
            pe.append(['Y', self.radius, False, grd, False, False])
            # 在圆曲线上进行变坡，满足坡长要求和竖缓不重合
            if (self.enough_pre_slope_len_vertical):
                for j in GRADIENTS:
                    change_grd = not (grd == j)
                    if change_grd:
                        self.get_next_constrain_inf(grd, j)
                        if (self.enough_next_slope_len_vertical and self.enough_next_slope_len_horizontal):
                            pe.append(
                                ['Y', self.radius, False, j, change_grd, False])
            # 在上述基础上，如果符合平面几何约束，还可以添加向直线过渡的缓和曲线，但坡度不变
            if (self.enough_curve_len and self.enough_pre_slope_len_horizontal):
                pe.append(['YZ', self.radius, True, grd, False, False])
        elif lt == 'ZY':
            pe.append(['Y', self.radius, True, grd, False, True])
        elif lt == 'YZ':
            pe.append(['Z', None, True, grd, False, True])
        return pe


class Config:
    def __init__(self, xw, yw, zw):
        self.xw = xw+1
        self.yw = yw+1
        self.zw = zw+1
        self.theta_xyw = 2*THETA_XY_IND+1
        self.gradient_zw = 2*GRADIENT_Z_IND+1
        self.minx = 0
        self.miny = 0
        self.minz = 0
        self.min_theta_xy = -THETA_XY_IND
        self.min_gradient_z = -GRADIENT_Z_IND


def get_neighbors(current, config):
    pe = current.pe
    if pe:
        x_old, y_old, z_old = current.xlist[-1], current.ylist[-1], current.zlist[-1]
        theta_xy_old = current.theta_xy_list[-1]
        # 数字拷贝，不能直接赋值
        seqZ, seqY = current.seqZ, current.seqY
        for i in pe:
            # [[线类型，半径，是否线形变化, 坡度，是否坡度变化, 是否被缓和曲线重置],...]
            t, r, c, g, gc, gs = i[0], i[1], i[2], i[3], i[4], i[5]
            # 更新连续平曲线或平直线长度
            if t == 'Z' or t == 'Y':   # 计算下一个节点的坐标
                arc_l = EXPLORE_RES  # 此次探索的长度
                seqZ, seqY = 0, 0
                if t == 'Z':
                    if c:
                        seqZ = EXPLORE_RES
                    else:
                        seqZ = current.seqZ + arc_l
                elif t == 'Y':
                    if c:
                        seqY = EXPLORE_RES
                    else:
                        seqY = current.seqY + arc_l

                x, y, z, theta_xy, gradient_z = tf.z_y_move(
                    x_old, y_old, z_old, theta_xy_old, g, EXPLORE_RES, radi=r)
            elif t == 'ZY':
                arc_l, seqZ, seqY = LEN_SPIRAL, 0, 0
                x, y, z, theta_xy, gradient_z = tf.spr_move1(
                    x_old, y_old, z_old, theta_xy_old, g, LEN_SPIRAL, LEN_SPIRAL, radi=r)
            elif t == 'YZ':
                arc_l, seqZ, seqY = LEN_SPIRAL, 0, 0
                x, y, z, theta_xy, gradient_z = tf.spr_move2(
                    x_old, y_old, z_old, theta_xy_old, g, LEN_SPIRAL, LEN_SPIRAL, radi=r)
            else:
                raise
            theta_xy = tf.pi_2_pi(theta_xy)

            # 更新连续坡段长度和上个坡的坡长的一半
            (seqS, pre_vc) = (EXPLORE_RES, abs(K_MIN*(current.gradient_z_list[-1]-g))/2) if gc else (
                current.seqS + arc_l, current.pre_ver_cur_len)
            # 起点的上个坡的坡长假定为0
            pre_vc = 0 if current.start else pre_vc
            seqSS = EXPLORE_RES if (gc or gs) else current.seqSS + arc_l

            if x < 0 or x > config.xw or y < 0 or y > config.yw:  # 检查越界
                continue

            is_exceed, cost, c_s_cost, loc_len_cost, f_c_list = get_cost(
                x, y, z, theta_xy, arc_l, g, current)
            if is_exceed:  # 横截面上坡度不能与地面线相交
                continue

            lens = current.lens + arc_l

            node = Node(round(x), round(y), round(z),
                        round(theta_xy/THETA_XY_RESOLUTION),
                        round(gradient_z/GRADIENT_Z_RESOLUTION),
                        [x], [y], [z], [theta_xy], [gradient_z],
                        line_type=t, cost=cost, radius=r, pind=current,
                        seqY=seqY, seqZ=seqZ, seqS=seqS, seqSS=seqSS,
                        pre_ver_cur_len=pre_vc, lens=lens,
                        fill_cut_volunm=f_c_list,
                        loc_and_len_cost=loc_len_cost,
                        cross_section_cost=c_s_cost)

            yield node


def get_cost(x, y, z, a, s, g, current):
    if current.is_first_out_slope:  # 若是竖曲线后的第一个节点，需要向前回溯重新计算挖填方等
        [nodes, deltazs, sign_z, new_gs] = current.lists_of_ver_curv_nodes_deltazs
        # 竖曲线之前的第一个节点开始重新计算
        f_c_list_tmp = copy.deepcopy(nodes[-1].pind.fill_cut_volunm)
        loc_len_cos_tmp = copy.deepcopy(nodes[-1].pind.loc_and_len_cost)
        # 需要把当前节点包含在重新计算的循环中，保障连续性
        for node, dz, new_g in zip(nodes+[current], deltazs+[0], new_gs+[g]):
            # 更新每个探索节点新的z坐标，其他坐标不变
            tx, ty, tz, ta = node.xlist[-1], node.ylist[-1], node.zlist[-1] + \
                sign_z*dz, node.theta_xy_list[-1]
            # 竖曲线上都为直线或圆曲线探索节点，所以探索长度都为EXPLORE_RES
            f_c_list_in, loc_len_cos_in, is_exceed = calculate_cost(
                f_c_list_tmp, loc_len_cos_tmp, tx, ty, tz, ta, EXPLORE_RES, new_g)
            if is_exceed:
                return (is_exceed, None, None, None, None)
            f_c_list_tmp, loc_len_cos_tmp = f_c_list_in, loc_len_cos_in
    else:
        f_c_list_in = current.fill_cut_volunm
        loc_len_cos_in = current.loc_and_len_cost

    f_c_list, loc_len_cos, is_exceed = calculate_cost(
        f_c_list_in, loc_len_cos_in, x, y, z, a, s, g)
    if is_exceed:
        return (is_exceed, None, None, None, None)
    # 计算横截面挖填方成本
    cross_section_cost = get_multi_layer_cost(
        f_c_list[2], f_c_list[3], f_c_list[1])
    # 计算总成本呢
    total_cost = sum(cross_section_cost) + sum(loc_len_cos)
    return (is_exceed, total_cost, cross_section_cost, loc_len_cos, f_c_list)


def calculate_cost(f_c_list_in, loc_len_cost_in, x, y, z, a, s, g):
    [tf1_0, tf2_0, tc1_0, tc2_0] = f_c_list_in
    [lc0, cc0, pc0, mc0] = loc_len_cost_in
    # 计算两层土对应的挖方和填方量，上层土为土层"2"
    tmp_f2, tmp_c2, tmp_wid, is_exceed = get_cross_fill_cut(
        x, y, z, a, terrain, 4.58, 20, math.tan(0.78), math.tan(0.59))
    if is_exceed:
        return (None, None, is_exceed)
    tmp_f1, tmp_c1, _, _ = get_cross_fill_cut(
        x, y, z, a, terrain1, 4.58, 20, math.tan(0.78), math.tan(0.59))
    tf1_0, tf2_0, tc1_0, tc2_0 = tf1_0+tmp_f1*s, tf2_0+tmp_f2 * \
        s, tc1_0+tmp_c1*s, tc2_0+tmp_c2*s
    # 计算占地成本，清理成本，铺路成本和养护成本
    lc0 = lc0 + (tmp_wid+6)*7*s
    cc0 = cc0 + 0.6*tmp_wid*s*(1+g**2)**0.5
    pc0 = pc0 + 327*s*(1+g**2)**0.5
    mc0 = mc0 + 60*s*(1+g**2)**0.5
    f_c_list = [tf1_0, tf2_0, tc1_0, tc2_0]
    loc_len_cos = [lc0, cc0, pc0, mc0]
    return (f_c_list, loc_len_cos, is_exceed)


def distance_point_to_line(A, B):
    """
    A是一个元组，包含向量A的x，y坐标和方位角；B是一个元组，包含点B的x，y坐标。
    函数返回点B到向量A的垂线距离
    """
    x1, y1, angle = A
    x2, y2 = B
    k = math.tan(angle)
    if k == 0:
        return abs(y2 - y1)
    else:
        k2 = -1 / k
        b1 = y1 - k * x1
        b2 = y2 - k2 * x2
        x = (b2 - b1) / (k - k2)
        y = k * x + b1
        return math.sqrt((x - x2) ** 2 + (y - y2) ** 2)


def distance_point_to_point(A, B):
    x1, y1 = A
    x2, y2 = B
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def intersection(x1, y1, radian1, x2, y2, radian2):
    # 计算方向向量
    v1 = np.array([np.cos(radian1), np.sin(radian1)])
    v2 = np.array([np.cos(radian2), np.sin(radian2)])
    # 计算点坐标
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    # 计算叉乘
    cross = np.cross(v2, v1)
    # 计算参数t
    t = np.cross(p1 - p2, v2) / cross
    # 计算交点坐标
    inter = p1 + t * v1
    return inter


# 主程序部分
plt.xlim(0,614)
plt.ylim(0,410)
ox, oy = [], []
(xw, yw) = terrain.shape
gx = 556.61   # [m]
gy = 196.75  # [m]
sx = 40.78  # [m]
sy = 304.66  # [m]
start = [sx, sy, terrain[round(sx), round(sy)], tf.pi_2_pi(np.deg2rad(-55))]
goal = [gx, gy, terrain[round(gx), round(gy)], tf.pi_2_pi(np.deg2rad(0))]
"""
挖填方体积：fill_cut_volunm，包含两层土，分别为[fill1, fill2, cut1, cut2]， "1"表示下层土
线路平面相关成本：loc_and_len_cost，包含[占地成本，清理成本，铺路成本，养护成本]
"""
nstart = Node(round(start[0]), round(start[1]), round(start[2]),
              round(start[3]/THETA_XY_RESOLUTION),
              0,
              [start[0]], [start[1]], [start[2]], [start[3]],
              start=True, line_type='Z', fill_cut_volunm=[0, 0, 0, 0],
              loc_and_len_cost=[0, 0, 0, 0])
ngoal = Node(round(goal[0]), round(goal[1]), round(goal[2]),
             round(goal[3]/THETA_XY_RESOLUTION),
             0,
             [goal[0]], [goal[1]], [goal[2]], [goal[3]])

tox, toy, iter = ox[:], oy[:], 0
# 这里的z取大一点避免出现节点指标计算的误判
config = Config(xw, yw, ZMAX)
openList, closedList = {}, {}
pq = []
openList[tf.calc_index(nstart, config)] = nstart
heapq.heappush(pq, (tf.calc_cost_new(nstart, heuristic),
               tf.calc_index(nstart, config)))
num_iter = 0

while True:
    if not openList:
        print("Error: Cannot find path, No open set")
        break

    cost, c_id = heapq.heappop(pq)
    if c_id in openList:
        current = openList.pop(c_id)
        closedList[c_id] = current
    else:
        continue

    num_iter += 1
    if num_iter % 50000 == 0:
        print(num_iter)
        plt.pause(0.001)
        # break
    
    if num_iter % 5000 == 0:
        plt.plot(current.xlist, current.ylist, 'xc')

    if abs(current.xind-ngoal.xind)+abs(current.yind-ngoal.yind) < 2*EXPLORE_RES:
        cx, cy, cz, ctheta, cg = current.xlist[-1], current.ylist[-1], current.zlist[
            -1], current.theta_xy_list[-1], current.gradient_z_list[-1]
        gx, gy, gz = goal[0], goal[1], goal[2]
        dis_points = distance_point_to_point((cx, cy), (gx, gy))
        dis_p_to_l = distance_point_to_line((cx, cy, ctheta), (gx, gy))
        delta_z = abs(cz + dis_points*cg - gz)
        # print('cuurent total cost is {}'.format(current.cost))
        if (current.line_type == 'Z') and (dis_points <= EXPLORE_RES) and (
                dis_p_to_l <= 1) and (current.enough_pre_slope_len_vertical) and (
                delta_z < 0.5):
            is_exceed, cost, c_s_cost, loc_len_cost, f_c_list = get_cost(
                gx, gy, gz, ctheta, dis_points, cg, current)
            print('delta z is {} m'.format(delta_z))
            print('cuurent total cost is {}'.format(cost))
            print('cuurent cross_section_cost cost is {}'.format(c_s_cost))
            print('cuurent loc_and_len_cost cost is {}'.format(loc_len_cost))
            break

    for neighbor in get_neighbors(current, config):
        neighbor_index = tf.calc_index(neighbor, config)
        if neighbor_index in closedList:
            continue
        if neighbor not in openList or openList[neighbor_index].cost > neighbor.cost:
            heapq.heappush(pq, (tf.calc_cost_new(
                neighbor, heuristic), neighbor_index))
            openList[neighbor_index] = neighbor

# 终点角度旋转
tmp0 = current
while tmp0.line_type != 'Y':
    tmp0 = tmp0.pind
tx, ty, ttheta, tr = tmp0.xlist[-1], tmp0.ylist[-1], tmp0.theta_xy_list[-1], tmp0.radius
ox, oy = tx - tr*math.sin(ttheta), ty + tr*math.cos(ttheta)
# plt.plot([ox, tx], [oy, ty], '.-k')
p_to_l = distance_point_to_line((cx, cy, ctheta), (ox, oy))
p_to_p = distance_point_to_point((ox, oy), (gx, gy))
angle_goal_to_center = math.atan((gy-oy)/(gx-ox))
d_theta = math.asin(p_to_l/p_to_p)
# 终点旋转后的角度
angle_goal = angle_goal_to_center + d_theta*np.sign(tr)
if abs(ctheta - angle_goal) > math.pi/2:
    angle_goal = angle_goal + math.pi
# 终点前一个圆的圆心角旋转角度，正为增大圆弧长度，负为减小圆弧长度
rotate_angle = (angle_goal - ctheta)*np.sign(tr)

# 求水平交点的位置坐标和变坡点的里程
pi_x, pi_y, pi_r, pi_w = [gx], [gy], [0], [0]
old_x, old_y, old_angle = gx, gy, angle_goal
bpd_lens, bpd_gs = [], []
tmp = current
while tmp.pind is not None:
    if tmp.line_type == 'YZ':
        tmp_r, cur_len = tmp.pind.radius, tmp.pind.seqY
        pi_r = [tmp_r] + pi_r
        pi_w = [abs(cur_len/tmp_r)] + pi_w
    if tmp.line_type == 'ZY':
        new_x, new_y = tmp.pind.xlist[-1], tmp.pind.ylist[-1]
        new_angle = tmp.pind.theta_xy_list[-1]
        [x_t, y_t] = intersection(
            old_x, old_y, old_angle, new_x, new_y, new_angle)
        pi_x = [x_t] + pi_x
        pi_y = [y_t] + pi_y
        old_x, old_y, old_angle = new_x, new_y, new_angle
    if tmp.gradient_z_list[-1] != tmp.pind.gradient_z_list[-1]:
        bpd_gs = tmp.pind.gradient_z_list + bpd_gs
        bpd_lens = [tmp.pind.lens] + bpd_lens
    tmp = tmp.pind
pi_w[-2] = pi_w[-2] + rotate_angle
pi_x.insert(0, start[0]), pi_y.insert(
    0, start[1]), pi_r.insert(0, 0), pi_w.insert(0, 0)
out_points = []
for x, y, r, w in zip(pi_x, pi_y, pi_r, pi_w):
    out_points += [x, y, r, abs(w)]
np.savetxt('points.txt', np.array(out_points))
np.savetxt('slopes.txt', np.array(bpd_gs[1:]))
np.savetxt('bpd_lens.txt', np.array(bpd_lens[1:]))
plt.plot(pi_x, pi_y, '.-y')

# 绘出探索节点的x-y坐标散点图
tmp = current
while tmp.pind is not None:
    if tmp.line_type == 'Z' or tmp.line_type == 'YZ':
        plt.plot(tmp.xlist[-1], tmp.ylist[-1], '.r')
    else:
        plt.plot(tmp.xlist[-1], tmp.ylist[-1], '.g')
    tmp = tmp.pind
plt.plot(tmp.xlist[-1], tmp.ylist[-1], '.r')

plt.plot(start[0], start[1], '*r')
plt.plot(goal[0], goal[1], '*r')
plt.axis("equal")
plt.figure()

# 绘出纵断面图，水平轴为里程
tmp = current
px, pz, pt = [], [], []
while tmp.pind is not None:
    px.append(tmp.lens), pz.append(tmp.zlist[-1]
                                   ), pt.append(terrain[tmp.xind][tmp.yind])
    # print('seqS={}, len={}'.format(tmp.seqS, tmp.lens))
    if tmp.gradient_z_list[-1] != tmp.pind.gradient_z_list[-1]:
        plt.plot(tmp.pind.lens, tmp.pind.zlist[-1], 'oy')
        # print('seqS={}, len={}........'.format(tmp.pind.seqS, tmp.pind.lens))
    if tmp.is_first_out_slope:
        [nodes, deltazs, sign_z, new_gs] = tmp.lists_of_ver_curv_nodes_deltazs
        for i in range(len(deltazs)):
            tmp_x, tmp_y = nodes[i].lens, nodes[i].zlist[-1]+sign_z*deltazs[i]
            plt.plot(tmp_x, tmp_y, 'oc')
            plt.plot([tmp_x, tmp_x + EXPLORE_RES],
                     [tmp_y, tmp_y + EXPLORE_RES*new_gs[i]], '-c')
    tmp = tmp.pind
px.append(tmp.lens), pz.append(tmp.zlist[-1]
                               ), pt.append(terrain[tmp.xind][tmp.yind])
plt.plot(px, pz, '-r')
plt.plot(px, pt, '-.k')
plt.show()
