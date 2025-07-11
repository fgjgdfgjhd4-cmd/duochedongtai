import numpy as np


def calculate_aabb(vertices):
    # 计算给定顶点的AABB
    min_x = min(vertices, key=lambda x: x[0])[0]
    max_x = max(vertices, key=lambda x: x[0])[0]
    min_y = min(vertices, key=lambda x: x[1])[1]
    max_y = max(vertices, key=lambda x: x[1])[1]
    return (min_x, min_y), (max_x, max_y)


def is_aabb_overlapping(aabb1, aabb2):
    # 判断两个AABB是否重叠
    return not (aabb1[1][0] <= aabb2[0][0] or
                aabb1[0][0] >= aabb2[1][0] or
                aabb1[1][1] <= aabb2[0][1] or
                aabb1[0][1] >= aabb2[1][1])


def get_orientation(p, q, r):
    """计算三个点形成的向量的方向：0表示共线，1表示顺时针，2表示逆时针。"""
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # 共线
    elif val > 0:
        return 1  # 顺时针
    else:
        return 2  # 逆时针

def on_segment(p, q, r):
    """判断点q是否在线段pr上。"""
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
        return True
    return False

def is_edge_intersecting(edge1, edge2):
    """判断两条线段是否相交。"""
    p1, q1 = edge1
    p2, q2 = edge2

    # 获取每对点的相对方向
    o1 = get_orientation(p1, q1, p2)
    o2 = get_orientation(p1, q1, q2)
    o3 = get_orientation(p2, q2, p1)
    o4 = get_orientation(p2, q2, q1)

    # 特殊情况：一条线段的端点在另一条线段上
    if o1 != o2 and o3 != o4:
        return True

    # 特殊情况：一条线段的两个端点分居另一条线段两端点的两侧
    if o1 == 0 and on_segment(p2, p1, q1):
        return True
    if o2 == 0 and on_segment(p1, p2, q1):
        return True
    if o3 == 0 and on_segment(p1, p2, q2):
        return True
    if o4 == 0 and on_segment(p2, p1, q2):
        return True

    return False

def is_point_in_triangle(point, triangle):
    """判断点是否在三角形内。"""
    # 将三角形顶点按照逆时针顺序排列
    triangle = [triangle[0], triangle[1], triangle[2]]

    # 计算点到三角形每条边的有向距离
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(point, triangle[0], triangle[1])
    d2 = sign(point, triangle[1], triangle[2])
    d3 = sign(point, triangle[2], triangle[0])

    # 如果点在三角形的同一侧，则不在三角形内
    if d1 == 0 and d2 == 0 and d3 == 0:
        return on_segment(triangle[0], point, triangle[2])
    elif d1 > 0 and d2 > 0 and d3 > 0:
        return True
    elif d1 < 0 and d2 < 0 and d3 < 0:
        return True
    else:
        return False


def is_collision(square, triangle):
    # 计算正方形和三角形的AABB
    square_aabb = calculate_aabb(square)
    triangle_aabb = calculate_aabb(triangle)

    # 边界框重叠检测
    if is_aabb_overlapping(square_aabb, triangle_aabb):
        # 精确碰撞检测
        # 检查正方形的每个顶点是否在三角形内
        for vertex in square:
            if is_point_in_triangle(vertex, triangle):
                return True
        # 检查正方形的每条边是否与三角形的边相交
        for i in range(4):
            edge1 = (square[i], square[(i + 1) % 4])
            for j in range(3):
                edge2 = (triangle[j], triangle[(j + 1) % 3])
                if is_edge_intersecting(edge1, edge2):
                    return True
    return False

def point_in_triangle(point_to_check, triangle_points):
    """判断点p是否在三角形p1p2p3内（包括边界）"""
    # 计算向量
    v0 = triangle_points[0] - point_to_check
    v1 = triangle_points[1] - point_to_check
    v2 = triangle_points[2] - point_to_check

    # 计算叉积
    a = np.cross(v1-v0, v2-v0)
    b = np.cross(v1-v0, v1)
    c = np.cross(v2-v0, v2)

    # 如果a, b, c同号，则p在三角形内（或边上）
    return (a * b >= 0) and (b * c >= 0) and (c * a >= 0)