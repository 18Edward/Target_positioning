import math
import os
import cv2
import numpy as np
import pandas as pd
from osgeo import gdal, osr
from pyproj import CRS, Transformer, Geod
import time
import json

from sympy import false

# 配置路径与参数
ORTHO_PATH = r"C:\Users\wyk31\Desktop\无人机目标定位算法\kongzhidian5_transparent_mosaic_group1.tif"
OUTPUT_TILES_DIR = r"C:\Users\wyk31\Desktop\无人机目标定位算法\50"
TILES_FEATURES_DIR = r"C:\Users\wyk31\Desktop\无人机目标定位算法\50"
FEATURES_SUBDIR = r"C:\Users\wyk31\Desktop\无人机目标定位算法\50"  # 生成 .npz 的目录
# 分块参数必须与生成 .npz 时一致
TILE_SIZE_M = 50.0
OVERLAP_M = 5.0

# 时间戳转秒数
def parse_timestamp(ts_str):
    parts = ts_str.strip().split('-->')[0].strip().split(':')
    if len(parts) == 2:
        minutes, sec = map(float, parts)
        return int(minutes) * 60 + sec
    elif len(parts) == 3:
        hours, minutes, sec = map(float, parts)
        return int(hours) * 3600 + int(minutes) * 60 + sec
    raise ValueError(f"无法解析: {ts_str}")

# 图像坐标转地理坐标
def img_to_geo(img_points, H):
    # 格式[(x1, y1), (x2, y2), ...]
    # 将H转换为numpy矩阵（方便矩阵运算）
    H_mat = np.array(H, dtype=np.float64)
    geo_points = []

    for (x, y) in img_points:
        # 1. 构建图像坐标的齐次形式 (x, y, 1)
        img_homogeneous = np.array([x, y, 1], dtype=np.float64).reshape(3, 1)

        # 2. 应用单应性矩阵H进行转换
        geo_homogeneous = H_mat @ img_homogeneous  # 矩阵乘法

        # 3. 归一化（除以第三个分量w）
        w = geo_homogeneous[2, 0]
        if abs(w) < 1e-8:  # 避免除以0（理论上不会发生，H是可逆矩阵）
            raise ValueError("单应性矩阵转换异常，w接近0")

        X = geo_homogeneous[0, 0] / w  # 地理X坐标
        Y = geo_homogeneous[1, 0] / w  # 地理Y坐标

        geo_points.append((X, Y))

    return geo_points

def change_Coordinate(x0, y0):
    # 创建转换器：WGS84 (EPSG:4326) -> CGCS2000 / 3-degree Gauss-Kruger CM 117E (EPSG:4548)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4548", always_xy=True)

    x, y = transformer.transform(x0, y0)

    print(f"原始经纬度: {x0:.6f}°E, {y0:.6f}°N")
    print(f"投影坐标: X={x:.2f} m, Y={y:.2f} m")
    return x, y

def reverse_coordinate(x, y):
    # 投影坐标 (CGCS2000 / 3-degree Gauss-Kruger CM 117E, EPSG:4548) -> 地理坐标 (WGS84 经纬度, EPSG:4326)
    # 创建反向转换器：EPSG:4548 → EPSG:4326
    transformer = Transformer.from_crs("EPSG:4548", "EPSG:4326", always_xy=True)

    lon, lat = transformer.transform(x, y)

    print(f"投影坐标: X={x:.2f} m, Y={y:.2f} m")
    print(f"转换为经纬度: {lon:.6f}°E, {lat:.6f}°N")

    return lon, lat

# 获取影像地理信息
def get_geo_info(dataset):
    wkt = dataset.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    transform = dataset.GetGeoTransform()
    pixel_size = abs(transform[1])
    return srs, transform, pixel_size

# 计算图块行列号和2×2块
def xy_to_tile_index(x, y, x0, y0, step_m, ncols, nrows, geo_x, geo_y):
    col = int((x - x0) // step_m)
    row = int((y0 - y) // step_m)
    tile_row = max(0, min(col, ncols - 1))
    tile_col = max(0, min(row, nrows - 1))
    print(f"当前点的行列号：{tile_row}行，{tile_col}列")

    # 计算相对坐标
    relative_x = x - col * step_m - x0
    relative_y = y - row * step_m - y0
    # 判断象限
    right_half = relative_x >= abs(geo_x) / 2
    bottom_half = relative_y >= abs(geo_y) / 2

    # 当前块始终包含
    neighbors = [(tile_row, tile_col)]

    if not bottom_half and right_half:
        candidates = [(-1, 0), (0, 1), (-1, 1)]  # 第一象限
    elif not bottom_half and not right_half:
        candidates = [(-1, 0), (0, -1), (-1, -1)]  # 第二象限
    elif bottom_half and not right_half:
        candidates = [(1, 0), (0, -1), (1, -1)]  # 第三象限
    else:
        candidates = [(1, 0), (0, 1), (1, 1)]  # 第四象限

    # 添加合法邻块
    for dr, dc in candidates:
        r, c = tile_row + dr, tile_col + dc
        if 0 <= r < nrows and 0 <= c < ncols:
            neighbors.append((r, c))

    # 转为 tile ID 并排序
    tile_ids = [r * ncols + c for r, c in neighbors]
    print("tile_ids:",tile_ids)
    return sorted(tile_ids)

# SIFT + RANSAC
def create_sift_detector():
    #创建SIFT检测器参数与生成.npz时一致
    return cv2.SIFT_create(
        nfeatures=1000,
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6
    )
# 图像预处理 + SIFT 检测器
def preprocess_image(img):
    #标准化预处理：转灰度到归一化到均衡化
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 归一化到 0-255 并转为 uint8与生成 .npz 时一致
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gray = cv2.equalizeHist(gray)  # 增强对比度
    return gray
def match_frame_to_tiles(video_frame, tile_ids, tiles_dir):
    sift = create_sift_detector()
    frame_gray = preprocess_image(video_frame)
    kp1, des1 = sift.detectAndCompute(frame_gray, None)

    if des1 is None or len(des1) == 0:
        print(f" 当前视频帧未检测到任何描述！")
        return -1, None, float('inf'), 0

    print(f" 视频帧检测到 {len(kp1)} 个 SIFT 关键点")

    best_tile_id = -1
    best_H = None
    min_error = float('inf')
    max_inliers = 0

    for tile_id in tile_ids:
        npz_path = os.path.join(tiles_dir, FEATURES_SUBDIR, f"tile_{tile_id:04d}.npz")
        if not os.path.exists(npz_path):
            print(f" 图块 {tile_id} 的特征文件不存在: {npz_path}")
            continue

        try:
            data = np.load(npz_path)
            kp_pts = data['keypoints']           # shape=(N, 2)
            des2 = data['descriptors'].astype(np.float32)  # 确保 float32

            if len(kp_pts) == 0 or des2.size == 0:
                print(f" 图块 {tile_id} 特征为空！")
                continue

            print(f" 加载图块 {tile_id}，关键点数: {len(kp_pts)}，描述子形状: {des2.shape}")

            # SIFT 匹配：BFMatcher + Lowe's Ratio Test
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)

            if len(matches) < 2:
                print(f" 图块 {tile_id}: 匹配对太少")
                continue

            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 10:
                print(f" 图块 {tile_id}: 匹配数不足 ({len(good_matches)})")
                continue

            print(f"图块 {tile_id}: 找到 {len(good_matches)} 个高质量匹配")

            # 计算 Homography
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_pts[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                print(f"图块 {tile_id}: findHomography 返回 None")
                continue

            inliers_mask = mask.ravel().astype(bool)
            inliers_count = np.sum(mask)
            print(f"图块 {tile_id}: 内点数 = {inliers_count}")

            if inliers_count >= 4:
                # 使用内点重新拟合
                src_inliers = src_pts[inliers_mask]
                dst_inliers = dst_pts[inliers_mask]
                H_refined, _ = cv2.findHomography(src_inliers, dst_inliers, method=0)  # 最小二乘
                proj_pts = cv2.perspectiveTransform(src_inliers, H_refined)
                reprojection_error = np.mean(np.linalg.norm(dst_inliers - proj_pts, axis=2))

                print(f" 图块 {tile_id}: 重投影误差 = {reprojection_error:.3f}px")

                if inliers_count > max_inliers or (inliers_count == max_inliers and reprojection_error < min_error):
                    best_tile_id = tile_id
                    best_H = H_refined
                    min_error = reprojection_error
                    max_inliers = inliers_count
            else:
                print(f" 图块 {tile_id}: 内点不足 {inliers_count}")

        except Exception as e:
            print(f" 图块 {tile_id} 处理异常: {e}")
            import traceback
            traceback.print_exc()
            continue

    if best_tile_id == -1:
        print("所有候选图块均匹配失败")
    else:
        print(f" 最佳匹配图块: {best_tile_id}, 内点={max_inliers}, 误差={min_error:.3f}px")

    return best_tile_id, best_H, min_error, max_inliers

def main_f(dic, frame, target_li):
    # 前提配置
    metadata_path = os.path.join(TILES_FEATURES_DIR, "metadata.json")
    print(metadata_path)
    with open(metadata_path, 'r', encoding='utf-8') as file:
        meta_data = json.load(file)
    print(meta_data)

    gdal.UseExceptions()
    dataset = gdal.Open(ORTHO_PATH)
    # print("datasets:",dataset.dst_srs.ExportToWkt())
    if not dataset:
        raise FileNotFoundError(f"无法打开正射影像: {ORTHO_PATH}")

    # 数据解构
    width = meta_data['image_size']['width_px']
    height = meta_data['image_size']['height_px']
    pixel_size_m = meta_data['pixel_resolution_m']
    x0, y0 = meta_data['geotransform'][0], meta_data['geotransform'][3]
    geo_x, geo_y = meta_data['geotransform'][1], meta_data['geotransform'][5]

    print(f"x0:{x0}, y0:{y0}")
    step_m = TILE_SIZE_M - OVERLAP_M
    step_px = int(step_m / pixel_size_m)
    ncols = (width + step_px - 1) // step_px
    nrows = (height + step_px - 1) // step_px

    print(f"正射影像: {width}x{height}, 分块: {nrows}×{ncols} ({TILE_SIZE_M}m, 步长{step_m:.1f}m)")

    # 处理开始时间
    start_time = time.time()

    # 将 GPS 坐标转为投影坐标 → 定位所在图块
    # 测试数据
    pitch = 60  # 俯仰角
    h = 150  # 相对高度
    yaw = -90  # 航向角
    lon, lat = 117.000000, 36.000000
    # lon, lat = dic['longitude'], dic['latitude']
    # yaw = dic['yaw']
    # h = dic['h']
    # pitch = dic['pitch']

    # 计算无人机视线与地面交点，即影像中心点
    geod = Geod(ellps="WGS84")
    distance_m = h * math.tan(math.radians(pitch))
    center_lon, center_lat, _ = geod.fwd(
        lons=lon,
        lats=lat,
        az=yaw,  # 航向角
        dist=distance_m  # 距离（米）
    )
    # 坐标运算是否正确
    print(f"center_x:{center_lon}, center_y:{center_lat}")
    center_x, center_y = change_Coordinate(center_lon, center_lat)

    tile_ids = xy_to_tile_index(center_x, center_y, x0, y0, step_m, ncols, nrows, geo_x, geo_y)

    # 执行 SIFT 配准
    best_tile_id, H, avg_error_pixel, inliers_count = match_frame_to_tiles(frame, tile_ids, OUTPUT_TILES_DIR)

    avg_error_meter = avg_error_pixel * pixel_size_m if avg_error_pixel != float('inf') else None
    processing_time = time.time() - start_time

    dic = {
        # 'video_time_sec': round(video_time_sec, 3),
        'latitude': lat,
        'longitude': lon,
        'yaw': yaw,
        # 'center_tile_row': center_row,
        # 'center_tile_col': center_col,
        'candidate_tile_ids': tile_ids,
        'best_tile_id': best_tile_id,
        'homography_matrix': H.tolist() if H is not None else None,
        'avg_reprojection_error_pixel': round(avg_error_pixel, 3) if avg_error_pixel != float('inf') else None,
        'avg_reprojection_error_meter': round(avg_error_meter, 3) if avg_error_meter else None,
        'inliers_count': inliers_count,
        'processing_time_sec': round(processing_time, 3),
    }

    status = f"成功(ID={best_tile_id})" if best_tile_id != -1 else "失败"
    error_str = f"{avg_error_pixel:.2f}px" if avg_error_pixel != float('inf') else "inf"
    print(f"帧: 匹配 {tile_ids}, 最佳: {status}, "
          f"误差={error_str}, 内点={inliers_count}, 耗时={processing_time:.3f}s")

    print(f"转换矩阵：{H}")
    geo_points = img_to_geo(target_li, H)
    print("geo_points:",geo_points)

    print(f"图块左上角投影: X={x0:.2f}, Y={y0:.2f}")

    # 构造GeoTransform
    tile_gt = (x0, pixel_size_m, 0, y0, 0, -pixel_size_m)

    # 转换每个目标点
    for (x_img, y_img) in target_li:
        src_pt = np.array([x_img, y_img, 1])
        dst_homogeneous = H @ src_pt
        w = dst_homogeneous[2]

        if abs(w) < 1e-8:
            raise ValueError("单应性变换 w ≈ 0")

        u = dst_homogeneous[0] / w
        v = dst_homogeneous[1] / w

        # 转换成地理坐标
        X_geo = tile_gt[0] + u * tile_gt[1] + v * tile_gt[2]
        Y_geo = tile_gt[3] + u * tile_gt[4] + v * tile_gt[5]
        X_geo,Y_geo = reverse_coordinate(X_geo, Y_geo)
        geo_points.append((X_geo, Y_geo))
        print(f"({x_img}, {y_img}) -> 图块像素({u:.1f}, {v:.1f}) -> 地理坐标({X_geo:.6f}, {Y_geo:.6f})")

    print("图像坐标 -> 地理坐标：")
    for img, geo in zip(target_li, geo_points):
        print(f"({img[0]}, {img[1]}) -> ({geo[0]:.2f}, {geo[1]:.2f})")

    return geo_points

dic = {
    # "timestamp": "00:00:00,033",
    "latitude": 28.716397,
    "longitude": 115.822704,
    "gb_yaw":  63.1,

}
pic = cv2.imread(r"E:\tcy_data\FullStack\Django\DJI_yolo\Algorithm\img_1.png")
li = [(200, 200)]

result = main_f(dic, pic, li)


