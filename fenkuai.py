import os
# 电脑自下载gdal版本与conda版本不一致，proj.db冲突，显示指定路径
os.environ['PROJ_LIB'] = r"D:\Users\wyk31\anaconda3\envs\DJI_yolo\Library\share\proj"
import cv2
import numpy as np
from osgeo import gdal, osr
import json


gdal.SetConfigOption('GTIFF_SRS_SOURCE', 'EPSG')  # 强制使用 EPSG 官方定义
gdal.UseExceptions()
# 主函数入口
def main():
    # 配置路径
    ORTHO_PATH = r"C:\Users\wyk31\Desktop\无人机目标定位算法\kongzhidian5_transparent_mosaic_group1.tif"  # 正射影像路径
    OUTPUT_DIR = r"C:\Users\wyk31\Desktop\无人机目标定位算法\50"  # 分块存储文件夹

    # 分块参数单位：米（若无默认60m/6m）
    TILE_SIZE_M = 60.0   # 每块 60m x 60m
    OVERLAP_M  = 6.0     # 重叠 6m

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"正在打开影像: {ORTHO_PATH}")
    gdal.UseExceptions()  # 确保gdal抛出异常信息，默认直接返回None
    dataset = gdal.Open(ORTHO_PATH)
    if dataset is None:
        raise FileNotFoundError(f"无法打开影像文件: {ORTHO_PATH}")  # 手动抛出异常

    width, height = dataset.RasterXSize, dataset.RasterYSize  # 像素数量
    geotransform = dataset.GetGeoTransform()  # 获取仿射变换参数（左上角坐标、x、y方向像素宽度）
    print(geotransform)
    projection = dataset.GetProjection()  # 获取空间参考
    print(projection)

    # 获取基本影像信息
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    # 计算像素分辨率
    pixel_size_x = abs(geotransform[1])
    pixel_size_y = abs(geotransform[5])
    pixel_size_m = round((pixel_size_x + pixel_size_y) / 2, 8)  # 平均值，保留高精度
    # 左上角坐标
    x0, y0 = geotransform[0], geotransform[3]

    # 初始化元数据字典
    srs_info = {
        "image_size": {
            "width_px": width,
            "height_px": height
        },
        "pixel_resolution_m": pixel_size_m,
        "geotransform": list(geotransform),  # 转为 list 以便 JSON 序列化
        "origin": {
            "x0_easting": round(x0, 6),
            "y0_northing": round(y0, 6)
        }
    }
    # 解析空间参考系统 (SRS)
    if projection:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection)

        # 尝试获取 EPSG 代码
        auth_code = srs.GetAuthorityCode(None)
        if auth_code and auth_code != "0":
            srs_info["epsg_code"] = f"EPSG:{auth_code}"
        else:
            srs_info["epsg_code"] = "无法识别"
    else:
        srs_info["epsg_code"] = "无"

    # 保存为 JSON 文件
    json_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(srs_info, f, indent=4, ensure_ascii=False)
    print(f"\n完整元数据已保存至: {json_path}")


    pixel_size = abs(geotransform[1])  # 像素宽度

    # 封装基本信息
    info = {
        'dataset': dataset,
        'width': width,
        'height': height,
        'geotransform': geotransform,
        'projection': projection,
        'pixel_x_size': pixel_size
    }

    # 计算分块参数
    tile_params = calculate_tile_params(info, tile_size_m=TILE_SIZE_M, overlap_m=OVERLAP_M)
    windows = generate_tile_windows(info, tile_params)

    # 初始化 SIFT 检测器
    sift = cv2.SIFT_create(
        nfeatures=1000,           # 每块最多 1000 个关键点
        nOctaveLayers=3,          # 金字塔层数
        contrastThreshold=0.04,   # 过滤低对比度点
        edgeThreshold=10,         # 过滤边缘响应
        sigma=1.6                 # 高斯核标准差
    )

    # 遍历每个图块：读取 → 提取SIFT → 保存 .tif + .npz + 索引
    for window in windows:
        process_and_save_tile(dataset, window, OUTPUT_DIR, sift)

    del dataset
    print(f"所有图块处理完成！共 {len(windows)} 个，保存至: {OUTPUT_DIR}")

# 计算分块参数即像素尺寸
def calculate_tile_params(info, tile_size_m=60.0, overlap_m=6.0):
    pixel_size = info['pixel_x_size']
    block_size_px = int(tile_size_m / pixel_size)  # 每块的方向像素个数
    overlap_px = int(overlap_m / pixel_size)  # 每块的方向重叠像素个数
    stride_px = block_size_px - overlap_px

    params = {
        'block_size_px': block_size_px,
        'overlap_px': overlap_px,
        'stride_px': stride_px
    }

    print(f" 分块参数计算完成：")
    print(f"  块大小: {block_size_px}px × {block_size_px}px ({tile_size_m}m×{tile_size_m}m)")
    print(f"  步长: {stride_px}px，重叠: {overlap_px}px ({overlap_m}m)")

    return params

# 生成所有图块窗口含地理信息
def generate_tile_windows(img_info, tile_params):
    width = img_info['width']
    height = img_info['height']
    block_size = tile_params['block_size_px']
    stride = tile_params['stride_px']
    gt = img_info['geotransform']

    windows = []
    block_id = 0

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            win_w = min(block_size, width - x)
            win_h = min(block_size, height - y)

            # 获取每块的边界坐标
            geo_x = gt[0] + x * gt[1]
            geo_y = gt[3] + y * gt[5]
            geo_x2 = geo_x + win_w * gt[1]
            geo_y2 = geo_y + win_h * gt[5]

            windows.append({
                'block_id': block_id,
                'x': x, 'y': y,
                'width': win_w, 'height': win_h,
                'geo_bbox': (geo_x, geo_y2, geo_x2, geo_y),  # left, bottom, right, top
                'new_gt': (geo_x, gt[1], 0, geo_y, 0, gt[5])
            })
            block_id += 1

    print(f" 共生成 {len(windows)} 个图块")
    return windows

# 处理并保存图块：.tif + .npz + 索引
def process_and_save_tile(dataset, window, output_path, sift):
    x, y, w, h = window["x"], window["y"], window["width"], window["height"]
    new_gt = window["new_gt"]

    # 读取图像数据[lt_x,lt_y,w,h(像素个数)]，若读取单波段栅格（如 DEM、灰度图）时，返回的是 2D 数组，需后续expand_dims处理统一数据格式
    data = dataset.ReadAsArray(x, y, w, h)
    if data is None:
        print(f" 读取失败: block_{window['block_id']}")
        return None

    # 统一维度 (bands, h, w)
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
    bands, h, w = data.shape

    # 取第一波段作为灰度图用于 SIFT
    gray_tile = data[0].astype(np.float32)
    gray_tile = cv2.normalize(gray_tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    #  提取 SIFT 特征
    kp, des = sift.detectAndCompute(gray_tile, None)  # des(N, 128)

    # 安全处理：防止 kp 或 des 为 None
    if kp is not None and des is not None:
        pts = np.array([k.pt for k in kp], dtype=np.float32)   # 转换为(N, 2)的数组
        des = des.astype(np.float32)  # 确保数据类型
        num_kp = len(kp)
    else:
        pts = np.zeros((0, 2), dtype=np.float32)
        des = np.zeros((0, 128), dtype=np.float32)
        num_kp = 0

    # 保存特征到 .npz 文件
    npz_filename = f"tile_{window['block_id']:04d}.npz"
    npz_path = os.path.join(output_path, npz_filename)
    np.savez_compressed(
        npz_path,
        keypoints=pts,
        descriptors=des,
        geotransform=new_gt,
        width=w,
        height=h,
        block_id=window['block_id']
    )
    print(f" 特征已保存: {npz_path} -> {num_kp} 个关键点")

    # 保存 GeoTIFF 图像
    tif_filename = f"tile_{window['block_id']:04d}.tif"
    tif_path = os.path.join(output_path, tif_filename)

    if data.dtype == np.float32:
        gdal_dtype = gdal.GDT_Float32
    elif data.dtype == np.float64:
        gdal_dtype = gdal.GDT_Float64
    else:
        gdal_dtype = gdal.GDT_Byte
        data = np.clip(data, 0, 255).astype(np.uint8)

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(tif_path, w, h, bands, gdal_dtype)  # 创建文件，还未写入像素
    out_ds.SetGeoTransform(new_gt)  # 添加6个仿射参数
    out_ds.SetProjection(dataset.GetProjection())  # 坐标系定义
    for i in range(bands):
        out_ds.GetRasterBand(i + 1).WriteArray(data[i])   # 写入像素值到每个波段，GDAL波段从1开始，所以用i+1
    out_ds.FlushCache()  # 强制将缓存中的数据刷入磁盘
    del out_ds  # 显式关闭文件句柄，否则导致文件被锁住，无法被其他程序访问

    # 更新索引文件
    index_file = os.path.join(output_path, "tiles_index.json")
    tiles_list = []
    if os.path.exists(index_file):
        with open(index_file, 'r', encoding='utf-8') as f:
            # 如果有数据，执行追加逻辑，而不是直接覆盖，支持代码二次运行
            try:
                tiles_list = json.load(f)
                if isinstance(tiles_list, dict):
                    tiles_list = list(tiles_list.values())
            except json.JSONDecodeError:
                pass

    tile_info = {
        'block_id': window['block_id'],
        'image_path': tif_path,
        'features_path': npz_path,
        'bbox': window['geo_bbox'],  # 左上角、右下角坐标
        'width_px': w,
        'height_px': h,
        'sift_keypoints': num_kp
    }
    tiles_list.append(tile_info)

    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(tiles_list, f, indent=2, ensure_ascii=False)

    print(f"已保存: {tif_filename} 和 {npz_filename}")

# 运行
if __name__ == '__main__':
    main()