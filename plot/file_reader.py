import numpy as np


def ecef2lla(x, y, z):
    """
    将ECEF坐标转换为大地坐标(纬度, 经度, 高度)

    参数:
    x, y, z: ECEF坐标(m)

    返回:
    lat: 纬度(度)
    lon: 经度(度)
    alt: 高度(m)
    """
    # WGS84椭球参数
    a = 6378137.0  # 长半轴(m)
    f = 1 / 298.257223563  # 扁率
    b = a * (1 - f)  # 短半轴
    e_sq = 1 - (b / a) ** 2  # 第一偏心率平方

    # 经度计算
    lon = np.arctan2(y, x)

    # 迭代计算纬度
    p = np.sqrt(x ** 2 + y ** 2)

    # 初始估计
    lat = np.arctan2(z, p * (1 - e_sq))

    # 迭代计算（通常2-3次迭代足够）
    for _ in range(5):
        N = a / np.sqrt(1 - e_sq * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat_new = np.arctan2(z, p * (1 - e_sq * N / (N + h)))
        if np.abs(lat_new - lat) < 1e-15:
            break
        lat = lat_new

    # 最终高度计算
    N = a / np.sqrt(1 - e_sq * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    # 转换为度
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)

    return lat_deg, lon_deg, alt


def ecef2enu(lat, lon, dx, dy, dz):
    """
    将ECEF坐标差转换为ENU坐标

    参数:
    lat, lon: 参考点的纬度和经度(度)
    dx, dy, dz: ECEF坐标系的差值

    返回:
    de, dn, du: ENU坐标系的东、北、天方向差值
    """
    # 将角度转换为弧度
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # 计算旋转矩阵
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # ECEF到ENU的旋转矩阵
    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])

    # 应用旋转矩阵
    enu = R @ np.array([dx, dy, dz])
    return enu[0], enu[1], enu[2]  # de, dn, du


def read_position_data_enu(file_data, file_ref,
                                     usecols_data=(0, 1, 2, 3), usecols_ref=(1, 2, 3, 4),
                                     skip_header_data=1, skip_header_ref=2):
    """
    Read position data, convert to ENU coordinates using fixed reference point
    Uses the first epoch of reference data as the fixed reference point

    Returns:
    matched_time, de, dn, du  (East, North, Up coordinates relative to fixed reference point)
    """
    # Read data
    data1 = np.genfromtxt(file_data, usecols=usecols_data, skip_header=skip_header_data)
    data2 = np.genfromtxt(file_ref, usecols=usecols_ref, skip_header=skip_header_ref)

    time1 = data1[:, 0]
    pos1 = data1[:, 1:4]  # position data (ECEF)
    time2 = data2[:, 0]
    pos2 = data2[:, 1:4]  # reference position data (ECEF)

    # ref_point
    first_ref_point = pos2[0]
    ref_lat, ref_lon, ref_alt = ecef2lla(first_ref_point[0], first_ref_point[1], first_ref_point[2])

    print(f"Fixed reference point: Lat={ref_lat:.6f}°, Lon={ref_lon:.6f}°, Alt={ref_alt:.2f}m")

    return _match_data_enu(time1, pos1, time2, pos2, ref_lat, ref_lon, first_ref_point)


def _match_data_enu(time1, values1, time2, values2, ref_lat, ref_lon, fixed_ref_point):
    ref_dict = {int(t): vals for t, vals in zip(time2, values2)}
    matched_time, de_list, dn_list, du_list = [], [], [], []

    for t, vals in zip(time1, values1):
        t_int = int(t)
        if t_int in ref_dict:
            ref_vals = ref_dict[t_int]

            #diff
            dx = vals[0] - ref_vals[0]
            dy = vals[1] - ref_vals[1]
            dz = vals[2] - ref_vals[2]

            # trans dxyz to denu
            de, dn, du = ecef2enu(ref_lat, ref_lon, dx, dy, dz)

            matched_time.append(t_int)
            de_list.append(de)
            dn_list.append(dn)
            du_list.append(du)

    return np.array(matched_time), np.array(de_list), np.array(dn_list), np.array(du_list)

## calculate xyz difference
def read_position_data(file_data, file_ref, usecols_data=(0, 1, 2, 3),
                       usecols_ref=(1, 2, 3, 4), skip_header_data=1, skip_header_ref=2):
    """
    Read position data and perform time matching (XYZ coordinates)

    Returns:
    matched_time, dx, dy, dz
    """
    # Read data
    data1 = np.genfromtxt(file_data, usecols=usecols_data, skip_header=skip_header_data)
    data2 = np.genfromtxt(file_ref, usecols=usecols_ref, skip_header=skip_header_ref)

    time1 = data1[:, 0]
    pos1 = data1[:, 1:4]  # position data
    time2 = data2[:, 0]
    pos2 = data2[:, 1:4]  # reference position data

    return _match_data(time1, pos1, time2, pos2)


def read_velocity_data(file_data, file_ref, usecols_data=(0, 4, 5, 6),
                       usecols_ref=(1, 15, 16, 17), skip_header_data=1, skip_header_ref=2):
    """
    Read velocity data and perform time matching

    Returns:
    matched_time, dvx, dvy, dvz
    """
    # Read data
    data1 = np.genfromtxt(file_data, usecols=usecols_data, skip_header=skip_header_data)
    data2 = np.genfromtxt(file_ref, usecols=usecols_ref, skip_header=skip_header_ref)

    time1 = data1[:, 0]
    vel1 = data1[:, 1:4]  # velocity data
    time2 = data2[:, 0]
    vel2 = data2[:, 1:4]  # reference velocity data

    return _match_data(time1, vel1, time2, vel2)


def read_attitude_data(file_data, file_ref, usecols_data=(0, 9, 7, 8),
                       usecols_ref=(1, 24, 25, 26), skip_header_data=1, skip_header_ref=2):
    """
    Read attitude data with special angle processing

    Returns:
    matched_time, dpitch, droll, dyaw
    """
    # Read data
    data1 = np.genfromtxt(file_data, usecols=usecols_data, skip_header=skip_header_data)
    data2 = np.genfromtxt(file_ref, usecols=usecols_ref, skip_header=skip_header_ref)

    time1 = data1[:, 0]
    att1 = data1[:, 1:4]  # attitude data (pitch, roll, yaw)
    time2 = data2[:, 0]
    att2 = data2[:, 1:4]  # reference attitude data

    # Create reference dictionary and match data
    ref_dict = {int(t): att for t, att in zip(time2, att2)}
    matched_time, dpitch_list, droll_list, dyaw_list = [], [], [], []

    for t, att in zip(time1, att1):
        t_int = int(t)
        if t_int in ref_dict:
            ref_att = ref_dict[t_int]

            # Special attitude angle processing
            if abs(ref_att[0]) < 180:
                ref_att[0] = -ref_att[0]
                diff = att - ref_att
                if ((abs(att[0]) > 170.0) & (abs(ref_att[0]) > 170.0) &
                        (att[0] * ref_att[0] < 0) & (att[0] + ref_att[0] < 5)):
                    diff[0] = att[0] + ref_att[0]
            else:
                diff = att - ref_att

            matched_time.append(t_int)
            dpitch_list.append(diff[0])
            droll_list.append(diff[1])
            dyaw_list.append(diff[2])

    return (np.array(matched_time),
            np.array(dpitch_list),
            np.array(droll_list),
            np.array(dyaw_list))


def _match_data(time1, values1, time2, values2):
    """
    Internal function: general data matching
    """
    ref_dict = {int(t): vals for t, vals in zip(time2, values2)}
    matched_time, errors_x, errors_y, errors_z = [], [], [], []

    for t, vals in zip(time1, values1):
        t_int = int(t)
        if t_int in ref_dict:
            ref_vals = ref_dict[t_int]
            diff = vals - ref_vals
            matched_time.append(t_int)
            errors_x.append(diff[0])
            errors_y.append(diff[1])
            errors_z.append(diff[2])

    return (np.array(matched_time),
            np.array(errors_x),
            np.array(errors_y),
            np.array(errors_z))