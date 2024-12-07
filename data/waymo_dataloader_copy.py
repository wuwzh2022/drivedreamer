import os
import glob
import torch
import tensorflow as tf
from torch.utils import data
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset  # Waymo 数据集协议缓冲区
from omegaconf import OmegaConf
import numpy as np
import cv2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset.protos import map_pb2
import plotly.express as px
import plotly.graph_objs as go
from typing import List
import dataclasses
import enum
import pandas as pd
from waymo_open_dataset.utils import plot_maps
class FeatureType(enum.Enum):
    """Definintions for map feature types."""
    UNKNOWN_FEATURE = 0
    FREEWAY_LANE = 1
    SURFACE_STREET_LANE = 2
    BIKE_LANE = 3
    BROKEN_SINGLE_WHITE_BOUNDARY = 6
    SOLID_SINGLE_WHITE_BOUNDARY = 7
    SOLID_DOUBLE_WHITE_BOUNDARY = 8
    BROKEN_SINGLE_YELLOW_BOUNDARY = 9
    BROKEN_DOUBLE_YELLOW_BOUNDARY = 10
    SOLID_SINGLE_YELLOW_BOUNDARY = 11
    SOLID_DOUBLE_YELLOW_BOUNDARY = 12
    PASSING_DOUBLE_YELLOW_BOUNDARY = 13
    ROAD_EDGE_BOUNDARY = 15
    ROAD_EDGE_MEDIAN = 16
    STOP_SIGN = 17
    CROSSWALK = 18
    SPEED_BUMP = 19
    DRIVEWAY = 20
    
@dataclasses.dataclass(frozen=True)
class MapPoints:
  """A container for map point data."""

  x: list[float] = dataclasses.field(default_factory=list)
  y: list[float] = dataclasses.field(default_factory=list)
  z: list[float] = dataclasses.field(default_factory=list)
  types: list[FeatureType] = dataclasses.field(default_factory=list)
  ids: list[int] = dataclasses.field(default_factory=list)

  def append_point(
      self, point: map_pb2.MapPoint, feature_type: FeatureType, feature_id: int
  ):
    """Append a given point to the container."""
    self.x.append(point.x)
    self.y.append(point.y)
    self.z.append(point.z)
    self.types.append(feature_type)
    self.ids.append(feature_id)
    
    
class WaymoDataloader(data.Dataset):
    def __init__(self, cfg, num_boxes, movie_len, split_name='train', return_pose_info=False, collect_condition=None):
        self.split_name = split_name
        self.cfg = cfg
        self.movie_len = movie_len
        self.num_boxes = num_boxes
        self.return_pose_info = return_pose_info
        self.collect_condition = collect_condition
        self.load_data_infos()

    def load_data_infos(self):
        # 获取所有 tfrecord 文件
        data_path = os.path.join(self.cfg['dataroot'], "*.tfrecord")
        self.data_files = glob.glob(data_path)
        self.video_infos = []
        self.action_infos = []
        
        print(f"Found {len(self.data_files)} tfrecord files.")
        
        for file_idx, tfrecord_file in enumerate(tqdm(self.data_files)):
            dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
            for record in dataset:
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(record.numpy()))
                self.video_infos.append(frame)  # 存储每帧信息
                # 可以在此处提取相关的动作信息，放入 self.action_infos
                self.action_infos.append(self.extract_actions_from_frame(frame))
                break

    def extract_actions_from_frame(self, frame):
        # 提取 Waymo 帧中的动作信息，例如车的速度、方向等
        # print("Frame attributes:", dir(frame))
        action_info = torch.tensor([frame.pose.transform])
        return action_info

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        return self.get_data_info(idx)

    def get_data_info(self, idx):
        frame = self.video_infos[idx]
        actions = self.action_infos[idx]
        out = {}

        # 初始化需要提取的数据结构
        for key in self.collect_condition:
            if key == '3Dbox':
                out[key] = torch.zeros((self.movie_len, self.num_boxes, 16))
                out['category'] = [None for _ in range(self.movie_len)]
            elif key == 'text':
                out[key] = [None for _ in range(self.movie_len)]
            # elif key == 'actions':
            #     out[key] = torch.zeros((self.movie_len, 14))
            else:
                out[key] = torch.zeros((self.movie_len, self.cfg['img_size'][0], self.cfg['img_size'][1], 3))

        # 遍历视频帧，提取每帧中的数据
        for i in range(self.movie_len):
            # 例如提取相机图像
            if 'reference_image' in self.collect_condition:
                camera_image = self.extract_camera_image(frame)
                out['reference_image'][i] = camera_image

            # 提取 3D 盒子（检测框）
            if '3Dbox' in self.collect_condition:
                boxes, categories = self.extract_boxes_from_front_view(frame)
                out['3Dbox'][i] = torch.tensor(boxes)
                out['category'][i] = categories

            if 'range_image' in self.collect_condition:
                range_image = self.extract_range_image(frame)
                out['range_image'][i] = torch.tensor(range_image)

            if 'HD_map' in self.collect_condition:
                HD_map = self.extract_HD_map(frame)
                out['HD_map'][i] = torch.tensor(HD_map)
                
            # if 'actions' in self.collect_condition:
            #     out['actions'][i] = actions

        return out
    
        
    def plot_map_points(self, map_points: MapPoints) -> go._figure.Figure:
        """Creates an interactive visualization of map data.

        Args:
            map_points: The set of map points to plot.

        Returns:
            A plotly figure object.
        """

        line_dict = {
            FeatureType.UNKNOWN_FEATURE: ('black', 'solid'),
            FeatureType.FREEWAY_LANE: ('white', 'solid'),
            FeatureType.SURFACE_STREET_LANE: ('royalblue', 'solid'),
            FeatureType.BIKE_LANE: ('magenta', 'solid'),
            FeatureType.BROKEN_SINGLE_WHITE_BOUNDARY: ('lightgray', 'dash'),
            FeatureType.SOLID_SINGLE_WHITE_BOUNDARY: ('lightgray', 'solid'),
            FeatureType.SOLID_DOUBLE_WHITE_BOUNDARY: ('lightgray', 'solid'),
            FeatureType.BROKEN_SINGLE_YELLOW_BOUNDARY: ('yellow', 'dash'),
            FeatureType.BROKEN_DOUBLE_YELLOW_BOUNDARY: ('yellow', 'dash'),
            FeatureType.SOLID_SINGLE_YELLOW_BOUNDARY: ('yellow', 'solid'),
            FeatureType.SOLID_DOUBLE_YELLOW_BOUNDARY: ('yellow', 'solid'),
            FeatureType.PASSING_DOUBLE_YELLOW_BOUNDARY: ('yellow', 'dash'),
            FeatureType.ROAD_EDGE_BOUNDARY: ('green', 'solid'),
            FeatureType.ROAD_EDGE_MEDIAN: ('green', 'solid'),
            FeatureType.STOP_SIGN: ('red', 'solid'),
            FeatureType.CROSSWALK: ('orange', 'solid'),
            FeatureType.SPEED_BUMP: ('cyan', 'solid'),
            FeatureType.DRIVEWAY: ('blue', 'solid'),
        }

        # Create a scatter plot of all points in the roadgraph data.
        feature_types_str = list(map(str, map_points.types))
        data1 = {
            'x': map_points.x,
            'y': map_points.y,
            'z': map_points.z,
            'feature_type': feature_types_str,
            'id': map_points.ids,
        }
        df = pd.DataFrame(data1)

        color_dict = {}
        for k in line_dict:
            color_dict[str(k)] = line_dict[k][0]

        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color='feature_type',
            color_discrete_map=color_dict,
        )
        fig.update_traces(marker_size=1.25)

        # Plot connecting lines for each individual roadgraph feature.
        start_index = 0
        end_index = 0
        point_type = map_points.types[0]
        feature_id = map_points.ids[0]

        color_dict = {}
        for k in line_dict:
            color_dict[k] = line_dict[k][0]

        num_points = len(map_points.x)
        while start_index < num_points:
            while end_index < num_points and map_points.ids[end_index] == feature_id:
                end_index += 1

            xvals = map_points.x[start_index:end_index]
            yvals = map_points.y[start_index:end_index]
            zvals = map_points.z[start_index:end_index]

            # Plot stop signs as larger points in 3D.
            width = 1.5
            if point_type == 1 or point_type == 2 or point_type == 3:
                width = 2.5
            if point_type == 17:
                fig.add_trace(
                    go.Scatter3d(
                        x=xvals,
                        y=yvals,
                        z=zvals,
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=color_dict[point_type],
                        ),
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=xvals,
                        y=yvals,
                        z=zvals,
                        mode='lines',
                        line=dict(
                            dash=line_dict[point_type][1],
                            color=color_dict[point_type],
                            width=width,
                        ),
                    )
                )

            start_index = end_index
            if start_index < num_points:
                point_type = map_points.types[start_index]
                feature_id = map_points.ids[start_index]

        # Format the plot.
        axis_config = dict(
            backgroundcolor='rgb(0, 0, 0)',
            gridcolor='gray',
            showgrid=False,
            showline=False,
            showticklabels=False,
            showbackground=True,
            zerolinecolor='gray',
            tickfont=dict(color='gray'),
        )
        fig.update_layout(
            showlegend=False,
            scene=dict(xaxis=axis_config, yaxis=axis_config, zaxis=axis_config),
            width=1600,
            height=1200,
            paper_bgcolor='black',
            plot_bgcolor='rgba(0,0,0,0)',
            scene_aspectmode='data',
        )
        fig.update_yaxes(
            scaleanchor='x',
            scaleratio=1,
        )

        # Set the initial camera position. This sets the camera to be directly above
        # the center of the scene looking down from a height to view the majority
        # of the scene.
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=3),
        )
        fig.update_layout(scene_camera=camera)
        fig.write_image("filtered_hd_map_visualization.png")    
        return fig


    def plot_map_features(self,
            map_features: List[map_pb2.MapFeature],
        ) -> go._figure.Figure:
        """Plots the map data for a Scenario proto from the open dataset.

        Args:
            map_features: A list of map features to be plotted.

        Returns:
            A plotly figure object.
        """
        lane_types = {
            map_pb2.LaneCenter.TYPE_UNDEFINED: FeatureType.UNKNOWN_FEATURE,
            map_pb2.LaneCenter.TYPE_FREEWAY: FeatureType.FREEWAY_LANE,
            map_pb2.LaneCenter.TYPE_SURFACE_STREET: FeatureType.SURFACE_STREET_LANE,
            map_pb2.LaneCenter.TYPE_BIKE_LANE: FeatureType.BIKE_LANE,
        }
        road_line_types = {
            map_pb2.RoadLine.TYPE_UNKNOWN: (
                FeatureType.UNKNOWN_FEATURE
            ),
            map_pb2.RoadLine.TYPE_BROKEN_SINGLE_WHITE: (
                FeatureType.BROKEN_SINGLE_WHITE_BOUNDARY
            ),
            map_pb2.RoadLine.TYPE_SOLID_SINGLE_WHITE: (
                FeatureType.SOLID_SINGLE_WHITE_BOUNDARY
            ),
            map_pb2.RoadLine.TYPE_SOLID_DOUBLE_WHITE: (
                FeatureType.SOLID_DOUBLE_WHITE_BOUNDARY
            ),
            map_pb2.RoadLine.TYPE_BROKEN_SINGLE_YELLOW: (
                FeatureType.BROKEN_SINGLE_YELLOW_BOUNDARY
            ),
            map_pb2.RoadLine.TYPE_BROKEN_DOUBLE_YELLOW: (
                FeatureType.BROKEN_DOUBLE_YELLOW_BOUNDARY
            ),
            map_pb2.RoadLine.TYPE_SOLID_SINGLE_YELLOW: (
                FeatureType.SOLID_SINGLE_YELLOW_BOUNDARY
            ),
            map_pb2.RoadLine.TYPE_PASSING_DOUBLE_YELLOW: (
                FeatureType.PASSING_DOUBLE_YELLOW_BOUNDARY
            ),
        }
        road_edge_types = {
            map_pb2.RoadEdge.TYPE_UNKNOWN: FeatureType.UNKNOWN_FEATURE,
            map_pb2.RoadEdge.TYPE_ROAD_EDGE_BOUNDARY: FeatureType.ROAD_EDGE_BOUNDARY,
            map_pb2.RoadEdge.TYPE_ROAD_EDGE_MEDIAN: FeatureType.ROAD_EDGE_MEDIAN,
        }

        def add_points(
            feature_id: int,
            points: List[map_pb2.MapPoint],
            feature_type: FeatureType,
            map_points: MapPoints,
            is_polygon=False,
        ):
            if feature_type is None:
                return
            for point in points:
                map_points.append_point(point, feature_type, feature_id)

            if is_polygon:
                map_points.append_point(points[0], feature_type, feature_id)

        # Create arrays of the map points to be plotted.
        map_points = MapPoints()

        for feature in map_features:
            if feature.HasField('lane'):
                add_points(
                    feature.id,
                    list(feature.lane.polyline),
                    lane_types.get(feature.lane.type),
                    map_points,
                )
            elif feature.HasField('road_line'):
                feature_type = road_line_types.get(feature.road_line.type)
                add_points(
                    feature.id, list(feature.road_line.polyline), feature_type, map_points
                )
            elif feature.HasField('road_edge'):
                feature_type = road_edge_types.get(feature.road_edge.type)
                add_points(
                    feature.id, list(feature.road_edge.polyline), feature_type, map_points
                )
            elif feature.HasField('stop_sign'):
                add_points(
                    feature.id,
                    [feature.stop_sign.position],
                    FeatureType.STOP_SIGN,
                    map_points,
                )
            elif feature.HasField('crosswalk'):
                add_points(
                    feature.id,
                    list(feature.crosswalk.polygon),
                    FeatureType.CROSSWALK,
                    map_points,
                    True,
                )
            elif feature.HasField('speed_bump'):
                add_points(
                    feature.id,
                    list(feature.speed_bump.polygon),
                    FeatureType.SPEED_BUMP,
                    map_points,
                    True,
                )
            elif feature.HasField('driveway'):
                add_points(
                    feature.id,
                    list(feature.driveway.polygon),
                    FeatureType.DRIVEWAY,
                    map_points,
                    True,
                )

        # Return the interactive 3D map plot.
        return self.plot_map_points(map_points)

    def extract_HD_map(self,frame):
        print('Lllllllllllllllll')
        print(frame.map_features)
        return self.plot_map_features(frame.map_features)

    
    def generate_range_image(self, range_image):
        H = self.cfg.img_size[0]
        W = self.cfg.img_size[1]
        depth_map = np.full((H, W), np.inf)
        
        # 遍历每个点，根据 x, y 坐标填入深度值
        for point in range_image:
            x, y, depth = point[0], point[1], point[2]
            # 将 x 和 y 转换到图像坐标范围
            x = int(x * W / self.cfg.waymo_size[1])
            y = int(y * H / self.cfg.waymo_size[0])
            
            # 确保坐标在图像范围内
            if 0 <= x < W and 0 <= y < H:
                depth_map[y, x] = min(depth_map[y,x], depth)
            else:
                print(f"Point ({x}, {y}) is out of range.")
        # 将深度图扩展成 (H, W, 1) 的形状
        depth_image = np.expand_dims(depth_map, axis=-1)

        # 复制深度图的最后一个维度，使输出形状为 (H, W, 3)
        depth_image = np.repeat(depth_image, 3, axis=-1)
        
        return depth_image



    def extract_range_image(self, frame):
        (range_images, camera_projections,
        _, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
            frame)
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)
        points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1)

        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)
        points_all_ri2 = np.concatenate(points_ri2, axis=0)
        # camera projection corresponding to each point.
        cp_points_all = np.concatenate(cp_points, axis=0)
        cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)


        images = sorted(frame.images, key=lambda i: i.name)
        cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
        cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

        # 提取 points_all 的 z 值而不是欧几里得距离
        z_values_tensor = points_all[..., 2:3]  # 只提取 z 坐标

        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)

        cp_points_all_tensor = tf.cast(tf.gather_nd(
            cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
        z_values_tensor = tf.gather_nd(z_values_tensor, tf.where(mask))

        # 拼接时使用原始的 z 坐标而不是距离
        projected_points_all_from_raw_data = tf.concat(
            [cp_points_all_tensor[..., 1:3], z_values_tensor], axis=-1).numpy()
        
        return self.generate_range_image(projected_points_all_from_raw_data)

    def extract_camera_image(self, frame):
        # 提取相机图像
        for image in frame.images:
            if image.name == open_dataset.CameraName.FRONT:
                decoded_img = tf.image.decode_jpeg(image.image).numpy()
                resized_img = cv2.resize(decoded_img, (self.cfg['img_size'][1], self.cfg['img_size'][0]))  # 注意顺序是 (width, height)
                cv2.imwrite('output_image.jpg', resized_img)
                img_tensor = torch.from_numpy(resized_img).float() / 255.0
                return img_tensor
        return  torch.from_numpy(np.zeros((self.cfg['img_size'][0], self.cfg['img_size'][1], 3)))  # 若无图像则返回空白

    def extract_boxes_from_front_view(self, frame):
        # 只提取前视相机标签
        boxes = []
        categories = []
        
        # 获取前视相机（FRONT）的ID
        front_camera_id = open_dataset.CameraName.FRONT

        # 获取相机内参
        camera_intrinsic = None
        for camera in frame.context.camera_calibrations:
            if camera.name == front_camera_id:
                camera_intrinsic = np.array(camera.intrinsic).reshape(3, 3)  # 3x3矩阵
                break

        if camera_intrinsic is None:
            raise ValueError("前视相机的内参未找到")

        # 遍历前视相机中的每个标签（即物体的检测框）
        for camera_labels in frame.camera_labels:
            if camera_labels.name == front_camera_id:  # 只处理前视相机
                for label in camera_labels.labels:
                    # 提取 3D 物体的八个顶点的 3D 坐标
                    box = label.box
                    box_vertices_3d = self.get_3d_box_vertices(box)

                    # 将 3D 坐标投影到 2D 图像平面
                    box_vertices_2d = self.project_3d_to_2d(box_vertices_3d, camera_intrinsic)

                    # 保存 2D 投影的顶点（16个值，每个顶点是xy坐标）
                    boxes.append(box_vertices_2d.flatten())
                    categories.append(label.type)  # 提取类别

        # 将边界框信息转换为 numpy 数组
        boxes = np.array(boxes).astype(np.float32)
    
        # 如果 boxes 为空，直接创建一个形状为 (70, 16) 的全零数组
        if len(boxes) == 0:
            boxes = np.zeros((70, 16), dtype=np.float32)
            categories = ["None"] * 70
        else:
            # 如果 box 数量超过 70，只取前 70 个
            if len(boxes) > 70:
                boxes = boxes[:70]
                categories = categories[:70]
            else:
                # 如果 box 数量不足 70，补 0
                padding = 70 - len(boxes)
                boxes = np.pad(boxes, ((0, padding), (0, 0)), mode='constant', constant_values=0)
                categories += ["None"] * padding

        return boxes, categories


    def get_3d_box_vertices(self, box):
        """
        根据 3D 物体框的信息计算 8 个顶点的 3D 坐标
        """
        cx, cy, cz = box.center_x, box.center_y, box.center_z
        l, w, h = box.length, box.width, box.height
        heading = box.heading

        # 创建 3D 框的局部坐标（物体中心点为原点）
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]

        # 将局部坐标旋转到全局坐标系
        rotation_matrix = np.array([
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading), np.cos(heading), 0],
            [0, 0, 1]
        ])

        # 组合顶点
        corners_3d = np.dot(rotation_matrix, np.vstack([x_corners, y_corners, z_corners]))

        # 将顶点平移到物体的世界坐标
        corners_3d[0, :] += cx
        corners_3d[1, :] += cy
        corners_3d[2, :] += cz

        return corners_3d


    def project_3d_to_2d(self, vertices_3d, camera_intrinsic):
        """
        将 3D 顶点坐标通过相机内参矩阵投影到 2D 图像平面
        """
        # 转换为齐次坐标
        ones = np.ones((1, vertices_3d.shape[1]))
        vertices_3d_hom = np.vstack([vertices_3d, ones])

        # 投影到 2D 平面
        vertices_2d_hom = np.dot(camera_intrinsic, vertices_3d_hom[:3, :])

        # 齐次坐标归一化
        vertices_2d = vertices_2d_hom[:2, :] / vertices_2d_hom[2, :]

        return vertices_2d.T

def collate_fn(batch):
    out = {}
    for i in range(len(batch)):
        for key,value in batch[i].items():
            if isinstance(value,torch.Tensor):
                if not key in out.keys():
                    out[key] = value.unsqueeze(0)
                else:
                    out[key] = torch.concat([out[key],value.unsqueeze(0)],dim=0)
            elif isinstance(value,list):
                if not key in out.keys():
                    out[key] = []
                    out[key].append(value)
                else:
                    out[key].append(value)
            elif isinstance(value,dict):
                out[key] = {}
                for k in value.keys():
                    if isinstance(value[k],list):
                        if not k in out[key].keys():
                            out[key][k] = []
                            out[key][k].append(value)
                        else:
                            out[key][k].append(value)
                    else:
                        raise NotImplementedError
            else:
                raise NotImplementedError
    return out

def main():
    # 加载配置文件
    cfg = OmegaConf.load("../configs/waymo.yaml")
    
    # 实例化数据加载器
    data_loader = WaymoDataloader(cfg, 
                                  num_boxes=cfg.data.params.train.num_boxes, 
                                  movie_len=cfg.data.params.train.movie_len, 
                                  split_name=cfg.data.params.train.split_name, 
                                  return_pose_info=cfg.data.params.train.return_pose_info, 
                                  collect_condition=cfg.data.params.train.collect_condition)
    
    # 使用 PyTorch DataLoader
    batch_size = 2
    data_loader_ = torch.utils.data.DataLoader(
        data_loader,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn
    )

    # 遍历并打印每个 batch 的内容
    for batch_idx, batch in tqdm(enumerate(data_loader_)):
        print(f"Batch {batch_idx + 1}:")
        
        # 遍历并打印 batch 字典中的每个键和值的形状或长度
        for key, value in batch.items():
            if hasattr(value, 'shape'):
                print(f"{key}: shape = {value.shape}")
            else:
                print(f"{key}: len = {len(value)}")
        
        # 只处理第一个 batch
        break

if __name__ == "__main__":
    main()