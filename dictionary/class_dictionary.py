# class name mapping table for each different dataset.
kitti_dict = {'to_waymo': {'Car': 'VEHICLE',
                           'Van': 'VEHICLE',
                           'Truck': 'VEHICLE',
                           'Pedestrian': 'PEDESTRIAN',
                           'Person_sitting': 'PEDESTRIAN',
                           'Cyclist': 'CYCLIST',
                           'Tram': None,
                           'Misc': 'UNKNOWN',
                           'DontCare': 'UNKNOWN',
                           },
              'to_nuscenes': {'Car': 'vehicle.car',
                              'Van': 'vehicle.car',
                              'Truck': 'vehicle.truck',
                              'Pedestrian': 'human.pedestrian.adult',
                              'Person_sitting': 'human.pedestrian.adult',
                              'Cyclist': 'vehicle.bicycle',
                              'Tram': None,
                              'Misc': None,
                              'DontCare': None
                              }
              }
waymo_dict = {'to_kitti': {'UNKNOWN': 'DontCare',
                           'VEHICLE': 'Car',
                           'PEDESTRIAN': 'Pedestrian',
                           'SIGN': 'DontCare',
                           'CYCLIST': 'Cyclist'
                           },
              'to_nuscenes': {'UNKNOWN': None,
                              'VEHICLE': 'vehicle.car',
                              'PEDESTRIAN': 'human.pedestrian.adult',
                              'SIGN': None,
                              'CYCLIST': 'vehicle.bicycle'
                              }
              }
nuscenes_dict = {'to_kitti': {'human.pedestrian.adult': 'Pedestrian',
                              'human.pedestrian.child': 'Pedestrian',
                              'vehicle.bicycle': 'Cyclist',
                              'vehicle.car': 'Car',
                              'vehicle.motorcycle': 'Cyclist',
                              'vehicle.truck': 'Car'
                              },
                 'to_waymo': {'human.pedestrian.adult': 'PEDESTRIAN',
                              'human.pedestrian.child': 'PEDESTRIAN',
                              'vehicle.bicycle': 'CYCLIST',
                              'vehicle.car': 'VEHICLE',
                              'vehicle.motorcycle': 'CYCLIST',
                              'vehicle.truck': 'VEHICLE'
                              }
                 }
