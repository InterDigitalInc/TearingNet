    '''
    Modify based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/kitti/kitti_dataset.py
    replace the create_groundtruth_database() function in kitti_dataset.py of OpenPCDet by the following one.
    '''

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch
        from itertools import combinations

        database_save_path = Path(self.root_path) / ('kitti_single' if split == 'train' else ('kitti_single_%s' % split))
        database_save_path.mkdir(parents=True, exist_ok=True)
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        # Parameters
        num_point = 1536
        max_num_obj = 22
        interest_class = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck']
        all_db_infos=[]
        db_obj=[]
        for obj_col in range(max_num_obj):
            all_db_infos.append([])
        all_db_infos.append(np.zeros(max_num_obj, dtype=int)) # accumulator
        db_obj_save_path = Path(self.root_path) / ('kitti_dbinfos_object.pkl')

        frame_obj_list = np.zeros((len(infos), max_num_obj), dtype=int)
        for k in range(len(infos)):
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            # Count the object occurence
            for obj_id in range(num_obj):
                if (names[obj_id] in interest_class) == False: continue
                filename = '%d_%d_%s.bin' % (k, obj_id, names[obj_id])
                filepath = database_save_path / filename
                db_path = str(filepath.relative_to(self.root_path))  # kitti_single/xxxxx.bin
                gt_points = points[point_indices[obj_id] > 0]

                db_obj.append({'name': names[obj_id], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': obj_id,
                    'box3d_lidar': gt_boxes[obj_id], 'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[obj_id], 'bbox': bbox[obj_id], 'score': annos['score'][obj_id]})

                with open(filepath, 'w') as f:
                    gt_points.tofile(f)
                for obj_col in range(max_num_obj):
                    if gt_points.shape[0] >= np.ceil(num_point/(obj_col+1)).astype(int):
                        frame_obj_list[k,obj_col] += 2 ** obj_id

            # Conclude how a frame can be used
            for obj_col in range(max_num_obj):
                if bin(frame_obj_list[k,obj_col])[2:].count('1') >= obj_col + 1:
                    obj_indicator = np.array(list(bin(frame_obj_list[k,obj_col])[2:].zfill(max_num_obj)[::-1]))=='1'
                    obj_choice = np.arange(max_num_obj)[obj_indicator]
                    comb = combinations(obj_choice, obj_col+1) 
                    for obj_scene in list(comb):

                        # Write down the scene configuration
                        db_info=[]
                        for obj_id in obj_scene:
                            filename = '%d_%d_%s.bin' % (k, obj_id, names[obj_id])
                            filepath = database_save_path / filename
                            db_path = str(filepath.relative_to(self.root_path))  # kitti_single/xxxxx.bin
                            db_info.append({'name': names[obj_id], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': obj_id,
                               'box3d_lidar': gt_boxes[obj_id], 'difficulty': difficulty[obj_id], 'bbox': bbox[obj_id], 
                               'score': annos['score'][obj_id]})
                        all_db_infos[len(obj_scene)-1].append(db_info)
                        all_db_infos[-1][len(obj_scene)-1] += 1
                        print(k,obj_scene)

        with open(db_obj_save_path, 'wb') as f:
            print(f.name)
            pickle.dump(db_obj, f)
