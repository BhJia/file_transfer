import os
import json
import shutil
import tqdm

def copy_data_folder(vid, mask_id, data_path, save_data_path):
    src_inpaint_path = os.path.join(data_path, "InpaintedImages", vid, mask_id)
    src_jpeg_path = os.path.join(data_path, "JPEGImages", vid)
    src_mask_path = os.path.join(data_path, "Annotations", vid)
    dst_inpaint_path = os.path.join(save_data_path, "InpaintedImages", vid+"_"+mask_id)
    dst_jpeg_path = os.path.join(save_data_path, "JPEGImages", vid+"_"+mask_id)
    dst_mask_path = os.path.join(save_data_path, "Annotations", vid+"_"+mask_id)

    shutil.copytree(src_inpaint_path, dst_inpaint_path)
    shutil.copytree(src_jpeg_path, dst_jpeg_path)
    shutil.copytree(src_mask_path, dst_mask_path)


data_meta_path = "VPLM_gt_train.json"
save_data_meta_path = "format_gt_train.json"
data_path = "./data"
save_data_path = "data_add_train"


with open(data_meta_path, 'r') as f:
    data = json.load(f)

result = []

for i in tqdm.tqdm(data):
    vid, mask_id, description, task = i['vid'], i['mask_id'], i['description'], i['task']
    if task!="adding":
        continue
    copy_data_folder(vid, mask_id, data_path, save_data_path)
    res_dict = dict(video = i['vid'] + "_" + mask_id, query = i['description'], response = i['description'])
    result.append(res_dict)

with open(save_data_meta_path, 'w') as f:
    json.dump(result, f)