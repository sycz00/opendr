# Copyright 1996-2020 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cv2
from engine.target import Pose

# More information on body-part id naming on target.py - Pose class.
# For in-depth explanation of BODY_PARTS_KPT_IDS and BODY_PARTS_PAF_IDS see
#  https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/TRAIN-ON-CUSTOM-DATASET.md
BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
                      [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19],
                      [26, 27])
sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                  dtype=np.float32) / 10.0
vars_ = (sigmas * 2) ** 2
last_id = -1
color = [0, 224, 255]


def get_bbox(keypoints):
    found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
    found_kpt_id = 0
    for kpt_id in range(Pose.num_kpts):
        if keypoints[kpt_id, 0] == -1:
            continue
        found_keypoints[found_kpt_id] = keypoints[kpt_id]
        found_kpt_id += 1
    bbox = cv2.boundingRect(found_keypoints)
    return bbox


def update_id(pose, id_=None):
    pose.id = id_
    if pose.id is None:
        pose.id = Pose.last_id + 1
        Pose.last_id += 1


def draw(img, pose):
    """
    Draws the provided pose on the provided image.

    :param img: the image to draw the pose on
    :param pose: the pose to draw on the image
    """
    assert pose.data.shape == (Pose.num_kpts, 2)

    for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        global_kpt_a_id = pose.data[kpt_a_id, 0]
        x_a, y_a, x_b, y_b = 0, 0, 0, 0
        if global_kpt_a_id != -1:
            x_a, y_a = pose.data[kpt_a_id]
            cv2.circle(img, (int(x_a), int(y_a)), 3, color, -1)
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
        global_kpt_b_id = pose.data[kpt_b_id, 0]
        if global_kpt_b_id != -1:
            x_b, y_b = pose.data[kpt_b_id]
            cv2.circle(img, (int(x_b), int(y_b)), 3, color, -1)
        if global_kpt_a_id != -1 and global_kpt_b_id != -1:
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), color, 2)


def get_similarity(a, b, threshold=0.5):
    """
    Calculates the Keypoint Similarity, explained in detail on the official COCO dataset site
    https://cocodataset.org/#keypoints-eval

    :param a: first pose
    :param b: second pose
    :param threshold: the similarity threshold to consider the keypoints similar
    :return: number of similar keypoints
    :rtype: int
    """
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.data[kpt_id, 0] != -1 and b.data[kpt_id, 0] != -1:
            distance = np.sum((a.data[kpt_id] - b.data[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * vars_[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt


def track_poses(previous_poses, current_poses, threshold=3, smooth=False):
    """
    Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.
    If correspondence between pose on previous and current frame was established, pose keypoints are smoothed.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :param smooth: smooth pose keypoints between frames
    """
    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose in current_poses:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for id_, previous_pose in enumerate(previous_poses):
            if not mask[id_]:
                continue
            iou = get_similarity(current_pose, previous_pose)
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_pose.id
                best_matched_id = id_
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_pose.update_id(best_matched_pose_id)

        # Smooth feature needs to use per-pose filters. Filters cannot be saved in OpenDR Pose, due to the fact that
        # they are implementation-specific. Code is commented out until a better solution is found.

        # if smooth:
        #     for kpt_id in range(Pose.num_kpts):
        #         if current_pose.data[kpt_id, 0] == -1:
        #             continue
        #         # reuse filter if previous pose has valid filter
        #         if (best_matched_pose_id is not None
        #                 and previous_poses[best_matched_id].data[kpt_id, 0] != -1):
        #             current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
        #         current_pose.data[kpt_id, 0] = current_pose.filters[kpt_id][0](current_pose.data[kpt_id, 0])
        #         current_pose.data[kpt_id, 1] = current_pose.filters[kpt_id][1](current_pose.data[kpt_id, 1])
        #     current_pose.bbox = get_bbox(current_pose.data)
