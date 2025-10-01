import numpy as np
from scipy.optimize import linear_sum_assignment
from numba import jit
import copy
import math

class Tracker(object):
    def __init__(self, opt):
        self.opt = opt
        self.motion_thresh = 15.0        # 运动匹配距离阈值
        self.iou_thresh = 0.3            # IOU匹配阈值
        self.dist_thresh = 15.0          # 距离匹配阈值
        self.size_factor = 0.3           # 尺寸过滤系数
        self.confidence_decay = 0.2     # 置信度衰减因子
        self.min_track_streak = 3        # 最小连续匹配次数
        
        self.reset()
    
    def reset(self):
        self.id_count = 0
        self.tracks = []  
        self.kalman_trackers = {}  
        self.track_streaks = {}    
    
    def init_track(self, results):
        for item in results:
            if item['score'] > self.opt.new_thresh:
                self.id_count += 1
         
                item['active'] = 1
                item['age'] = 1
                item['tracking_id'] = self.id_count
                
                self.init_kalman(item)
                
                if not ('ct' in item):
                    bbox = item['bbox']
                    item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                self.tracks.append(item)
                self.track_streaks[self.id_count] = 1  

    def init_kalman(self, item):
        if 'tracking_id' not in item:
            return
            
        bbox = item['bbox']
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        state = {
            'mean': np.array([cx, cy, w, h, 0, 0], dtype=np.float32),
            'covariance': np.eye(6, dtype=np.float32) * 8  
        }
        self.kalman_trackers[item['tracking_id']] = state
    
    def predict_kalman(self, track_id):
        if track_id not in self.kalman_trackers:
            return None
            
        state = self.kalman_trackers[track_id]
        F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        mean_pred = F.dot(state['mean'])
        covariance_pred = F.dot(state['covariance']).dot(F.T) + np.eye(6) * 0.1
        
        self.kalman_trackers[track_id] = {
            'mean': mean_pred,
            'covariance': covariance_pred
        }
        
        return mean_pred[0:2]
    
    def update_kalman(self, track_id, measurement):
        if track_id not in self.kalman_trackers:
            return
            
        state = self.kalman_trackers[track_id]
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        z = np.array(measurement, dtype=np.float32)
        
        S = H.dot(state['covariance']).dot(H.T) + np.eye(2) * 0.05
        K = state['covariance'].dot(H.T).dot(np.linalg.inv(S))
        
        y = z - H.dot(state['mean'])
        mean_updated = state['mean'] + K.dot(y)
        covariance_updated = (np.eye(6) - K.dot(H)).dot(state['covariance'])
        
        self.kalman_trackers[track_id] = {
            'mean': mean_updated,
            'covariance': covariance_updated
        }
    
    def step(self, results, public_det=None):
        for track in self.tracks:
            track_id = track['tracking_id']
            pred_ct = self.predict_kalman(track_id)
            if pred_ct is not None:
                track['pred_ct'] = pred_ct
        
        matched_tracks = set()
        matched_dets = set()
        ret = []
        
        # 第一级: 运动匹配 (基于卡尔曼预测)
        if len(self.tracks) > 0 and len(results) > 0:
            matches1 = self.match_by_motion(results)
            for m in matches1:
                track_idx = m[1]
                det_idx = m[0]
                track = self.tracks[track_idx]
                det = results[det_idx]
                
                det['tracking_id'] = track['tracking_id']
                det['age'] = 1
                det['active'] = track['active'] + 1
                ret.append(det)
                
                if 'ct' in det:
                    self.update_kalman(track['tracking_id'], det['ct'])

                if track['tracking_id'] in self.track_streaks:
                    self.track_streaks[track['tracking_id']] += 1
                else:
                    self.track_streaks[track['tracking_id']] = 1
                
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)
        
        # 第二级: IOU匹配
        if len(self.tracks) > 0 and len(results) > 0:
            matches2 = self.match_by_iou(results, matched_tracks, matched_dets)
            for m in matches2:
                track_idx = m[1]
                det_idx = m[0]
                track = self.tracks[track_idx]
                det = results[det_idx]
                
                det['tracking_id'] = track['tracking_id']
                det['age'] = 1
                det['active'] = track['active'] + 1
                ret.append(det)
                
                if 'ct' in det:
                    self.update_kalman(track['tracking_id'], det['ct'])
                
                if track['tracking_id'] in self.track_streaks:
                    self.track_streaks[track['tracking_id']] += 1
                else:
                    self.track_streaks[track['tracking_id']] = 1
                
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)
        
        # 第三级: 位置匹配 (原始距离匹配)
        if len(self.tracks) > 0 and len(results) > 0:
            matches3 = self.match_by_distance(results, matched_tracks, matched_dets)
            for m in matches3:
                track_idx = m[1]
                det_idx = m[0]
                track = self.tracks[track_idx]
                det = results[det_idx]
                
                det['tracking_id'] = track['tracking_id']
                det['age'] = 1
                det['active'] = track['active'] + 1
                ret.append(det)
                
                if 'ct' in det:
                    self.update_kalman(track['tracking_id'], det['ct'])
                
                if track['tracking_id'] in self.track_streaks:
                    self.track_streaks[track['tracking_id']] += 1
                else:
                    self.track_streaks[track['tracking_id']] = 1
                
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)
        
        unmatched_dets = [d for d in range(len(results)) if d not in matched_dets]
        for i in unmatched_dets:
            track = results[i]
            if track['score'] > self.opt.new_thresh:
                self.id_count += 1
                track['active'] = 1
                track['age'] = 1
                track['tracking_id'] = self.id_count
                
                self.init_kalman(track)
                
                if not ('ct' in track):
                    bbox = track['bbox']
                    track['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                ret.append(track)
                self.track_streaks[self.id_count] = 1  
        
        unmatched_tracks = [d for d in range(len(self.tracks)) if d not in matched_tracks]
        for i in unmatched_tracks:
            track = self.tracks[i]
            if track['age'] < self.opt.max_age:
                track['age'] += 1
                track['active'] = 0
                
                if 'pred_ct' in track:
                    pred_ct = track['pred_ct']
                    v = [pred_ct[0] - track['ct'][0], pred_ct[1] - track['ct'][1]]
                    bbox = track['bbox']
                    track['bbox'] = [
                        bbox[0] + v[0], bbox[1] + v[1],
                        bbox[2] + v[0], bbox[3] + v[1]]
                    track['ct'] = [pred_ct[0], pred_ct[1]]
                
                track['score'] = track['score'] * self.confidence_decay
                
                if track['tracking_id'] in self.track_streaks:
                    if self.track_streaks[track['tracking_id']] < self.min_track_streak:
                        self.track_streaks[track['tracking_id']] = max(
                            1, self.track_streaks[track['tracking_id']] - 1)
                
                ret.append(track)
        
        self.tracks = ret
        return ret

    def match_by_motion(self, results):
        if len(self.tracks) == 0 or len(results) == 0:
            return []
        
        dets_ct = np.array([det['ct'] for det in results], dtype=np.float32)
        
        tracks_pred_ct = []
        valid_tracks = []
        for i, track in enumerate(self.tracks):
            if 'pred_ct' in track:
                tracks_pred_ct.append(track['pred_ct'])
                valid_tracks.append(i)
        
        if not tracks_pred_ct:
            return []
            
        tracks_pred_ct = np.array(tracks_pred_ct, dtype=np.float32)
        
        diff_x = dets_ct[:, 0, None] - tracks_pred_ct[None, :, 0]
        diff_y = dets_ct[:, 1, None] - tracks_pred_ct[None, :, 1]
        dist_matrix = np.sqrt(diff_x ** 2 + diff_y ** 2)
        
        class_matrix = np.zeros_like(dist_matrix)
        for d, det in enumerate(results):
            for t_idx, track_idx in enumerate(valid_tracks):
                track = self.tracks[track_idx]
                if det.get('class', -1) != track.get('class', -2):
                    class_matrix[d, t_idx] = 1000
        
        size_matrix = np.zeros_like(dist_matrix)
        for t_idx, track_idx in enumerate(valid_tracks):
            track = self.tracks[track_idx]
            bbox = track['bbox']
            track_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            size_matrix[:, t_idx] = track_size * self.size_factor
        
        cost_matrix = dist_matrix + class_matrix
        cost_matrix[dist_matrix > size_matrix] = 1e9
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.motion_thresh:  
                orig_track_idx = valid_tracks[c]
                matches.append([r, orig_track_idx])
        
        return matches

    def match_by_iou(self, results, matched_tracks, matched_dets):
        if len(self.tracks) == 0 or len(results) == 0:
            return []
        
        unmatched_tracks = [t for t in range(len(self.tracks)) if t not in matched_tracks]
        unmatched_dets = [d for d in range(len(results)) if d not in matched_dets]
        
        if len(unmatched_tracks) == 0 or len(unmatched_dets) == 0:
            return []
        
        det_bboxes = np.array([results[i]['bbox'] for i in unmatched_dets], dtype=np.float32)
        track_bboxes = np.array([self.tracks[i]['bbox'] for i in unmatched_tracks], dtype=np.float32)
        
        iou_matrix = self.batch_calc_iou(det_bboxes, track_bboxes)
        
        for i, d_idx in enumerate(unmatched_dets):
            det = results[d_idx]
            for j, t_idx in enumerate(unmatched_tracks):
                track = self.tracks[t_idx]
                if det.get('class', -1) != track.get('class', -2):
                    iou_matrix[i, j] = 0
        
        cost_matrix = 1.0 - iou_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] > self.iou_thresh:
                orig_d_idx = unmatched_dets[r]
                orig_t_idx = unmatched_tracks[c]
                matches.append([orig_d_idx, orig_t_idx])
        
        return matches

    def match_by_distance(self, results, matched_tracks, matched_dets):
        if len(self.tracks) == 0 or len(results) == 0:
            return []
        
        unmatched_tracks = [t for t in range(len(self.tracks)) if t not in matched_tracks]
        unmatched_dets = [d for d in range(len(results)) if d not in matched_dets]
        
        if len(unmatched_tracks) == 0 or len(unmatched_dets) == 0:
            return []
        
        dets_ct = np.array([results[i]['ct'] for i in unmatched_dets], dtype=np.float32)
        
        tracks_ct = np.array([self.tracks[i]['ct'] for i in unmatched_tracks], dtype=np.float32)
        
        diff_x = dets_ct[:, 0, None] - tracks_ct[None, :, 0]
        diff_y = dets_ct[:, 1, None] - tracks_ct[None, :, 1]
        dist_matrix = np.sqrt(diff_x ** 2 + diff_y ** 2)
        
        class_matrix = np.zeros_like(dist_matrix)
        for i, d_idx in enumerate(unmatched_dets):
            det = results[d_idx]
            for j, t_idx in enumerate(unmatched_tracks):
                track = self.tracks[t_idx]
                if det.get('class', -1) != track.get('class', -2):
                    class_matrix[i, j] = 1000
        
        size_matrix = np.zeros_like(dist_matrix)
        for j, t_idx in enumerate(unmatched_tracks):
            track = self.tracks[t_idx]
            bbox = track['bbox']
            track_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            size_matrix[:, j] = track_size * self.size_factor
        
        cost_matrix = dist_matrix + class_matrix
        cost_matrix[dist_matrix > size_matrix] = 1e9
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.dist_thresh:  
                orig_d_idx = unmatched_dets[r]
                orig_t_idx = unmatched_tracks[c]
                matches.append([orig_d_idx, orig_t_idx])
        
        return matches

    @staticmethod
    @jit(nopython=True)
    def calc_iou(bbox1, bbox2):

        xx1 = max(bbox1[0], bbox2[0])
        yy1 = max(bbox1[1], bbox2[1])
        xx2 = min(bbox1[2], bbox2[2])
        yy2 = min(bbox1[3], bbox2[3])
        
        w = max(0.0, xx2 - xx1)
        h = max(0.0, yy2 - yy1)
        inter = w * h
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        iou = inter / (area1 + area2 - inter + 1e-8)
        return iou

    @staticmethod
    @jit(nopython=True, parallel=False)  
    def batch_calc_iou(det_bboxes, track_bboxes):
        n = det_bboxes.shape[0]
        m = track_bboxes.shape[0]
        iou_matrix = np.zeros((n, m), dtype=np.float32)
        
        for i in range(n):
            for j in range(m):
                bbox1 = det_bboxes[i]
                bbox2 = track_bboxes[j]
                
                xx1 = max(bbox1[0], bbox2[0])
                yy1 = max(bbox1[1], bbox2[1])
                xx2 = min(bbox1[2], bbox2[2])
                yy2 = min(bbox1[3], bbox2[3])
                
                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h
                
                area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                
                iou = inter / (area1 + area2 - inter + 1e-8)
                iou_matrix[i, j] = iou
                
        return iou_matrix

def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)