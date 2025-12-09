import os
import json
import os
import json
import random
from typing import List, Dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


category_list = ['gibbon', 'person', 'marmot', 'snow leopard', 'wind', 'truck', 'blue whale', 'bus', 'llama', 'lizard', 'wildebeest', 'raccoon', 'gunner', 'frog', 'bulldog', 'dog', 'bear', 'cartoon', 'camel', 'alpaca', 'mage', 'bicycle', 'rider', 'ermine', 'woodworker', 'mouse', 'turtle', 'soldier', 'sheep', 'sculpture', 'hamburger', 'squirrel', 'red-panda', 'autorickshaw', 'kapibala', 'elephant', 'duck', 'musician', 'hyena', 'rabbit', 'buffalo', 'malamute', 'performer', 'dragon', 'serval', 'dinosaur', 'orcs', 'mouth', 'motorcycle', 'hand', 'pig', 'lion', 'lynx', 'toy', 'cat', 'tiger', 'cow', 'robot', 'deer', 'hamster', 'other', 'monkey', 'chef', 'red fox', 'giraffe', 'husky dog', 'panda', 'tigger', 'dancer', 'horseman', 'caribou', 'donkey', 'caracal', 'manatee', 'fox', 'terrier', 'car', 'cheetah', 'animal', 'killer whale', 'horse', 'seal', 'extinguisher', 'chinchilla', 'person_or_animal', 'athlete', 'jaguar', 'flying fox']


motion_list = ['daily_action-Chinese_traffic_gesture-LANE_CHANGING', 'daily_action-Chinese_traffic_gesture-LEFT_TURN', 'daily_action-Chinese_traffic_gesture-LEFT_TURN_WAITING', 'daily_action-Chinese_traffic_gesture-MOVE_STRAIGHT', 'daily_action-Chinese_traffic_gesture-PULL_OVER', 'daily_action-Chinese_traffic_gesture-RIGHT_TURN', 'daily_action-Chinese_traffic_gesture-SLOW_DOWN', 'daily_action-Chinese_traffic_gesture-STOP', 'daily_action-commuting-push_car', 'daily_action-commuting-ride_bicycle', 'daily_action-commuting-ride_bike', 'daily_action-commuting-ride_horse', 'daily_action-commuting-ride_motorcycle', 'daily_action-commuting-ride_scooter', 'daily_action-commuting-ride_tricycle', 'daily_action-commuting-riding_elephant', 'daily_action-commuting-segway', 'daily_action-emergency-CPR', 'daily_action-emergency-bandaging', 'daily_action-emergency-fire_extinguisher', 'daily_action-emergency-heimlich_maneuver', 'daily_action-emergency-rescue_breathing', 'daily_action-hand_gesture-Applaud', 'daily_action-hand_gesture-Dual_hands_heart', 'daily_action-hand_gesture-Make_a_phone_call', 'daily_action-hand_gesture-Measure_distance', 'daily_action-hand_gesture-Pause', 'daily_action-hand_gesture-Sweep_Checkmark', 'daily_action-hand_gesture-Sweep_Circle', 'daily_action-hand_gesture-Sweep_Cross', 'daily_action-hand_gesture-Sweep_diagonal', 'daily_action-hand_gesture-Walk', 'daily_action-hand_gesture-Wave_finger', 'daily_action-hand_gesture-Wave_hand', 'daily_action-hand_gesture-Zoom_in_with_fingers', 'daily_action-hand_gesture-Zoom_out_with_fingers', 'daily_action-hand_gesture-knock', 'daily_action-housework-carpentry', 'daily_action-housework-chopping', 'daily_action-housework-cleaning_glass_surface', 'daily_action-housework-hanging_clothes', 'daily_action-housework-ironing', 'daily_action-housework-knitting', 'daily_action-housework-plunging_toilet', 'daily_action-housework-shoe_shining', 'daily_action-housework-sweeping_floor', 'daily_action-housework-wall_painting', 'daily_action-housework-washing', 'daily_action-housework-watering_plants', 'daily_action-phylogenetic_action-drinking', 'daily_action-phylogenetic_action-eat_cake', 'daily_action-phylogenetic_action-gives_birth_to_a_baby', 'daily_action-phylogenetic_action-hunts_other_animals', 'daily_action-phylogenetic_action-sleep', 'daily_action-sign_language-bed', 'daily_action-sign_language-before', 'daily_action-sign_language-black', 'daily_action-sign_language-bowling', 'daily_action-sign_language-candy', 'daily_action-sign_language-computer', 'daily_action-sign_language-cool', 'daily_action-sign_language-cousin', 'daily_action-sign_language-deaf', 'daily_action-sign_language-drink', 'daily_action-sign_language-fine', 'daily_action-sign_language-finish', 'daily_action-sign_language-fire', 'daily_action-sign_language-go', 'daily_action-sign_language-help', 'daily_action-sign_language-hot', 'daily_action-sign_language-like', 'daily_action-sign_language-many', 'daily_action-sign_language-mother', 'daily_action-sign_language-no', 'daily_action-sign_language-orange', 'daily_action-sign_language-pizza', 'daily_action-sign_language-shirt', 'daily_action-sign_language-study', 'daily_action-sign_language-tall', 'daily_action-sign_language-thanksgiving', 'daily_action-sign_language-thin', 'daily_action-sign_language-walk', 'daily_action-sign_language-white', 'daily_action-sign_language-who', 'daily_action-sign_language-woman', 'daily_action-sign_language-wrong', 'daily_action-sign_language-yes', 'daily_action-social_action-be_bended', 'daily_action-social_action-be_boxed', 'daily_action-social_action-be_kicked', 'daily_action-social_action-be_pated', 'daily_action-social_action-be_pushed', 'daily_action-social_action-bend', 'daily_action-social_action-blowing_kisses', 'daily_action-social_action-box', 'daily_action-social_action-celebrate-night_night', 'daily_action-social_action-celebrate-siu', 'daily_action-social_action-celebrate-slippery_kneeling', 'daily_action-social_action-handshake', 'daily_action-social_action-highfive', 'daily_action-social_action-hug', 'daily_action-social_action-kick', 'daily_action-social_action-laugh', 'daily_action-social_action-pat', 'daily_action-social_action-push', 'daily_action-social_action-salute', 'daily_action-social_action-wave', 'daily_action-vehicle_behavior-Confrontation', 'daily_action-vehicle_behavior-Crossing', 'daily_action-vehicle_behavior-Cut_in', 'daily_action-vehicle_behavior-Deviate', 'daily_action-vehicle_behavior-Left_Turn', 'daily_action-vehicle_behavior-Merging', 'daily_action-vehicle_behavior-Overtake', 'daily_action-vehicle_behavior-Right_Turn', 'daily_action-vehicle_behavior-U_Turn', 'entertainment-billiard_hit', 'entertainment-building_snowman', 'entertainment-dart_throw', 'entertainment-discuss_throw', 'entertainment-draw_handgun', 'entertainment-musical_instrument-play_bamboon_flute', 'entertainment-musical_instrument-play_cello', 'entertainment-musical_instrument-play_erhu', 'entertainment-musical_instrument-play_grandpiano', 'entertainment-musical_instrument-play_guitar', 'entertainment-musical_instrument-play_jazzdrum', 'entertainment-musical_instrument-play_piano', 'entertainment-musical_instrument-play_pipa', 'entertainment-musical_instrument-play_saxophone', 'entertainment-musical_instrument-play_suona', 'entertainment-musical_instrument-play_trumpet', 'entertainment-musical_instrument-play_violin', 'entertainment-performance-card_throw', 'entertainment-performance-dice_stack_shuffle', 'entertainment-performance-fire_breathing', 'entertainment-performance-handstand', 'entertainment-performance-play_yoyo', 'entertainment-performance-playing_rubiks_cube', 'entertainment-performance-shake_cocktail', 'entertainment-performance-spinning_plate', 'entertainment-performance-sword_swallowing', 'entertainment-photography', 'entertainment-read', 'entertainment-shoot_gun', 'entertainment-smoke', 'entertainment-spraying_wall', 'entertainment-stone_skipping', 'special-magic', 'special-self_replication', 'special-teleport', 'special-transform-metamorphosis', 'special-transform-reversion', 'special-transform-transformation', 'sports-baseball-bunt', 'sports-baseball-catch_catcher', 'sports-baseball-pitch', 'sports-baseball-swing', 'sports-basketball-drive', 'sports-basketball-dunk', 'sports-basketball-free_throw', 'sports-basketball-hookshot', 'sports-basketball-interfere_shot', 'sports-basketball-jabstep', 'sports-basketball-layup', 'sports-basketball-pass', 'sports-basketball-screen', 'sports-basketball-shoot', 'sports-basketball-spinning_basketball', 'sports-dance-ballet_jazz', 'sports-dance-break', 'sports-dance-house', 'sports-dance-krump', 'sports-dance-la_style_hip_hop', 'sports-dance-lock', 'sports-dance-middle_hip_hop', 'sports-dance-moonwalk', 'sports-dance-pop', 'sports-dance-street_jazz', 'sports-dance-waack', 'sports-golf', 'sports-gym-burpee', 'sports-gym-dips', 'sports-gym-frog_jump', 'sports-gym-gym_push', 'sports-gym-hulahoop', 'sports-gym-lift', 'sports-gym-lunges', 'sports-gym-plank', 'sports-gym-pull', 'sports-gym-pull_up', 'sports-gym-pushup', 'sports-gym-situp', 'sports-gym-squat', 'sports-gymnastics-balancebeam_rotate', 'sports-gymnastics-balancebeam_walk', 'sports-gymnastics-flip', 'sports-gymnastics-jump', 'sports-gymnastics-spin', 'sports-gymnastics-unevenbar_cross', 'sports-punch', 'sports-run_starting', 'sports-skiing', 'sports-soccer-diving', 'sports-soccer-header', 'sports-soccer-shoot', 'sports-soccer-steal', 'sports-soccer-throw', 'sports-surfing', 'sports-swimming-backstroke', 'sports-swimming-breast_stroke', 'sports-swimming-butterfly_stroke', 'sports-swimming-freestyle', 'sports-taichi_fan', 'sports-volleyball-adjust', 'sports-volleyball-overhand', 'sports-volleyball-pass', 'sports-volleyball-underhand', 'sports-yoga-yoga_cat', 'sports-yoga-yoga_dancer', 'sports-yoga-yoga_firefly', 'sports-yoga-yoga_gate', 'sports-yoga-yoga_pigeon', 'sports-yoga-yoga_tree', 'daily_action_CityMove_car', 'daily_action_CityMove_person', 'daily_action_SportMove_person', 'daily_action_WildMove_animal']


class ActionHierarchy:
    def __init__(self, action_groups_path: str, group_id, train=False):
        self.action_groups_path = action_groups_path
        self.action_groups = self._load_action_groups()
        if train:
            if isinstance(group_id, int):
                self.action_groups = [self.action_groups[i] for i in range(4) if i != group_id]
            elif isinstance(group_id, list):
                self.action_groups = [self.action_groups[i] for i in range(4) if i not in group_id]
            else: raise NotImplementedError
        else:
            print(len(self.action_groups))
            if isinstance(group_id, int):
                self.action_groups = [self.action_groups[group_id]]
            elif isinstance(group_id, list):
                self.action_groups = [self.action_groups[i] for i in range(4) if i in group_id]
            else: raise NotImplementedError
        self.action_hierarchy = self._build_hierarchy()

        
    def _load_action_groups(self) -> List[Dict]:
        with open(self.action_groups_path, 'r') as f:
            return json.load(f)
    
    def _build_hierarchy(self) -> Dict:
        hierarchy = {}
        
        for group in self.action_groups:
            for action in group['actions']:
                parts = action.split('-')
                if len(parts) >= 2:
                    current_level = hierarchy
                    for i, part in enumerate(parts[:-1]):
                        if part not in current_level:
                            current_level[part] = {}
                        current_level = current_level[part]
                    if parts[-1]:  # if the last level is not empty
                        if 'actions' not in current_level:
                            current_level['actions'] = set()
                        current_level['actions'].add(parts[-1])
        
        return hierarchy
    
    def get_similar_actions(self, action: str, num_actions: int, level: int = None) -> List[str]:
        """Get similar actions list for given action
        
        Args:
            action: target action
            num_actions: number of similar actions to return
            level: specified search level, None means using original level
            
        Returns:
            similar actions list
        """
        parts = action.split('-')
        if level is not None:
            parts = parts[:level]
        similar_actions = []
        
        def traverse_hierarchy(current_dict, current_path):
            if 'actions' in current_dict:
                for act in current_dict['actions']:
                    full_path = '-'.join(current_path + [act])
                    if full_path != action:
                        similar_actions.append(full_path)
            for key in current_dict:
                if key != 'actions' and isinstance(current_dict[key], dict):
                    traverse_hierarchy(current_dict[key], current_path + [key])
        
        # search from same prefix
        current_dict = self.action_hierarchy
        for i in range(len(parts)-1):
            if parts[i] in current_dict:
                current_dict = current_dict[parts[i]]
            else:
                break
        
        traverse_hierarchy(current_dict, parts[:-1])
        
        if len(similar_actions) > num_actions:
            return random.sample(similar_actions, num_actions)
        return similar_actions
    
    def sample_fine_grained_episode(self, num_ways: int) -> List[str]:
        all_actions = []
        
        def collect_actions(current_dict, current_path):
            if 'actions' in current_dict:
                for act in current_dict['actions']:
                    all_actions.append('-'.join(current_path + [act]))
            for key in current_dict:
                if key != 'actions' and isinstance(current_dict[key], dict):
                    collect_actions(current_dict[key], current_path + [key])
        
        collect_actions(self.action_hierarchy, [])
        
        base_action = random.choice(all_actions)
        selected_actions = [base_action]
        
        current_level = len(base_action.split('-'))
        while len(selected_actions) < num_ways and current_level > 1:
            similar_actions = self.get_similar_actions(base_action, num_ways - len(selected_actions), current_level)
            selected_actions.extend(similar_actions)
            
            if len(selected_actions) < num_ways:
                current_level -= 1
                
        if len(selected_actions) < num_ways:
            remaining_actions = list(set(all_actions) - set(selected_actions))
            if remaining_actions:
                additional_actions = random.sample(remaining_actions, min(num_ways - len(selected_actions), len(remaining_actions)))
                selected_actions.extend(additional_actions)
        
        return selected_actions[:num_ways]


class MOVEDataset(Dataset):
    def __init__(self, 
                 data_path=None,
                 train=True, 
                 valid=False,
                 set_index=1, 
                 finetune_idx=None,
                 support_frames=10, 
                 query_frames=1,  # N-way-K-shot setting
                 num_ways=1, 
                 num_shots=5,  # N-way-K-shot setting
                 transforms=None, 
                 another_transform=None, 
                 group=0, 
                 setting='default', 
                 proposal_mask=False):
        self.train = train
        self.valid = valid
        self.set_index = set_index
        self.support_frames = support_frames
        self.query_frames = query_frames
        self.num_ways = num_ways  # N-way
        self.num_shots = num_shots  # K-shot
        self.transforms = transforms
        self.another_transform = another_transform
        self.group = group
        self.setting = setting
        
        self.proposal_mask = proposal_mask
        
        # Setup data paths
        self.data_dir = os.path.join('data', 'MOVE_release')
        self.img_dir = os.path.join(self.data_dir, 'frames')
        self.ann_dir = os.path.join(self.data_dir, 'annotations')

        # Initialize data structures
        self.video_ids = []
        self.action_categories = set()  # Store unique action categories
        self.category_to_videos = {}  # Map categories to video IDs
        self.video_to_categories = {}  # Map videos to their action categories
        self.action_segments = {}  # Store video length and objects info
        
        # Load annotations and collect action categories
        for vid in os.listdir(self.img_dir):
            if not os.path.isdir(os.path.join(self.img_dir, vid)):
                continue
                
            ann_file = os.path.join(self.ann_dir, f"{vid}.json")
            if os.path.exists(ann_file):
                with open(ann_file, 'r') as f:
                    ann_data = json.load(f)
                    if 'length' in ann_data and 'objects' in ann_data:
                        # Store basic video info
                        self.action_segments[vid] = {
                            'length': ann_data['length'],
                            'objects': ann_data['objects']
                        }
                        
                        # Extract action categories
                        video_categories = set()
                        for obj in ann_data['objects']:
                            if 'actions' in obj:
                                for action_info in obj['actions']:
                                    action_name = action_info['action']
                                    self.action_categories.add(action_name)
                                    video_categories.add(action_name)
                                    
                                    # Add to category mapping
                                    if action_name not in self.category_to_videos:
                                        self.category_to_videos[action_name] = []
                                    if vid not in self.category_to_videos[action_name]:
                                        self.category_to_videos[action_name].append(vid)
                        
                        # Store video's categories
                        self.video_to_categories[vid] = list(video_categories)

        # Sort categories for reproducibility
        self.action_categories = sorted(list(self.action_categories))
        
        # Load action groups from JSON
        if setting == 'default':
            action_groups_path = os.path.join(os.path.dirname(self.ann_dir), 'action_groups.json')
        elif setting == 'challenging':
            action_groups_path = os.path.join(os.path.dirname(self.ann_dir), 'challenging_group.json')
        else: assert False, 'setting input is not valid! please try "default" or "challenging"'
        with open(action_groups_path, 'r') as f:
            action_groups = json.load(f)
        
        # Determine train and test categories based on group
        if isinstance(self.group, int):
            test_categories = action_groups[self.group]['actions']
        elif isinstance(self.group, list):
            test_categories = []
            for group_id in self.group:
                test_categories.extend(action_groups[group_id]['actions'])
        else:
            raise NotImplementedError
        train_categories = [cat for cat in self.action_categories if cat not in test_categories]
        self.train_categories = train_categories
        self.test_categories = test_categories
        
        if train and not valid:
            self.selected_categories = sorted(list(self.action_categories)) # HACK
        else:
            self.selected_categories = sorted(list(self.action_categories)) # HACK
            
        # Update video IDs to only include those from selected categories
        self.video_ids = []
        for category in self.selected_categories:
            self.video_ids.extend(self.category_to_videos[category])
        self.video_ids = list(set(self.video_ids))  # Remove duplicates
            
        if finetune_idx is not None:
            self.video_ids = [self.video_ids[finetune_idx]]

        print(f"{'Train' if train else 'Test'} set: {len(self.video_ids)} videos")
        print(f"Number of action categories: {len(self.selected_categories)}")

        if train and not valid:
            self.action_hierarchy = ActionHierarchy(action_groups_path, self.group, train=True)
        else:
            self.action_hierarchy = ActionHierarchy(action_groups_path, self.group, train=False)

    def get_frames(self, video_id, frame_indices, action_name=None, proposal_mask=False):
        """
        Get frames and corresponding masks for a specific action
        Args:
            video_id: ID of the video
            frame_indices: List of frame indices to get
            action_name: Target action name to get masks for
            proposal_mask: Whether to return proposal masks that ignore action categories and timing
        """
        frames = []
        masks = []
        proposal_masks = []
        video_dir = os.path.join(self.img_dir, video_id)
        
        # Get all frame files and sort them
        frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        
        # Find objects that have the target action
        target_objects = []
        if action_name is not None and video_id in self.action_segments:
            for obj in self.action_segments[video_id]['objects']:
                if 'actions' in obj:
                    for action_info in obj['actions']:
                        if action_info['action'] == action_name:
                            # Check if frame is within action's time range
                            start_frame = action_info.get('start_frame', 0)
                            end_frame = action_info.get('end_frame', len(frame_files) - 1)
                            target_objects.append({
                                'object': obj,
                                'start_frame': start_frame,
                                'end_frame': end_frame
                            })
        
        # Load selected frames
        for idx in frame_indices:
            # Load image
            try:
                img_path = os.path.join(video_dir, frame_files[idx])
            except:
                print(f"Error loading image at index {idx} for video {video_id}")
            img = np.array(Image.open(img_path))
            frames.append(img)
            
            # Create combined mask for all objects with target action at this frame
            h, w = img.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            proposal_combined_mask = np.zeros((h, w), dtype=np.uint8)
            
            if video_id in self.action_segments:
                # For proposal mask - include all object masks regardless of action
                if proposal_mask:
                    for obj in self.action_segments[video_id]['objects']:
                        if obj['masks'][idx] is not None:
                            obj_mask = self.decode_mask(obj['masks'][idx])[:, :, 0]
                            proposal_combined_mask = np.logical_or(proposal_combined_mask, obj_mask)
                
                # For target action mask
                if action_name is not None and target_objects:
                    for target in target_objects:
                        obj = target['object']
                        start_frame = target['start_frame']
                        end_frame = target['end_frame']
                        
                        # Only include mask if frame is within action's time range
                        if start_frame <= idx <= end_frame:
                            if obj['masks'][idx] is not None:
                                obj_mask = self.decode_mask(obj['masks'][idx])[:, :, 0]
                                combined_mask = np.logical_or(combined_mask, obj_mask)
            
            masks.append(combined_mask)
            if proposal_mask:
                proposal_masks.append(proposal_combined_mask)
        
        if proposal_mask:
            return frames, masks, proposal_masks
        return frames, masks

    def decode_mask(self, mask_data):
        from pycocotools import mask as mask_utils
        
        h, w = mask_data['size']
        rle = {'size': [h, w], 'counts': mask_data['counts']}
        
        # Use COCO's decode function
        mask = mask_utils.decode(rle)
        
        # Add channel dimension and ensure uint8 type
        mask = np.expand_dims(mask, axis=2).astype(np.uint8)
        
        return mask

    def get_valid_frames(self, video_id, action_name, category=False):
        """Get valid frames for specified action in video
        Args:
            video_id: video ID
            action_name: action name
            category: whether to return category information
        Returns:
            valid frames list, all frames list, [object category], [action category]
        """
        action_frames = []
        video_data = self.action_segments[video_id]
        video_length = video_data['length']
        all_frames = list(range(video_length))
        all_categories = []
        all_motion_categories = []
        
        for obj in video_data['objects']:
            if 'actions' in obj:
                for action_info in obj['actions']:
                    if action_info['action'] == action_name:
                        start_frame = action_info.get('start_frame', 0)
                        end_frame = action_info.get('end_frame', video_length - 1)
                        end_frame = min(end_frame, video_length - 1)
                        action_frames.extend(range(start_frame, end_frame + 1))
                        if category:
                            all_categories.append(category_list.index(obj['category']))
                            all_motion_categories.append(motion_list.index(action_info['action']))
        if category:
            return sorted(list(set(action_frames))), all_frames, list(set(all_categories))[-1], list(set(all_motion_categories))[-1]
        return sorted(list(set(action_frames))), all_frames

    def sample_frames_with_action(self, valid_frames, all_frames, num_frames, min_action_frames=1):
        """Sample frames ensuring at least specified number of action frames
        Args:
            valid_frames: frames containing target action
            all_frames: all available frames
            num_frames: total number of frames to sample
            min_action_frames: minimum number of action frames required
        Returns:
            sampled frame indices list
        """
        if not valid_frames:
            return random.sample(all_frames, num_frames)
            
        # Ensure at least min_action_frames action frames are sampled
        num_action_frames = min(len(valid_frames), min(min_action_frames, num_frames))
        selected_action_frames = random.sample(valid_frames, num_action_frames)
        
        # Sample remaining frames from all frames
        remaining_frames = num_frames - num_action_frames
        other_frames = [f for f in all_frames if f not in selected_action_frames]
        if remaining_frames > 0:
            if len(other_frames) >= remaining_frames:
                selected_other_frames = random.sample(other_frames, remaining_frames)
            else:
                try:
                    selected_other_frames = random.choices(other_frames, k=remaining_frames)
                except:
                    print(f"Error sampling other frames for video {video_id}")
                    selected_other_frames = []
        else:
            selected_other_frames = []
        
        # Merge all selected frames and sort by time order
        selected_frames = sorted(selected_action_frames + selected_other_frames)
        return selected_frames

    def sample_frames_with_action_support(self, valid_frames, all_frames, num_frames, min_action_frames=1):
        """Sample frames ensuring at least specified number of action frames, uniformly sampling from entire action sequence
        Args:
            valid_frames: frames containing target action
            all_frames: all available frames
            num_frames: total number of frames to sample
            min_action_frames: minimum number of action frames required
        Returns:
            sampled frame indices list
        """
        if not valid_frames:
            return random.sample(all_frames, num_frames)
            
        # Ensure at least min_action_frames action frames are sampled, uniformly from entire action sequence
        num_action_frames = min(len(valid_frames), min(min_action_frames, num_frames))
        
        if num_action_frames == len(valid_frames):
            # If required action frames equal valid frames, use all
            selected_action_frames = valid_frames
        else:
            # Uniform sampling: calculate sampling interval and select frames
            valid_frames = sorted(valid_frames)
            
            # If not enough frames, expand from both ends
            if len(valid_frames) < num_action_frames:
                # Calculate additional frames needed
                extra_frames_needed = num_action_frames - len(valid_frames)
                
                # Expand from both ends
                left_expand = extra_frames_needed // 2
                right_expand = extra_frames_needed - left_expand
                
                # Get expandable frames from left and right
                min_valid = min(valid_frames)
                max_valid = max(valid_frames)
                
                left_candidates = [f for f in all_frames if f < min_valid]
                right_candidates = [f for f in all_frames if f > max_valid]
                
                # Select closest frames
                left_candidates.sort(reverse=True)  # descending order, select closest
                right_candidates.sort()  # ascending order, select closest
                
                left_expand_frames = left_candidates[:left_expand] if left_candidates else []
                right_expand_frames = right_candidates[:right_expand] if right_candidates else []
                
                # If one side expansion is insufficient, supplement from the other side
                if len(left_expand_frames) < left_expand and right_candidates:
                    right_expand_frames.extend(right_candidates[right_expand:right_expand+(left_expand-len(left_expand_frames))])
                elif len(right_expand_frames) < right_expand and left_candidates:
                    left_expand_frames.extend(left_candidates[left_expand:left_expand+(right_expand-len(right_expand_frames))])
                
                # Merge expanded frames
                expanded_valid_frames = left_expand_frames + valid_frames + right_expand_frames
                expanded_valid_frames.sort()
                
                # Uniform sampling
                step = len(expanded_valid_frames) / num_action_frames
                indices = [int(i * step) for i in range(num_action_frames)]
                selected_action_frames = [expanded_valid_frames[i] for i in indices]
            else:
                # Normal uniform sampling
                step = len(valid_frames) / num_action_frames
                indices = [int(i * step) for i in range(num_action_frames)]
                selected_action_frames = [valid_frames[i] for i in indices]
        
        # Sample remaining frames from all frames
        remaining_frames = num_frames - len(selected_action_frames)
        other_frames = [f for f in all_frames if f not in selected_action_frames]
        if remaining_frames > 0:
            if len(other_frames) >= remaining_frames:
                selected_other_frames = random.sample(other_frames, remaining_frames)
            else:
                try:
                    selected_other_frames = random.choices(other_frames, k=remaining_frames)
                except:
                    print(f"Error sampling other frames for video")
                    selected_other_frames = []
        else:
            selected_other_frames = []
        
        # Merge all selected frames and sort by time order
        selected_frames = sorted(selected_action_frames + selected_other_frames)
        return selected_frames
    
    def get_consecutive_frames(self, center_frame, num_frames, max_frame):
        """Get consecutive frame sequence centered at center_frame
        Args:
            center_frame: center frame index
            num_frames: number of frames needed
            max_frame: maximum frame index
        Returns:
            consecutive frame sequence
        """
        half = num_frames // 2
        start_idx = max(0, center_frame - half)
        end_idx = min(max_frame, start_idx + num_frames)
        start_idx = max(0, end_idx - num_frames)  # adjust start position to ensure enough frames
        return list(range(start_idx, end_idx))

    def __getitem__(self, idx):
        if self.train:
            if self.proposal_mask:
                return self.__gettrainitem__(idx, proposal_mask=True)
            else:
                return self.__gettrainitem__(idx)
        else:
            return self.__gettestitem__(idx)

    def __gettrainitem__(self, idx, proposal_mask=False):
        # Randomly select N action categories
        if len(self.selected_categories) < self.num_ways:
            raise ValueError(f"Not enough categories for {self.num_ways}-way setting")

        all_support_frames = []
        all_support_masks = []
        all_query_frames = []
        all_query_masks = []
        if random.random() < 0.5:
            selected_categories = random.sample(self.selected_categories, self.num_ways + 1)
            weights = [1 for _ in range(self.num_ways)]
            if self.num_ways == 5:
                weights.append(8)
            elif self.num_ways == 2:
                weights.append(5)
            else:
                raise ValueError(f"Not supported num_ways: {self.num_ways}")
            query_category = random.choices(selected_categories, weights=weights, k=1)[0]
            selected_categories = selected_categories[:self.num_ways]

            # Get all videos for this category
            category_videos = self.category_to_videos[query_category]
            
            # Ensure enough videos for support and query
            if len(category_videos) < self.num_shots + 1:
                query_video_id = random.choice(category_videos)
            else:
                query_video_id = random.choice(category_videos)
        else:
            selected_categories = random.sample(self.selected_categories, self.num_ways)
        
            # Find videos that contain at least one of the selected categories
            eligible_query_videos = []
            for vid in self.video_ids:
                if any(cat in self.video_to_categories[vid] for cat in selected_categories):
                    eligible_query_videos.append(vid)
            
            if len(eligible_query_videos) < 1:
                raise ValueError("Not enough eligible query videos")
            
            # First get the query frames from one random category
            query_category = random.choice(selected_categories)
            # Get all videos for this category
            category_videos = self.category_to_videos[query_category]
            
            # Ensure enough videos for support and query
            if len(category_videos) < self.num_shots + 1:
                # If not enough videos, allow reuse
                query_video_id = random.choice(category_videos)
            else:
                query_video_id = random.choice(category_videos)


        valid_query_frames, all_query_frames_list = self.get_valid_frames(query_video_id, query_category)
        
        # Ensure at least 1 action frame for query frames
        query_indices = self.sample_frames_with_action_support(
            valid_query_frames,
            all_query_frames_list,
            self.query_frames,
            min_action_frames=self.query_frames - 1
        )
        
        if self.proposal_mask:
            query_frames, query_masks, proposal_masks = self.get_frames(query_video_id, query_indices, query_category, proposal_mask=self.proposal_mask)
        else:
            query_frames, query_masks = self.get_frames(query_video_id, query_indices, query_category)
        
        # For each category, get support data and corresponding query masks
        all_support_object_categories = []
        all_support_motion_categories = []
        for category in selected_categories:
            # Get support videos (excluding query video)
            available_support_videos = [v for v in self.category_to_videos[category] 
                                     if self.action_segments[v]['length'] >= self.support_frames]
            
            if len(available_support_videos) < self.num_shots:
                support_video_ids = random.choices(available_support_videos, k=self.num_shots)
            else:
                support_video_ids = random.sample(available_support_videos, self.num_shots)
            
            # Get support data
            for support_video_id in support_video_ids:
                valid_support_frames, all_support_frames_list, object_categories, motion_categories = self.get_valid_frames(support_video_id, category, True)

                valid_num = len(valid_query_frames)
                
                # Ensure at least 2 action frames for support frames
                shot_indices = self.sample_frames_with_action_support(
                    valid_support_frames,
                    all_support_frames_list,
                    self.support_frames,
                    min_action_frames=min(valid_num, self.support_frames - 1) if self.support_frames > 1 else 1
                )
                
                frames, masks = self.get_frames(support_video_id, shot_indices, category)
                all_support_frames.extend(frames)
                all_support_masks.extend(masks)
                all_support_object_categories.append(object_categories)
                all_support_motion_categories.append(motion_categories)
            
            # Get query masks for this category using the same frames
            if category in self.video_to_categories[query_video_id]:
                _, category_masks = self.get_frames(query_video_id, query_indices, category)
                all_query_masks.extend(category_masks)
            else:
                # If query video doesn't contain this category, use zero masks
                h, w = query_masks[0].shape[:2]
                zero_masks = [np.zeros((h, w), dtype=np.bool_) for _ in range(self.query_frames)]
                all_query_masks.extend(zero_masks)
            
            # Duplicate query frames for each category
            all_query_frames.extend(query_frames)

        # Apply transforms
        if self.transforms is not None:
            try:
                if self.proposal_mask:
                    all_query_masks.extend(proposal_masks * self.num_ways)
                    query_frames_, query_masks_ = self.transforms(all_query_frames * 2, all_query_masks)
                    support_frames, support_masks = self.transforms(all_support_frames, all_support_masks)
                    query_frames = query_frames_[:len(all_query_frames)]
                    query_masks = query_masks_[:len(all_query_frames)]
                    proposal_masks = query_masks_[len(all_query_frames):]
                else:
                    query_frames, query_masks = self.transforms(all_query_frames, all_query_masks)
                    support_frames, support_masks = self.transforms(all_support_frames, all_support_masks)
            except Exception as e:
                print(f"Error applying transforms: {e}")
                print(f"Query frames shape: {len(all_query_frames)}")
                print(f"Query masks shape: {len(all_query_masks)}")
                print(f"Support frames shape: {len(all_support_frames)}")
                print(f"Support masks shape: {len(all_support_masks)}")
                raise e

        return query_frames, query_masks, support_frames, support_masks, [query_video_id], proposal_masks, all_support_object_categories, all_support_motion_categories

    def __gettestitem__(self, idx):
        if False: # [FIXED BY TEACHER]
            selected_categories = self.action_hierarchy.sample_fine_grained_episode(self.num_ways)
            query_category = random.choice(selected_categories)
        else:
            selected_categories = random.sample(self.selected_categories, self.num_ways + 1)
            query_category = random.choice(selected_categories)
        selected_categories = selected_categories[:self.num_ways]
        
        # Prepare support set and query set data
        all_support_frames = []
        all_support_masks = []
        all_query_frames = []
        all_query_masks = []
        
        # Get all videos for this category
        category_videos = self.category_to_videos[query_category]
        
        # Ensure enough videos for support and query
        if len(category_videos) < self.num_shots + 1:
            query_video_id = random.choice(category_videos)
        else:
            query_video_id = random.choice(category_videos)
        
        # Get all frames from query video
        _, all_query_frames_list = self.get_valid_frames(query_video_id, query_category)
        # Use all frames for testing
        query_indices = all_query_frames_list
        
        query_frames, query_masks = self.get_frames(query_video_id, query_indices, query_category)
        
        # Get support set data for each category
        for category in selected_categories:
            # Get all videos for this category
            category_videos = self.category_to_videos[category]
            
            # Get support set videos
            if category == query_category:
                support_video_ids = random.sample([v for v in category_videos ], self.num_shots)
            else:
                if len(category_videos) < self.num_shots:
                    support_video_ids = random.choices(category_videos, k=self.num_shots)
                else:
                    support_video_ids = random.sample(category_videos, self.num_shots)
            
            # Get frames from support set videos
            for support_video_id in support_video_ids:
                valid_support_frames, all_support_frames_list = self.get_valid_frames(support_video_id, category)
                shot_indices = self.sample_frames_with_action(
                    valid_support_frames,
                    all_support_frames_list,
                    self.support_frames,
                    min_action_frames=2
                )
                
                frames, masks = self.get_frames(support_video_id, shot_indices, category)
                all_support_frames.extend(frames)
                all_support_masks.extend(masks)
            
            # Get category masks for query frames
            if category in self.video_to_categories[query_video_id]:
                _, category_masks = self.get_frames(query_video_id, query_indices, category)
                all_query_masks.extend(category_masks)
            else:
                # If query video doesn't contain this category, use zero masks
                h, w = query_masks[0].shape[:2]
                zero_masks = [np.zeros((h, w), dtype=np.bool_) for _ in range(len(query_indices))]
                all_query_masks.extend(zero_masks)
            
            # Duplicate query frames for each category
            all_query_frames.extend(query_frames)
        
        # Apply data augmentation
        if self.transforms is not None:
            try:
                query_frames, query_masks = self.transforms(all_query_frames, all_query_masks)
                support_frames, support_masks = self.transforms(all_support_frames, all_support_masks)
            except Exception as e:
                print(f"Error applying transforms: {e}")
                raise e
        
        return query_frames, query_masks, support_frames, support_masks, [query_video_id], selected_categories

    def __len__(self):
        # Return a sufficiently large number to allow enough episodes
        return 1000000  # This number can be adjusted as needed
