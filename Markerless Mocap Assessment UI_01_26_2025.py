import numpy as np
import pandas as pd
import json
import logging
import tkinter as tk
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class DataManager:
    """
    Manages motion capture data loading, validation, and processing with a focus on 2D views
    while maintaining 3D data structure for future expansion.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Configure logging
        self._setup_logging()
        
        # Define view-specific requirements
        self.required_fields = {
            'sagittal': {
                'angles2D': ['spine_angle', 'shin_angle', 'knee_angle', 'hip_angle'],
                'keypoints2D': ['nose', 'shoulder', 'hip', 'knee', 'ankle']
            },
            'anterior': {
                'angles2D': ['leftKneeAngle', 'rightKneeAngle', 'leftHipAngle', 'rightHipAngle'],
                'keypoints2D': ['leftHip', 'rightHip', 'leftKnee', 'rightKnee']
            }
        }
        
        # Initialize data storage
        self.motion_data = None
        self.metadata = {}
        self.validation_errors = []

    def _setup_logging(self):
        """Configure logging settings"""
        fh = logging.FileHandler('movement_analysis.log')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    def load_data(self, filepath: Union[str, Path], view_type: str) -> Tuple[bool, Optional[Dict]]:
        """
        Load and validate motion capture data for a specific view.
        
        Args:
            filepath: Path to the data file
            view_type: Type of view ('sagittal' or 'anterior')
            
        Returns:
            Tuple containing:
                - Success flag (bool)
                - Validated data (Dict) or None if validation fails
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            # Load data based on file type
            data = self._load_file(filepath)
            
            # Validate for specific view requirements
            if self._validate_view_data(data, view_type):
                self.motion_data = self._standardize_data_format(data)
                self.logger.info(f"Successfully loaded {view_type} view data from {filepath}")
                return True, self.motion_data
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            self.validation_errors.append(str(e))
            return False, None

    def _load_file(self, filepath: Path) -> Dict:
        """Load data file based on extension"""
        if filepath.suffix == '.json':
            return self._load_json(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def _validate_view_data(self, data: Dict, view_type: str) -> bool:
        """
        Validate data contains required fields for specified view.
        
        Args:
            data: Motion capture data
            view_type: View type to validate against
            
        Returns:
            bool indicating if data meets requirements
        """
        try:
            if view_type not in self.required_fields:
                raise ValueError(f"Unsupported view type: {view_type}")
                
            required = self.required_fields[view_type]
            first_frame = data[0] if isinstance(data, list) else data
            
            # Validate angles
            if not self._validate_angles(first_frame, required['angles2D']):
                return False
                
            # Validate keypoints
            if not self._validate_keypoints(first_frame, required['keypoints2D']):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            self.validation_errors.append(str(e))
            return False


    def get_sagittal_angles(self, frame_idx: int) -> Optional[Dict[str, float]]:
        """Get sagittal plane angles for specified frame"""
        try:
            if not self.motion_data or frame_idx >= len(self.motion_data):
                return None
                
            frame = self.motion_data[frame_idx]
            return {
                'spine': frame['angles2D']['spine_angle'],
                'shin': frame['angles2D']['shin_angle'],
                'knee': frame['angles2D']['knee_angle'],
                'hip': frame['angles2D']['hip_angle']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting sagittal angles: {str(e)}")
            return None

    def get_anterior_angles(self, frame_idx: int) -> Optional[Dict[str, float]]:
        """Get anterior view angles for specified frame"""
        try:
            if not self.motion_data or frame_idx >= len(self.motion_data):
                return None
                
            frame = self.motion_data[frame_idx]
            return {
                'left_knee': frame['angles2D']['leftKneeAngle'],
                'right_knee': frame['angles2D']['rightKneeAngle'],
                'left_hip': frame['angles2D']['leftHipAngle'],
                'right_hip': frame['angles2D']['rightHipAngle']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting anterior angles: {str(e)}")
            return None
    
    def _load_json(self, filepath: Path) -> Dict:
        """Load and parse JSON motion capture data"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Handle single frame vs multiple frames
            if isinstance(data, dict):
                data = [data]  # Convert single frame to list
                
            return self._standardize_data_format(data)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error loading JSON: {str(e)}")

    def _load_excel(self, filepath: Path) -> Dict:
        """Load and parse Excel motion capture data"""
        try:
            df = pd.read_excel(filepath)
            return self._convert_dataframe_to_standard_format(df)
        except Exception as e:
            raise RuntimeError(f"Error loading Excel file: {str(e)}")

    def _load_csv(self, filepath: Path) -> Dict:
        """Load and parse CSV motion capture data"""
        try:
            df = pd.read_csv(filepath)
            return self._convert_dataframe_to_standard_format(df)
        except Exception as e:
            raise RuntimeError(f"Error loading CSV file: {str(e)}")

    def _standardize_data_format(self, data: List[Dict]) -> Dict:
        """Convert data to standard internal format"""
        try:
            standardized = {
                'frames': [],
                'metadata': {
                    'frame_count': len(data),
                    'keypoint_count': len(data[0].get('keypoints2D', [])),
                    'timestamp_start': data[0].get('timestamp', 0),
                    'timestamp_end': data[-1].get('timestamp', 0)
                }
            }
            
            for frame in data:
                processed_frame = {
                    'keypoints': self._process_keypoints(frame.get('keypoints2D', [])),
                    'keypoints3D': self._process_keypoints(frame.get('keypoints3D', [])),
                    'angles': self._process_angles(frame),
                    'com': {
                        'x': frame.get('com2D', {}).get('x', 0),
                        'y': frame.get('com2D', {}).get('y', 0),
                        'realX': frame.get('com2D', {}).get('realX', 0),
                        'realY': frame.get('com2D', {}).get('realY', 0)
                    },
                    'timestamp': frame.get('timestamp', 0)
                }
                standardized['frames'].append(processed_frame)
                
            return standardized
            
        except Exception as e:
            raise RuntimeError(f"Error standardizing data format: {str(e)}")

    def _process_keypoints(self, keypoints):
        """Convert keypoint array into dictionary by name"""
        keypoint_dict = {}
        for kp in keypoints:
            name = kp['name']
            keypoint_dict[name] = {
                'x': kp['x'],
                'y': kp['y'],
                'z': kp['z'],
                'score': kp['score']
            }
            if 'realX' in kp:
                keypoint_dict[name]['realX'] = kp['realX']
                keypoint_dict[name]['realY'] = kp['realY']
        return keypoint_dict

    def _process_angles(self, frame: Dict) -> Dict[str, float]:
        """Process joint angle data into standard format"""
        try:
            processed_angles = {}
            
            # Process 2D angles
            if 'angles2D' in frame:
                for angle_name, value in frame['angles2D'].items():
                    processed_angles[f"{angle_name}_2D"] = float(value)
                    
            # Process 3D angles 
            if 'angles3D' in frame:
                for angle_name, value in frame['angles3D'].items():
                    processed_angles[f"{angle_name}_3D"] = float(value)
                    
            return processed_angles
            
        except Exception as e:
            raise RuntimeError(f"Error processing angles: {str(e)}")

    def _convert_dataframe_to_standard_format(self, df: pd.DataFrame) -> Dict:
        """Convert DataFrame to standard internal format"""
        try:
            standardized = {
                'frames': [],
                'metadata': {
                    'frame_count': len(df),
                    'timestamp_start': df['timestamp'].iloc[0] if 'timestamp' in df else 0,
                    'timestamp_end': df['timestamp'].iloc[-1] if 'timestamp' in df else 0
                }
            }
            
            for _, row in df.iterrows():
                frame = {
                    'keypoints': self._extract_keypoints_from_row(row),
                    'angles': self._extract_angles_from_row(row),
                    'com': self._extract_com_from_row(row),
                    'timestamp': row.get('timestamp', 0)
                }
                standardized['frames'].append(frame)
                
            return standardized
            
        except Exception as e:
            raise RuntimeError(f"Error converting DataFrame: {str(e)}")

    def _extract_keypoints_from_row(self, row: pd.Series) -> Dict[str, Dict]:
        """Extract keypoint data from DataFrame row"""
        keypoints = {}
        for kp_name in self.required_fields['keypoints']:
            if all(f"{kp_name}_{coord}" in row for coord in ['x', 'y', 'z']):
                keypoints[kp_name] = {
                    'x': row[f"{kp_name}_x"],
                    'y': row[f"{kp_name}_y"],
                    'z': row[f"{kp_name}_z"],
                    'confidence': row.get(f"{kp_name}_confidence", 1.0)
                }
        return keypoints

    def _extract_angles_from_row(self, row: pd.Series) -> Dict[str, float]:
        """Extract joint angle data from DataFrame row"""
        angles = {}
        for angle_name in self.required_fields['angles']:
            angles[angle_name] = row.get(angle_name, 0.0)
        return angles

    def _extract_com_from_row(self, row: pd.Series) -> Dict[str, float]:
        """Extract center of mass data from DataFrame row"""
        com = {}
        for coord in self.required_fields['com']:
            com[coord] = row.get(f"com_{coord}", 0.0)
        return com


    def _validate_angles_data(self, frame_data):
        """Validate angles data structure and content"""
        try:
            # Check if angles2D exists
            if 'angles2D' not in frame_data:
                self.logger.error("Missing angles2D dictionary in frame data")
                return False, "Missing angles2D dictionary"
                
            angles2d = frame_data['angles2D']
            
            # Check that angles2D is a dictionary
            if not isinstance(angles2d, dict):
                self.logger.error(f"angles2D must be a dictionary, got {type(angles2d)}")
                return False, f"angles2D must be a dictionary"
                
            # Get required angles from configuration
            required_angles = self.required_fields['angles2D']['required_attributes']
            
            # Check for missing angles with detailed logging
            available_angles = set(angles2d.keys())
            missing_angles = [angle for angle in required_angles if angle not in available_angles]
            
            if missing_angles:
                self.logger.error(f"Missing angles: {missing_angles}")
                self.logger.error(f"Available angles: {available_angles}")
                return False, f"Missing required angles: {missing_angles}"
                
            # Validate angle values
            invalid_angles = []
            for angle_name, value in angles2d.items():
                if not isinstance(value, (int, float)):
                    invalid_angles.append(f"{angle_name}: {type(value)}")
                elif value < -360 or value > 360:
                    invalid_angles.append(f"{angle_name}: {value}")
                    
            if invalid_angles:
                self.logger.error(f"Invalid angle values: {invalid_angles}")
                return False, f"Invalid angle values detected"
                
            return True, None
            
        except Exception as e:
            self.logger.error(f"Error in angle validation: {str(e)}")
            return False, f"Angle validation error: {str(e)}"
    
    def _validate_data_structure(self, data: Dict) -> bool:
        """Validate motion capture data structure based on expected JSON format"""
        try:
            if not isinstance(data, list):
                self.validation_errors.append("Data must be an array of frames")
                return False
                
            # Validate first frame structure
            frame = data[0]
            
            # Validate each required section
            for section, requirements in self.required_fields.items():
                if section not in frame:
                    self.validation_errors.append(f"Missing required section: {section}")
                    return False
                
                # Validate array lengths where specified
                if 'count' in requirements:
                    if section == 'keypoints2D' or section == 'keypoints3D':
                        if not isinstance(frame[section], list) or len(frame[section]) != requirements['count']:
                            self.validation_errors.append(
                                f"{section} must be an array of {requirements['count']} items"
                            )
                            return False
                
                # Validate required attributes
                if section == 'keypoints2D' or section == 'keypoints3D':
                    # Validate keypoint structure
                    for keypoint in frame[section]:
                        missing_attrs = [
                            attr for attr in requirements['required_attributes'] 
                            if attr not in keypoint
                        ]
                        if missing_attrs:
                            self.validation_errors.append(
                                f"Missing required attributes in {section}: {missing_attrs}"
                            )
                            return False
                else:
                    # Validate other sections
                    section_data = frame[section]
                    missing_attrs = [
                        attr for attr in requirements['required_attributes'] 
                        if attr not in section_data
                    ]
                    if missing_attrs:
                        self.validation_errors.append(
                            f"Missing required attributes in {section}: {missing_attrs}"
                        )
                        return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Data validation error: {str(e)}")
            return False

    def _validate_keypoints(self, keypoints: Dict) -> bool:
        """Validate keypoint data"""
        missing = [kp for kp in self.required_fields['keypoints'] if kp not in keypoints]
        if missing:
            self.validation_errors.append(f"Missing required keypoints: {missing}")
            return False
        return True

    def _validate_angles(self, angles: Dict) -> bool:
        """Validate joint angle data"""
        missing = [angle for angle in self.required_fields['angles'] if angle not in angles]
        if missing:
            self.validation_errors.append(f"Missing required angles: {missing}")
            return False
        return True

    def _validate_com(self, com: Dict) -> bool:
        """Validate center of mass data"""
        missing = [coord for coord in self.required_fields['com'] if coord not in com]
        if missing:
            self.validation_errors.append(f"Missing required COM coordinates: {missing}")
            return False
        return True

    def get_frame_data(self, frame_idx: int) -> Optional[Dict]:
        """Get data for a specific frame"""
        try:
            if not self.motion_data or frame_idx >= len(self.motion_data['frames']):
                return None
            return self.motion_data['frames'][frame_idx]
        except Exception as e:
            self.logger.error(f"Error retrieving frame {frame_idx}: {str(e)}")
            return None

    def get_keypoint_trajectory(self, keypoint_name: str) -> Optional[np.ndarray]:
        """Get trajectory for specific keypoint"""
        try:
            if not self.motion_data:
                return None
                
            trajectory = []
            for frame in self.motion_data['frames']:
                if keypoint_name in frame['keypoints']:
                    kp = frame['keypoints'][keypoint_name]
                    trajectory.append([kp['x'], kp['y'], kp['z']])
                    
            return np.array(trajectory)
            
        except Exception as e:
            self.logger.error(f"Error getting keypoint trajectory: {str(e)}")
            return None

    def calculate_spine_to_shin_angle(self, frame):
        """Analyze spine-to-shin alignment using 2D sagittal angles"""
        try:
            # Get angles directly from 2D angle data
            spine_angle = frame['angles2D']['spine_angle']
            shin_angle = frame['angles2D']['shin_angle']
            
            # Calculate parallel deviation 
            parallel_deviation = abs(spine_angle - shin_angle)

            # Calculate individual angles relative to vertical for visualization
            vertical_ref = 90.0  # Vertical reference is 90 degrees
            spine_vertical = abs(spine_angle - vertical_ref)
            shin_vertical = abs(shin_angle - vertical_ref)

            return {
                'spine_angle': spine_angle,
                'shin_angle': shin_angle,
                'parallel_deviation': parallel_deviation,
                'spine_vertical': spine_vertical,
                'shin_vertical': shin_vertical
            }

        except Exception as e:
            self.logger.error(f"Error calculating spine-to-shin angles: {str(e)}")
            # Add more detailed error information
            self.logger.error(f"Frame angles2D structure: {frame.get('angles2D', 'Not found')}")
            return None

    def calculate_spine_shin_alignment(self, frame_idx: int) -> Optional[Dict[str, float]]:
        """
        Calculate spine-to-shin alignment metrics for sagittal view.
        
        Args:
            frame_idx: Frame index to analyze
            
        Returns:
            Dict containing alignment metrics or None if calculation fails
        """
        try:
            angles = self.get_sagittal_angles(frame_idx)
            if not angles:
                return None
                
            return {
                'spine_angle': angles['spine'],
                'shin_angle': angles['shin'],
                'parallel_deviation': abs(angles['spine'] - angles['shin']),
                'knee_angle': angles['knee']
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating alignment: {str(e)}")
            return None

    def analyze_squat_mechanics(self, frame_data):
        """
        Analyze squat mechanics throughout the movement sequence.
        Identifies potential issues with form and alignment.
        """
        try:
            angles = self.calculate_spine_to_shin_angle(frame_data)
            if not angles:
                return None

            # Define thresholds for analysis
            PARALLEL_THRESHOLD = 15.0  # Maximum acceptable deviation from parallel
            FORWARD_LEAN_THRESHOLD = 45.0  # Maximum acceptable trunk angle
            SHIN_ANGLE_MIN = 25.0  # Minimum acceptable shin angle
            SHIN_ANGLE_MAX = 65.0  # Maximum acceptable shin angle

            # Analyze mechanics
            mechanics_issues = []

            if angles['parallel_deviation'] > PARALLEL_THRESHOLD:
                mechanics_issues.append("Loss of spine-shin parallel alignment")

            if angles['trunk_angle'] > FORWARD_LEAN_THRESHOLD:
                mechanics_issues.append("Excessive forward trunk lean")

            if angles['shin_angle'] < SHIN_ANGLE_MIN:
                mechanics_issues.append("Shin angle too vertical")
            elif angles['shin_angle'] > SHIN_ANGLE_MAX:
                mechanics_issues.append("Excessive shin angle")

            return {
                'angles': angles,
                'issues': mechanics_issues,
                'quality_score': self.calculate_movement_quality(angles)
            }

        except Exception as e:
            self.logger.error(f"Error analyzing squat mechanics: {str(e)}")
            return None

    def calculate_movement_quality(self):
        """Calculate overall movement quality score"""
        try:
            # Component scores (0-100 scale)
            alignment_score = max(0, 100 - (
                self.results['angles']['max_deviation'] * 2
            ))
            
            symmetry_score = max(0, 100 - (
                self.results['symmetry']['mean_asymmetry'] * 2
            ))
            
            stability_score = max(0, 100 - (
                self.results['com']['max_deviation'] / self.params['com_deviation'] * 100
            ))
            
            # Weighted total score
            weights = {
                'alignment': 0.4,
                'symmetry': 0.4,
                'stability': 0.2
            }
            
            total_score = (
                alignment_score * weights['alignment'] +
                symmetry_score * weights['symmetry'] +
                stability_score * weights['stability']
            )
            
            self.results['quality'].update({
                'alignment_score': alignment_score,
                'symmetry_score': symmetry_score,
                'stability_score': stability_score,
                'total_score': total_score
            })
            
        except Exception as e:
            self.logger.error(f"Error calculating movement quality: {str(e)}")
            raise

    def get_angle_data(self, angle_name: str) -> Optional[np.ndarray]:
        """Get time series data for specific joint angle"""
        try:
            if not self.motion_data:
                return None
                
            return np.array([frame['angles'][angle_name] 
                           for frame in self.motion_data['frames']
                           if angle_name in frame['angles']])
                           
        except Exception as e:
            self.logger.error(f"Error getting angle data: {str(e)}")
            return None

    def get_com_trajectory(self) -> Optional[np.ndarray]:
        """Get center of mass trajectory"""
        try:
            if not self.motion_data:
                return None
                
            trajectory = []
            for frame in self.motion_data['frames']:
                com = frame['com']
                trajectory.append([com['x'], com['y'], com['z']])
                
            return np.array(trajectory)
            
        except Exception as e:
            self.logger.error(f"Error getting COM trajectory: {str(e)}")
            return None

    def calculate_geometric_entropy(self, trajectory: np.ndarray) -> Optional[float]:
        """Calculate geometric entropy of movement trajectory"""
        try:
            if len(trajectory) < 3:
                raise ValueError("Insufficient points for entropy calculation")
                
            # Calculate path length
            path_length = np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1)))
            
            # Calculate convex hull perimeter
            hull = ConvexHull(trajectory[:, :2])  # Use only x,y coordinates
            hull_perimeter = 0
            for simplex in hull.simplices:
                pt1 = hull.points[simplex[0]]
                pt2 = hull.points[simplex[1]]
                hull_perimeter += np.sqrt(np.sum((pt2 - pt1)**2))
                
            # Calculate entropy
            entropy = (np.log(2 * path_length) - np.log(hull_perimeter)) / np.log(2)
            return entropy
            
        except Exception as e:
            self.logger.error(f"Error calculating geometric entropy: {str(e)}")
            return None

    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors"""
        return self.validation_errors.copy()

    def clear_validation_errors(self):
        """Clear validation error list"""
        self.validation_errors = []

    def validate_motion_data(self, filename):
        """Validate motion capture data file"""
        try:
            self.logger.info(f"Validating motion data from: {filename}")
            
            # Load and parse data
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                self.logger.error("Data must be an array of frames")
                return False, None
                
            # Print data structure analysis
            self.print_data_structure(data)
            
            # Validate first frame
            first_frame = data[0]
            
            # Check required sections
            for section in ['keypoints2D', 'keypoints3D', 'com2D', 'angles2D', 'angles3D']:
                if section not in first_frame:
                    self.logger.error(f"Missing required section: {section}")
                    return False, None
                    
            # Validate angles specifically
            angles_valid, angles_error = self._validate_angles_data(first_frame)
            if not angles_valid:
                return False, None
                
            self.motion_data = data
            self.logger.info("Motion data validation successful")
            return True, data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format: {str(e)}")
            return False, None
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False, None
        
    def inspect_file_structure(self, filename):
        """
        Analyze and report the structure of an uploaded motion data file
        
        Args:
            filename: Path to the data file
            
        Returns:
            Dict containing file structure information
        """
        try:
            structure = {
                'file_type': None,
                'columns': [],
                'sample_data': None,
                'hierarchy': {},
                'validation_issues': []
            }
            
            # Determine file type
            if filename.endswith('.json'):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                structure['file_type'] = 'JSON'
                
                # Analyze JSON structure
                if isinstance(data, list):
                    structure['hierarchy'] = {
                        'type': 'array',
                        'length': len(data),
                        'sample_keys': list(data[0].keys()) if data else []
                    }
                    # Get nested structure from first frame
                    if data:
                        for key, value in data[0].items():
                            if isinstance(value, dict):
                                structure['hierarchy'][key] = {
                                    'type': 'object',
                                    'keys': list(value.keys())
                                }
                            elif isinstance(value, list):
                                structure['hierarchy'][key] = {
                                    'type': 'array',
                                    'length': len(value),
                                    'sample': value[0] if value else None
                                }
                    
                elif isinstance(data, dict):
                    structure['hierarchy'] = {
                        'type': 'object',
                        'keys': list(data.keys())
                    }
                
                structure['sample_data'] = data[0] if isinstance(data, list) and data else data
                
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filename)
                structure['file_type'] = 'Excel'
                structure['columns'] = list(df.columns)
                structure['sample_data'] = df.iloc[0].to_dict() if not df.empty else None
                
            elif filename.endswith('.csv'):
                df = pd.read_csv(filename)
                structure['file_type'] = 'CSV'
                structure['columns'] = list(df.columns)
                structure['sample_data'] = df.iloc[0].to_dict() if not df.empty else None
                
            # Check for required fields
            missing_fields = []
            if structure['file_type'] in ['Excel', 'CSV']:
                required_prefixes = ['keypoints', 'com', 'angle']
                found_fields = [col for col in structure['columns'] 
                              if any(col.startswith(prefix) for prefix in required_prefixes)]
                if not found_fields:
                    missing_fields.append("No keypoint, COM or angle data columns found")
                    
            elif structure['file_type'] == 'JSON':
                if not structure.get('hierarchy', {}).get('sample_keys'):
                    missing_fields.append("No data fields found in JSON structure")
                else:
                    required_keys = ['keypoints2D', 'keypoints3D', 'com2D']
                    missing = [key for key in required_keys 
                              if key not in structure['hierarchy']['sample_keys']]
                    if missing:
                        missing_fields.append(f"Missing required keys: {', '.join(missing)}")
                        
            structure['validation_issues'] = missing_fields
            
            return structure
            
        except Exception as e:
            return {
                'file_type': 'Unknown',
                'validation_issues': [f"Error analyzing file structure: {str(e)}"]
            }

    def print_structure_report(self, structure):
        """Generate a formatted report of the file structure"""
        report = []
        report.append(f"File Type: {structure['file_type']}")
        report.append("\nStructure Overview:")
        
        if structure['file_type'] in ['Excel', 'CSV']:
            report.append("Columns Found:")
            for col in structure['columns']:
                report.append(f"  - {col}")
                
        elif structure['file_type'] == 'JSON':
            report.append("Data Hierarchy:")
            for key, value in structure['hierarchy'].items():
                if key == 'type':
                    report.append(f"  Root Type: {value}")
                elif key == 'sample_keys':
                    report.append("  Top-level Keys:")
                    for k in value:
                        report.append(f"    - {k}")
                elif isinstance(value, dict):
                    report.append(f"  {key}:")
                    for k, v in value.items():
                        report.append(f"    {k}: {v}")
                        
        if structure['validation_issues']:
            report.append("\nValidation Issues:")
            for issue in structure['validation_issues']:
                report.append(f"  - {issue}")
                
        return "\n".join(report)

    def inspect_file_structure(self, filename):
        """
        Analyze and report the structure of an uploaded motion data file
        
        Args:
            filename: Path to the data file
            
        Returns:
            Dict containing file structure information and prints detailed report to console
        """
        try:
            structure = {
                'file_type': None,
                'columns': [],
                'sample_data': None,
                'hierarchy': {},
                'validation_issues': []
            }
            
            print("\n=== Motion Data File Structure Analysis ===")
            print(f"Analyzing file: {filename}")
            
            # Determine and analyze file type
            if filename.endswith('.json'):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                structure['file_type'] = 'JSON'
                print("\nFile Type: JSON")
                
                # Analyze JSON structure
                if isinstance(data, list):
                    print(f"\nData Format: Array of {len(data)} frames")
                    structure['hierarchy'] = {
                        'type': 'array',
                        'length': len(data),
                        'sample_keys': list(data[0].keys()) if data else []
                    }
                    # Show structure of first frame
                    if data:
                        print("\nFrame Structure:")
                        self._print_json_structure(data[0], level=1)
                        
                elif isinstance(data, dict):
                    print("\nData Format: Single frame object")
                    structure['hierarchy'] = {
                        'type': 'object',
                        'keys': list(data.keys())
                    }
                    print("\nStructure:")
                    self._print_json_structure(data, level=1)
                
                structure['sample_data'] = data[0] if isinstance(data, list) and data else data
                
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filename)
                structure['file_type'] = 'Excel'
                print("\nFile Type: Excel")
                self._print_dataframe_structure(df)
                structure['columns'] = list(df.columns)
                structure['sample_data'] = df.iloc[0].to_dict() if not df.empty else None
                
            elif filename.endswith('.csv'):
                df = pd.read_csv(filename)
                structure['file_type'] = 'CSV'
                print("\nFile Type: CSV")
                self._print_dataframe_structure(df)
                structure['columns'] = list(df.columns)
                structure['sample_data'] = df.iloc[0].to_dict() if not df.empty else None
            
            # Validate against required fields
            missing_fields = []
            if structure['file_type'] in ['Excel', 'CSV']:
                required_prefixes = ['keypoints', 'com', 'angle']
                found_fields = [col for col in structure['columns'] 
                              if any(col.startswith(prefix) for prefix in required_prefixes)]
                if not found_fields:
                    missing_fields.append("No keypoint, COM or angle data columns found")
                    
            elif structure['file_type'] == 'JSON':
                if not structure.get('hierarchy', {}).get('sample_keys'):
                    missing_fields.append("No data fields found in JSON structure")
                else:
                    required_keys = ['keypoints2D', 'keypoints3D', 'com2D']
                    missing = [key for key in required_keys 
                              if key not in structure['hierarchy']['sample_keys']]
                    if missing:
                        missing_fields.append(f"Missing required keys: {', '.join(missing)}")
            
            structure['validation_issues'] = missing_fields
            
            # Print validation results
            print("\n=== Validation Results ===")
            if missing_fields:
                print("\nValidation Issues Found:")
                for issue in missing_fields:
                    print(f"- {issue}")
            else:
                print("All required fields present")
                
            print("\n=== End of Structure Analysis ===\n")
            
            return structure
            
        except Exception as e:
            print(f"\nError analyzing file structure: {str(e)}")
            return {
                'file_type': 'Unknown',
                'validation_issues': [f"Error analyzing file structure: {str(e)}"]
            }

    def _print_json_structure(self, data, level=0):
        """Print formatted JSON structure with types and sample values"""
        indent = "  " * level
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    print(f"{indent}{key}: {type(value).__name__}")
                    self._print_json_structure(value, level + 1)
                else:
                    print(f"{indent}{key}: {type(value).__name__} = {str(value)[:50]}")
        elif isinstance(data, list) and data:
            print(f"{indent}Array of {len(data)} items:")
            if data:
                self._print_json_structure(data[0], level + 1)

    def _print_dataframe_structure(self, df):
        """Print detailed DataFrame structure analysis"""
        print(f"\nDataFrame Structure ({len(df)} rows):")
        print("\nColumns:")
        
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].count()
            null_count = df[col].isnull().sum()
            sample = str(df[col].iloc[0])[:50] if len(df) > 0 else "No data"
            
            print(f"\n- {col}")
            print(f"  Type: {dtype}")
            print(f"  Non-null count: {non_null}")
            print(f"  Null count: {null_count}")
            print(f"  Sample value: {sample}")

    def print_data_structure(self, data):
        """Print detailed analysis of data structure"""
        try:
            print("\n=== Motion Data Structure Analysis ===")
            print(f"\nTotal Frames: {len(data)}")
            
            # Analyze first frame
            first_frame = data[0]
            print("\nFrame Structure:")
            
            # Print keypoints structure
            for keypoint_type in ['keypoints2D', 'keypoints3D']:
                if keypoint_type in first_frame:
                    keypoints = first_frame[keypoint_type]
                    print(f"\n{keypoint_type}:")
                    print(f"  Count: {len(keypoints)}")
                    if keypoints:
                        print("  Sample keypoint structure:")
                        for key, value in keypoints[0].items():
                            print(f"    {key}: {type(value).__name__} = {value}")
                            
            # Print angles structure
            for angle_type in ['angles2D', 'angles3D']:
                if angle_type in first_frame:
                    angles = first_frame[angle_type]
                    print(f"\n{angle_type}:")
                    for angle, value in angles.items():
                        print(f"  {angle}: {value}")
                        
            # Print COM structure
            if 'com2D' in first_frame:
                print("\ncom2D:")
                for key, value in first_frame['com2D'].items():
                    print(f"  {key}: {value}")
                    
            print("\n=== End of Structure Analysis ===\n")
            
        except Exception as e:
            print(f"Error analyzing data structure: {str(e)}")

    def calculate_bilateral_symmetry(self, frame_idx: int) -> Optional[Dict[str, float]]:
        """
        Calculate bilateral symmetry metrics for anterior view.
        
        Args:
            frame_idx: Frame index to analyze
            
        Returns:
            Dict containing symmetry metrics or None if calculation fails
        """
        try:
            angles = self.get_anterior_angles(frame_idx)
            if not angles:
                return None
                
            knee_diff = abs(angles['left_knee'] - angles['right_knee'])
            hip_diff = abs(angles['left_hip'] - angles['right_hip'])
            
            return {
                'knee_difference': knee_diff,
                'hip_difference': hip_diff,
                'symmetry_index': 100 * (1 - (knee_diff + hip_diff) / 360)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating symmetry: {str(e)}")
            return None
      
class AlignmentTool:
    """Tool for measuring postural alignment angles"""
    
    def __init__(self, canvas: tk.Canvas, view_type: str):
        self.canvas = canvas
        self.view_type = view_type  # 'anterior' or 'sagittal'
        self.points = []
        self.lines = []
        self.active = False
        self.current_line = None
        self.angle_text = None
        self.logger = logging.getLogger(__name__)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<Motion>', self.on_motion)
        self.canvas.bind('<Button-3>', self.on_right_click)

    def activate(self):
        """Activate the alignment tool"""
        self.active = True
        self.canvas.config(cursor="crosshair")

    def deactivate(self):
        """Deactivate the alignment tool"""
        self.active = False
        self.canvas.config(cursor="")
        self.clear_preview()

    def on_click(self, event):
        """Handle left mouse click for point placement"""
        if not self.active:
            return
            
        try:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            
            # Add point marker
            point_id = self.canvas.create_oval(
                x-3, y-3, x+3, y+3,
                fill="red",
                outline="white",
                tags="measurement"
            )
            
            self.points.append((x, y, point_id))
            
            if len(self.points) == 2:
                self.complete_measurement()
                
        except Exception as e:
            self.logger.error(f"Error placing point: {str(e)}")
            self.clear_preview()

    def on_motion(self, event):
        """Handle mouse motion for measurement preview"""
        if not self.active or not self.points:
            return
            
        try:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            
            self.update_preview(x, y)
            
        except Exception as e:
            self.logger.error(f"Error updating preview: {str(e)}")

    def on_right_click(self, event):
        """Handle right click to cancel current measurement"""
        if self.active:
            self.clear_preview()

    def update_preview(self, x: float, y: float):
        """Update measurement preview lines"""
        try:
            self.clear_preview()
            
            start_x, start_y, _ = self.points[0]
            
            # Draw reference line based on view type
            if self.view_type == "anterior":
                ref_line = self.canvas.create_line(
                    start_x, start_y,
                    x, start_y,
                    fill="lightblue",
                    dash=(4,4),
                    tags="preview"
                )
            else:  # sagittal
                ref_line = self.canvas.create_line(
                    x, y,
                    x, start_y,
                    fill="lightblue",
                    dash=(4,4),
                    tags="preview"
                )
                
            # Draw measurement line
            meas_line = self.canvas.create_line(
                start_x, start_y,
                x, y,
                fill="blue",
                dash=(4,4),
                tags="preview"
            )
            
            self.current_line = [ref_line, meas_line]
            
        except Exception as e:
            self.logger.error(f"Error updating preview: {str(e)}")

    def complete_measurement(self):
        """Complete the current measurement"""
        try:
            start_x, start_y, _ = self.points[0]
            end_x, end_y, _ = self.points[1]
            
            # Calculate angle
            angle = self.calculate_angle(start_x, start_y, end_x, end_y)
            
            # Draw permanent measurement lines
            if self.view_type == "anterior":
                self.canvas.create_line(
                    start_x, start_y,
                    end_x, start_y,
                    fill="lightblue",
                    tags="measurement"
                )
            else:
                self.canvas.create_line(
                    end_x, end_y,
                    end_x, start_y,
                    fill="lightblue",
                    tags="measurement"
                )
                
            line_id = self.canvas.create_line(
                start_x, start_y,
                end_x, end_y,
                fill="blue",
                width=2,
                tags="measurement"
            )
            
            # Draw angle arc
            self.draw_angle_marker(start_x, start_y, end_x, end_y, angle)
            
            self.lines.append(line_id)
            self.points = []
            self.clear_preview()
            
        except Exception as e:
            self.logger.error(f"Error completing measurement: {str(e)}")
            self.clear_preview()

    def calculate_angle(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate angle based on view type"""
        try:
            if self.view_type == "anterior":
                # Calculate angle from horizontal
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                return angle
            else:
                # Calculate angle from vertical
                angle = math.degrees(math.atan2(x2 - x1, -(y2 - y1)))
                return abs(angle)
                
        except Exception as e:
            self.logger.error(f"Error calculating angle: {str(e)}")
            return 0.0

    def draw_angle_marker(self, x1: float, y1: float, x2: float, y2: float, angle: float):
        """Draw angle arc and text"""
        try:
            radius = 30
            text_x = x1 + 50
            text_y = y1
            
            # Draw arc
            start_angle = 0 if self.view_type == "anterior" else 90
            arc_id = self.canvas.create_arc(
                x1 - radius, y1 - radius,
                x1 + radius, y1 + radius,
                start=start_angle,
                extent=angle,
                style="arc",
                outline="red",
                tags="measurement"
            )
            
            # Draw angle text
            text_id = self.canvas.create_text(
                text_x, text_y,
                text=f"{abs(angle):.1f}°",
                fill="red",
                font=("Arial", 12),
                tags="measurement"
            )
            
            self.lines.extend([arc_id, text_id])
            
        except Exception as e:
            self.logger.error(f"Error drawing angle marker: {str(e)}")

    def clear_preview(self):
        """Clear preview lines"""
        if self.current_line:
            for line_id in self.current_line:
                self.canvas.delete(line_id)
            self.current_line = None

    def clear_measurements(self):
        """Clear all measurements"""
        self.canvas.delete("measurement")
        self.canvas.delete("preview")
        self.points = []
        self.lines = []
        self.current_line = None

class StandingPostureTab:
    """Handles the Standing Posture assessment functionality"""
    def __init__(self, notebook, data_manager):
        self.tab = ttk.Frame(notebook, padding="10")
        self.data_manager = data_manager
        
        # Initialize state variables
        self.images = {"anterior": None, "sagittal": None}
        self.photo_images = {"anterior": None, "sagittal": None}
        self.canvases = {}
        self.alignment_tools = {}
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Create tab content
        self.create_tab_layout()
        self.create_views()
        self.create_controls()
        
        # Initialize posture metrics
        self.metrics = {
            'head_alignment': tk.StringVar(value="Head Alignment: --°"),
            'pelvic_tilt': tk.StringVar(value="Pelvic Tilt: --°"),
            'spine_angle': tk.StringVar(value="Spine Angle: --°"),
            'bilateral_symmetry': tk.StringVar(value="Symmetry Index: --%")
        }

    def create_tab_layout(self):
        """Create main tab layout with split views"""
        # Create view frames for anterior and sagittal views
        self.view_frames = {
            "anterior": ttk.LabelFrame(self.tab, text="Anterior View", padding="5"),
            "sagittal": ttk.LabelFrame(self.tab, text="Sagittal View", padding="5")
        }
        
        # Configure grid weights
        self.tab.grid_columnconfigure(0, weight=1)
        self.tab.grid_columnconfigure(1, weight=1)
        self.tab.grid_rowconfigure(0, weight=1)
        
        # Place view frames
        self.view_frames["anterior"].grid(row=0, column=0, sticky="nsew", padx=5)
        self.view_frames["sagittal"].grid(row=0, column=1, sticky="nsew", padx=5)

    def create_views(self):
        """Create image viewing areas with measurement capabilities"""
        for view_type, frame in self.view_frames.items():
            # Create canvas with scrollbars
            canvas_frame = ttk.Frame(frame)
            canvas_frame.pack(fill=tk.BOTH, expand=True)
            
            h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
            v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
            
            canvas = tk.Canvas(
                canvas_frame,
                bg='white',
                width=500,
                height=600,
                xscrollcommand=h_scroll.set,
                yscrollcommand=v_scroll.set
            )
            
            # Configure scrollbars
            h_scroll.config(command=canvas.xview)
            v_scroll.config(command=canvas.yview)
            
            # Layout canvas and scrollbars
            canvas.grid(row=0, column=0, sticky="nsew")
            h_scroll.grid(row=1, column=0, sticky="ew")
            v_scroll.grid(row=0, column=1, sticky="ns")
            
            canvas_frame.grid_rowconfigure(0, weight=1)
            canvas_frame.grid_columnconfigure(0, weight=1)
            
            self.canvases[view_type] = canvas
            
            # Create alignment tools
            self.alignment_tools[view_type] = AlignmentTool(canvas, view_type)

    def create_controls(self):
        """Create control panels for each view"""
        for view_type, frame in self.view_frames.items():
            control_frame = ttk.Frame(frame)
            control_frame.pack(fill=tk.X, pady=5)
            
            # Load image button
            ttk.Button(
                control_frame,
                text="Load Image",
                command=lambda vt=view_type: self.load_image(vt)
            ).pack(side=tk.LEFT, padx=2)
            
            # Measure alignment button
            ttk.Button(
                control_frame,
                text="Measure Alignment",
                command=lambda vt=view_type: self.toggle_alignment_tool(vt)
            ).pack(side=tk.LEFT, padx=2)
            
            # Clear measurements button
            ttk.Button(
                control_frame,
                text="Clear Measurements",
                command=lambda vt=view_type: self.clear_measurements(vt)
            ).pack(side=tk.LEFT, padx=2)
            
            # Reset view button
            ttk.Button(
                control_frame,
                text="Reset View",
                command=lambda vt=view_type: self.reset_view(vt)
            ).pack(side=tk.LEFT, padx=2)

    def load_image(self, view_type):
        """Load and validate image for specified view"""
        try:
            filetypes = [
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*")
            ]
            
            filename = filedialog.askopenfilename(
                title=f"Select {view_type.title()} View Image",
                filetypes=filetypes
            )
            
            if not filename:
                return
                
            # Validate image
            if not self.validate_image(filename):
                messagebox.showerror(
                    "Invalid Image",
                    "Please select a valid image file."
                )
                return
                
            # Load and process image
            image = Image.open(filename)
            self.images[view_type] = image
            
            # Display image
            self.display_image(view_type, image)
            
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def validate_image(self, filename):
        """Validate image file"""
        try:
            with Image.open(filename) as img:
                # Check format
                if img.format not in ['JPEG', 'PNG', 'BMP']:
                    return False
                    
                # Check dimensions
                if img.size[0] < 400 or img.size[1] < 600:
                    return False
                    
                # Check color mode
                if img.mode not in ['RGB', 'RGBA']:
                    return False
                    
                return True
                
        except Exception as e:
            self.logger.error(f"Image validation error: {str(e)}")
            return False

    def display_image(self, view_type, image):
        """Display image with proper scaling"""
        try:
            canvas = self.canvases[view_type]
            
            # Calculate scaling factor
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            img_width, img_height = image.size
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h, 1.0)
            
            # Resize image if necessary
            if scale < 1.0:
                new_size = (int(img_width * scale), int(img_height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and store reference
            photo = ImageTk.PhotoImage(image)
            self.photo_images[view_type] = photo
            
            # Clear canvas and display image
            canvas.delete("all")
            canvas.create_image(
                canvas_width//2,
                canvas_height//2,
                image=photo,
                anchor="center",
                tags="image"
            )
            
            # Update canvas scroll region
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        except Exception as e:
            self.logger.error(f"Error displaying image: {str(e)}")
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")

    def toggle_alignment_tool(self, view_type):
        """Toggle the alignment tool state"""
        tool = self.alignment_tools[view_type]
        if tool.active:
            tool.deactivate()
        else:
            tool.activate()

    def clear_measurements(self, view_type):
        """Clear all measurements for specified view"""
        tool = self.alignment_tools[view_type]
        tool.clear_measurements()

    def reset_view(self, view_type):
        """Reset view to original state"""
        canvas = self.canvases[view_type]
        canvas.delete("all")
        
        self.images[view_type] = None
        self.photo_images[view_type] = None

    def analyze_posture(self):
        """Analyze posture data from motion capture"""
        try:
            if not self.data_manager.motion_data:
                messagebox.showwarning("No Data", "Please load motion capture data first.")
                return

            # Get the first frame of motion data for static posture analysis
            frame_data = self.data_manager.get_frame_data(0)
            if not frame_data:
                return

            # Calculate head to pelvis alignment
            head_alignment = self.calculate_head_alignment(frame_data)
            if head_alignment is not None:
                self.metrics['head_alignment'].set(f"Head Alignment: {head_alignment:.1f}°")

            # Calculate pelvic tilt
            pelvic_tilt = self.calculate_pelvic_tilt(frame_data)
            if pelvic_tilt is not None:
                self.metrics['pelvic_tilt'].set(f"Pelvic Tilt: {pelvic_tilt:.1f}°")

            # Calculate spine angle
            spine_angle = self.calculate_spine_angle(frame_data)
            if spine_angle is not None:
                self.metrics['spine_angle'].set(f"Spine Angle: {spine_angle:.1f}°")

            # Calculate bilateral symmetry
            symmetry = self.calculate_bilateral_symmetry(frame_data)
            if symmetry is not None:
                self.metrics['bilateral_symmetry'].set(f"Symmetry Index: {symmetry:.1f}%")

        except Exception as e:
            self.logger.error(f"Error analyzing posture: {str(e)}")
            messagebox.showerror("Error", f"Failed to analyze posture: {str(e)}")

    def calculate_head_alignment(self, frame_data):
        """Calculate head to pelvis alignment angle"""
        try:
            # Get relevant keypoints
            nose = frame_data['keypoints']['nose']
            neck = frame_data['keypoints']['neck']
            mid_hip = {
                'x': (frame_data['keypoints']['leftHip']['x'] + frame_data['keypoints']['rightHip']['x']) / 2,
                'y': (frame_data['keypoints']['leftHip']['y'] + frame_data['keypoints']['rightHip']['y']) / 2,
                'z': (frame_data['keypoints']['leftHip']['z'] + frame_data['keypoints']['rightHip']['z']) / 2
            }
            
            # Calculate angle between vertical and head-pelvis line
            dx = nose['x'] - mid_hip['x']
            dy = nose['y'] - mid_hip['y']
            angle = math.degrees(math.atan2(dx, -dy))  # Negative dy for correct orientation
            
            return abs(angle)
            
        except Exception as e:
            self.logger.error(f"Error calculating head alignment: {str(e)}")
            return None

    def calculate_spine_angle(self, frame_data):
        """Calculate spine angle in sagittal plane"""
        try:
            # Get spine keypoints
            neck = frame_data['keypoints']['neck']
            mid_hip = {
                'x': (frame_data['keypoints']['leftHip']['x'] + frame_data['keypoints']['rightHip']['x']) / 2,
                'y': (frame_data['keypoints']['leftHip']['y'] + frame_data['keypoints']['rightHip']['y']) / 2,
                'z': (frame_data['keypoints']['leftHip']['z'] + frame_data['keypoints']['rightHip']['z']) / 2
            }
            
            # Calculate spine vector
            spine_vector = np.array([neck['x'] - mid_hip['x'], 
                                   neck['y'] - mid_hip['y'], 
                                   neck['z'] - mid_hip['z']])
            
            # Calculate angle with vertical
            vertical = np.array([0, 0, 1])
            angle = np.arccos(
                np.dot(spine_vector, vertical) /
                (np.linalg.norm(spine_vector) * np.linalg.norm(vertical))
            )
            
            return np.degrees(angle)
            
        except Exception as e:
            self.logger.error(f"Error calculating spine angle: {str(e)}")
            return None

    def calculate_pelvic_tilt(self, frame_data):
        """Calculate pelvic tilt angle"""
        try:
            # Get ASIS and PSIS points
            l_asis = frame_data['keypoints']['leftASIS']
            r_asis = frame_data['keypoints']['rightASIS']
            l_psis = frame_data['keypoints']['leftPSIS']
            r_psis = frame_data['keypoints']['rightPSIS']
            
            # Calculate midpoints
            mid_asis = {
                'y': (l_asis['y'] + r_asis['y']) / 2,
                'z': (l_asis['z'] + r_asis['z']) / 2
            }
            mid_psis = {
                'y': (l_psis['y'] + r_psis['y']) / 2,
                'z': (l_psis['z'] + r_psis['z']) / 2
            }
            
            # Calculate angle
            dy = mid_asis['y'] - mid_psis['y']
            dz = mid_asis['z'] - mid_psis['z']
            angle = math.degrees(math.atan2(dy, dz))
            
            return angle
            
        except Exception as e:
            self.logger.error(f"Error calculating pelvic tilt: {str(e)}")
            return None

    def calculate_bilateral_symmetry(self, frame_data):
        """Calculate bilateral symmetry index"""
        try:
            # Get relevant joint angles
            l_hip = frame_data['angles']['leftHipAngle']
            r_hip = frame_data['angles']['rightHipAngle']
            l_knee = frame_data['angles']['leftKneeAngle']
            r_knee = frame_data['angles']['rightKneeAngle']
            l_ankle = frame_data['angles']['leftAnkleAngle']
            r_ankle = frame_data['angles']['rightAnkleAngle']
            
            # Calculate absolute differences
            hip_diff = abs(l_hip - r_hip)
            knee_diff = abs(l_knee - r_knee)
            ankle_diff = abs(l_ankle - r_ankle)
            
            # Calculate mean joint angles
            hip_mean = (abs(l_hip) + abs(r_hip)) / 2
            knee_mean = (abs(l_knee) + abs(r_knee)) / 2
            ankle_mean = (abs(l_ankle) + abs(r_ankle)) / 2
            
            # Calculate symmetry index for each joint
            # Using: (1 - |L-R|/((L+R)/2)) * 100
            hip_sym = (1 - hip_diff / hip_mean) * 100 if hip_mean != 0 else 100
            knee_sym = (1 - knee_diff / knee_mean) * 100 if knee_mean != 0 else 100
            ankle_sym = (1 - ankle_diff / ankle_mean) * 100 if ankle_mean != 0 else 100
            
            # Calculate overall symmetry index (weighted average)
            weights = [0.4, 0.4, 0.2]  # Hip and knee weighted more than ankle
            symmetry_index = (
                weights[0] * hip_sym +
                weights[1] * knee_sym +
                weights[2] * ankle_sym
            )
            
            return max(0, min(100, symmetry_index))  # Clamp between 0-100
            
        except Exception as e:
            self.logger.error(f"Error calculating bilateral symmetry: {str(e)}")
            return None

class AlignmentTool:
    """Tool for measuring postural alignment angles"""
    
    def __init__(self, canvas, view_type):
        self.canvas = canvas
        self.view_type = view_type  # 'anterior' or 'sagittal'
        self.points = []
        self.lines = []
        self.active = False
        self.current_line = None
        self.angle_text = None
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<Motion>', self.on_motion)
        self.canvas.bind('<Button-3>', self.on_right_click)

    def activate(self):
        """Activate the alignment tool"""
        self.active = True
        self.canvas.config(cursor="crosshair")

    def deactivate(self):
        """Deactivate the alignment tool"""
        self.active = False
        self.canvas.config(cursor="")
        self.clear_preview()

    def on_click(self, event):
        """Handle left mouse click for point placement"""
        if not self.active:
            return
            
        try:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            
            # Add point marker
            point_id = self.canvas.create_oval(
                x-3, y-3, x+3, y+3,
                fill="red",
                outline="white",
                tags="measurement"
            )
            
            self.points.append((x, y, point_id))
            
            if len(self.points) == 2:
                self.complete_measurement()
                
        except Exception as e:
            logging.error(f"Error placing point: {str(e)}")
            self.clear_preview()

    def on_motion(self, event):
        """Handle mouse motion for measurement preview"""
        if not self.active or not self.points:
            return
            
        try:
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            
            self.update_preview(x, y)
            
        except Exception as e:
            logging.error(f"Error updating preview: {str(e)}")

    def on_right_click(self, event):
        """Handle right click to cancel current measurement"""
        if self.active:
            self.clear_preview()

    def update_preview(self, x, y):
        """Update measurement preview lines"""
        try:
            self.clear_preview()
            
            start_x, start_y, _ = self.points[0]
            
            # Draw reference line based on view type
            if self.view_type == "anterior":
                ref_line = self.canvas.create_line(
                    start_x, start_y,
                    x, start_y,
                    fill="lightblue",
                    dash=(4,4),
                    tags="preview"
                )
            else:  # sagittal
                ref_line = self.canvas.create_line(
                    x, y,
                    x, start_y,
                    fill="lightblue",
                    dash=(4,4),
                    tags="preview"
                )
                
            # Draw measurement line
            meas_line = self.canvas.create_line(
                start_x, start_y,
                x, y,
                fill="blue",
                dash=(4,4),
                tags="preview"
            )
            
            self.current_line = [ref_line, meas_line]
            
        except Exception as e:
            logging.error(f"Error updating preview: {str(e)}")

    def complete_measurement(self):
        """Complete the current measurement"""
        try:
            start_x, start_y, _ = self.points[0]
            end_x, end_y, _ = self.points[1]
            
            # Calculate angle
            angle = self.calculate_angle(start_x, start_y, end_x, end_y)
            
            # Draw permanent measurement lines
            if self.view_type == "anterior":
                self.canvas.create_line(
                    start_x, start_y,
                    end_x, start_y,
                    fill="lightblue",
                    tags="measurement"
                )
            else:
                self.canvas.create_line(
                    end_x, end_y,
                    end_x, start_y,
                    fill="lightblue",
                    tags="measurement"
                )
                
            line_id = self.canvas.create_line(
                start_x, start_y,
                end_x, end_y,
                fill="blue",
                width=2,
                tags="measurement"
            )
            
            # Draw angle arc and text
            self.draw_angle_marker(start_x, start_y, end_x, end_y, angle)
            
            self.lines.append(line_id)
            self.points = []
            self.clear_preview()
            
        except Exception as e:
            logging.error(f"Error completing measurement: {str(e)}")
            self.clear_preview()

    def calculate_angle(self, x1, y1, x2, y2):
        """Calculate angle based on view type"""
        try:
            if self.view_type == "anterior":
                # Calculate angle from horizontal
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                return angle
            else:
                # Calculate angle from vertical
                angle = math.degrees(math.atan2(x2 - x1, -(y2 - y1)))
                return abs(angle)
                
        except Exception as e:
            logging.error(f"Error calculating angle: {str(e)}")
            return 0.0

    def draw_angle_marker(self, x1, y1, x2, y2, angle):
        """Draw angle arc and text"""
        try:
            radius = 30
            text_x = x1 + 50
            text_y = y1
            
            # Draw arc
            start_angle = 0 if self.view_type == "anterior" else 90
            arc_id = self.canvas.create_arc(
                x1 - radius, y1 - radius,
                x1 + radius, y1 + radius,
                start=start_angle,
                extent=angle,
                style="arc",
                outline="red",
                tags="measurement"
            )
            
            # Draw angle text
            text_id = self.canvas.create_text(
                text_x, text_y,
                text=f"{abs(angle):.1f}°",
                fill="red",
                font=("Arial", 12),
                tags="measurement"
            )
            
            self.lines.extend([arc_id, text_id])
            
        except Exception as e:
            logging.error(f"Error drawing angle marker: {str(e)}")

    def clear_preview(self):
        """Clear preview lines"""
        if self.current_line:
            for line_id in self.current_line:
                self.canvas.delete(line_id)
            self.current_line = None

    def clear_measurements(self):
        """Clear all measurements and reset state"""
        self.canvas.delete("measurement")
        self.canvas.delete("preview")
        self.points = []
        self.lines = []
        self.current_line = None

    def load_motion_data(self, filename: str) -> bool:
        """Load and validate JSON motion capture data"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            print("\n=== Motion Data File Structure Analysis ===")
            print(f"Analyzing file: {filename}")
            
            if self._validate_data_structure(data):
                self.motion_data = self.standardize_data_format(data)
                print("\n=== Validation Results ===")
                print("All required fields present")
                print("\n=== End of Structure Analysis ===\n")
                return True
                
            print("\n=== Validation Results ===")
            print("Validation failed:")
            for error in self.validation_errors:
                print(f"- {error}")
            print("\n=== End of Structure Analysis ===\n")
            return False
                
        except Exception as e:
            self.logger.error(f"Error loading motion data: {str(e)}")
            return False

    def standardize_data_format(self, raw_data):
        """Standardize motion data into internal format"""
        try:
            standardized = {
                'frames': [],
                'metadata': {
                    'frame_count': len(raw_data),
                    'keypoint_count': len(raw_data[0]['keypoints2D'])
                }
            }
            
            # Process each frame
            for frame in raw_data:
                processed_frame = {
                    'keypoints2D': self._process_keypoints(frame['keypoints2D']),
                    'keypoints3D': self._process_keypoints(frame['keypoints3D']),
                    'angles2D': frame['angles2D'],
                    'angles3D': frame['angles3D'],
                    'com2D': frame['com2D'],
                    'timestamp': frame.get('timestamp', 0)
                }
                standardized['frames'].append(processed_frame)
                
            return standardized
            
        except Exception as e:
            raise RuntimeError(f"Error standardizing data format: {str(e)}")
        
class BodyWeightSquatTab:
    """Tab for bodyweight squat movement assessment"""
    
    def __init__(self, notebook, data_manager):
        self.tab = ttk.Frame(notebook, padding="10")
        self.data_manager = data_manager

        # Initialize state variables
        self.motion_data = None 
        self.sagittal_data = None
        self.anterior_data = None
        self.analysis_complete = False
        self.results = {}
        
        # Initialize analysis parameters
        self.params = {
            'spine_shin_threshold': 15.0,  # Maximum acceptable deviation from parallel
            'knee_valgus_threshold': 10.0,  # Maximum acceptable knee valgus angle
            'depth_threshold': 90.0,  # Target knee flexion angle
            'stability_threshold': 50.0  # mm of COM deviation considered unstable
        }
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Create tab content
        self.create_tab_layout()
        self.setup_visualization()
        self.create_controls()
        self.create_tab_content()

    def create_tab_layout(self):
        """Create main tab layout"""
        # Configure grid
        self.tab.grid_columnconfigure(1, weight=1)  # Visualization area
        self.tab.grid_rowconfigure(0, weight=1)
        
        # Create control panel
        self.control_frame = ttk.Frame(self.tab, padding="5")
        self.control_frame.grid(row=0, column=0, sticky="ns", padx=5)
        
        # Create visualization area
        self.viz_frame = ttk.Frame(self.tab)
        self.viz_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        
        # Configure visualization grid
        for i in range(2):
            self.viz_frame.grid_rowconfigure(i, weight=1)
            self.viz_frame.grid_columnconfigure(i, weight=1)

    def setup_visualization(self):
        """Setup matplotlib visualization areas"""
        # Create figures
        self.figs = {
            'motion': Figure(figsize=(6, 4)),
            'angles': Figure(figsize=(6, 4)),
            'symmetry': Figure(figsize=(6, 4)),
            'summary': Figure(figsize=(6, 4))
        }
        
        # Create axes
        self.axes = {}
        for key, fig in self.figs.items():
            self.axes[key] = fig.add_subplot(111)
            
        # Create canvases
        self.canvases = {}
        for key, fig in self.figs.items():
            self.canvases[key] = FigureCanvasTkAgg(fig, self.viz_frame)
            
        # Grid layout
        self.canvases['motion'].get_tk_widget().grid(
            row=0, column=0, padx=5, pady=5, sticky="nsew"
        )
        self.canvases['angles'].get_tk_widget().grid(
            row=0, column=1, padx=5, pady=5, sticky="nsew"
        )
        self.canvases['symmetry'].get_tk_widget().grid(
            row=1, column=0, padx=5, pady=5, sticky="nsew"
        )
        self.canvases['summary'].get_tk_widget().grid(
            row=1, column=1, padx=5, pady=5, sticky="nsew"
        )

    def create_controls(self):
        """Create control panel elements"""
        # Data loading section
        load_frame = ttk.LabelFrame(self.control_frame, text="Data Import", padding="5")
        load_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            load_frame,
            text="Load Anterior View",
            command=lambda: self.load_motion_data("anterior")
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            load_frame,
            text="Load Sagittal View",
            command=lambda: self.load_motion_data("sagittal")
        ).pack(fill=tk.X, pady=2)
        
        # Analysis parameters section
        param_frame = ttk.LabelFrame(
            self.control_frame,
            text="Analysis Parameters",
            padding="5"
        )
        param_frame.pack(fill=tk.X, pady=5)
        
        # Parameter controls
        for param, value in self.params.items():
            frame = ttk.Frame(param_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(
                frame,
                text=param.replace('_', ' ').title()
            ).pack(side=tk.LEFT)
            
            var = tk.DoubleVar(value=value)
            ttk.Entry(
                frame,
                textvariable=var,
                width=10
            ).pack(side=tk.RIGHT)
            
            setattr(self, f"{param}_var", var)
            
        # Metrics display
        self.create_metrics_display()
        
        # Analysis button
        self.analyze_btn = ttk.Button(
            self.control_frame,
            text="Analyze Movement",
            command=self.analyze_movement,
            state="disabled"
        )
        self.analyze_btn.pack(pady=5)
        
        # Export button
        self.export_btn = ttk.Button(
            self.control_frame,
            text="Export Results",
            command=self.export_results,
            state="disabled"
        )
        self.export_btn.pack(pady=5)

    def create_tab_content(self):
        """Create the tab layout and controls"""
        # Configure grid
        self.tab.grid_columnconfigure(1, weight=1)
        self.tab.grid_rowconfigure(0, weight=1)
        
        # Create control panel
        self.control_frame = ttk.Frame(self.tab, padding="5")
        self.control_frame.grid(row=0, column=0, sticky="ns", padx=5)
        
        # Create visualization area with split views
        self.viz_frame = ttk.Frame(self.tab)
        self.viz_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        
        # Create anterior and sagittal view frames
        self.anterior_frame = ttk.LabelFrame(self.viz_frame, text="Anterior View")
        self.anterior_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.sagittal_frame = ttk.LabelFrame(self.viz_frame, text="Sagittal View")
        self.sagittal_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Configure visualization grid
        self.viz_frame.grid_columnconfigure(0, weight=1)
        self.viz_frame.grid_columnconfigure(1, weight=1)
        self.viz_frame.grid_rowconfigure(0, weight=1)
        
        # Setup visualization plots
        self.setup_visualization()
        
        # Create controls
        self.create_controls()
    
    def create_metrics_display(self):
        """Create metrics display area"""
        metrics_frame = ttk.LabelFrame(
            self.control_frame,
            text="Movement Metrics",
            padding="5"
        )
        metrics_frame.pack(fill=tk.X, pady=5)
        
        # Initialize metric variables
        self.metrics = {
            'max_spine_angle': tk.StringVar(value="Max Spine Angle: --°"),
            'max_knee_valgus': tk.StringVar(value="Max Knee Valgus: --°"),
            'squat_depth': tk.StringVar(value="Squat Depth: --%"),
            'symmetry_index': tk.StringVar(value="Symmetry Index: --"),
            'movement_quality': tk.StringVar(value="Movement Quality: --")
        }
        
        # Create metric labels
        for var in self.metrics.values():
            ttk.Label(
                metrics_frame,
                textvariable=var
            ).pack(anchor=tk.W, pady=2)

    def load_motion_data(self, view_type: str):
        """Load and validate motion data for specified view"""
        try:
            filename = filedialog.askopenfilename(
                title=f"Select {view_type.title()} View Data",
                filetypes=[("JSON files", "*.json")]
            )
            
            if not filename:
                return
                
            success, data = self.data_manager.load_data(filename, view_type)
            
            if success:
                self.view_data[view_type] = data
                self.update_analysis_state()
                messagebox.showinfo(
                    "Success",
                    f"{view_type.title()} view data loaded successfully"
                )
            else:
                messagebox.showerror(
                    "Error",
                    f"Failed to load {view_type} view data\n\n" +
                    "\n".join(self.data_manager.validation_errors)
                )
                
        except Exception as e:
            self.logger.error(f"Error loading {view_type} data: {str(e)}")
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def update_analysis_state(self):
        """Update UI state based on loaded data"""
        if self.anterior_data and self.sagittal_data:
            self.analyze_btn.configure(state="normal")
        else:
            self.analyze_btn.configure(state="disabled")
        
    def clear_visualizations(self):
        """Clear all visualization plots"""
        for ax in self.axes.values():
            ax.clear()
        for canvas in self.canvases.values():
            canvas.draw()

    def update_preview(self):
        """Update preview visualization of loaded data"""
        if not self.motion_data:
            return
            
        try:
            # Clear motion plot
            self.axes['motion'].clear()
            
            # Plot first frame skeleton
            frame = self.motion_data[0]
            self.plot_skeleton(frame['keypoints3D'])
            
            # Configure plot
            self.axes['motion'].set_title('Motion Data Preview')
            self.axes['motion'].set_aspect('equal')
            self.axes['motion'].grid(True)
            
            # Refresh canvas
            self.canvases['motion'].draw()
            
        except Exception as e:
            self.logger.error(f"Error updating preview: {str(e)}")

    def plot_skeleton(self, keypoints):
        """Plot skeleton from keypoints"""
        try:
            # Define segments to plot
            segments = [
                ('leftShoulder', 'rightShoulder'),
                ('leftShoulder', 'leftHip'),
                ('rightShoulder', 'rightHip'),
                ('leftHip', 'rightHip'),
                ('leftHip', 'leftKnee'),
                ('rightHip', 'rightKnee'),
                ('leftKnee', 'leftAnkle'),
                ('rightKnee', 'rightAnkle')
            ]
            
            # Plot each segment
            for start, end in segments:
                if start in keypoints and end in keypoints:
                    start_point = keypoints[start]
                    end_point = keypoints[end]
                    
                    # Plot in sagittal plane (Y-Z)
                    self.axes['motion'].plot(
                        [start_point[1], end_point[1]],  # Y coordinates
                        [start_point[2], end_point[2]],  # Z coordinates
                        'b-', linewidth=2
                    )
                    
            # Plot joint markers
            for point in keypoints.values():
                self.axes['motion'].plot(
                    point[1], point[2],  # Y-Z coordinates
                    'ro', markersize=6
                )
                
        except Exception as e:
            self.logger.error(f"Error plotting skeleton: {str(e)}")

    def export_results(self):
        """Export analysis results"""
        if not self.analysis_results:
            messagebox.showwarning(
                "No Results",
                "Please analyze the movement before exporting results."
            )
            return
            
        try:
            # Get export filename
            filename = filedialog.asksaveasfilename(
                title="Export Analysis Results",
                defaultextension=".xlsx",
                filetypes=[
                    ("Excel files", "*.xlsx"),
                    ("CSV files", "*.csv")
                ]
            )
            
            if not filename:
                return
                
            # Prepare results for export
            export_data = {
                'Parameters': list(self.params.keys()),
                'Values': [getattr(self, f"{param}_var").get() for param in self.params],
                'Metrics': list(self.metrics.keys()),
                'Results': [var.get() for var in self.metrics.values()]
            }
            
            # Export to Excel/CSV
            df = pd.DataFrame(export_data)
            if filename.endswith('.xlsx'):
                df.to_excel(filename, index=False)
            else:
                df.to_csv(filename, index=False)
                
            messagebox.showinfo(
                "Success",
                "Analysis results exported successfully."
            )
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            messagebox.showerror(
                "Error",
                f"Failed to export results: {str(e)}"
            )

    def analyze_movement(self):
        """Analyze squat movement mechanics"""
        if not self.motion_data:
            messagebox.showwarning("No Data", "Please load motion data first.")
            return
            
        try:
            # Initialize results storage
            self.results = {
                'angles': {'spine_angles': [], 'shin_angles': [], 'parallel_deviation': []},
                'symmetry': {'knee_differences': [], 'hip_differences': []},
                'com': {'displacements': [], 'max_deviation': 0},
                'quality': {'alignment_score': 0, 'symmetry_score': 0, 'stability_score': 0}
            }
            
            # Process each frame
            for frame in self.motion_data['frames']:
                # Analyze spine-to-shin alignment
                alignment = self.data_manager.calculate_spine_to_shin_angle(frame)
                if alignment:
                    self.results['angles']['spine_angles'].append(alignment['spine_angle'])
                    self.results['angles']['shin_angles'].append(alignment['shin_angle'])
                    self.results['angles']['parallel_deviation'].append(alignment['parallel_deviation'])
                
                # Analyze bilateral symmetry
                symmetry = self.data_manager.calculate_bilateral_symmetry(frame)
                if symmetry:
                    self.results['symmetry']['knee_differences'].append(symmetry['knee_difference'])
                    self.results['symmetry']['hip_differences'].append(symmetry['hip_difference'])
                
                # Track COM deviation
                com_deviation = self.data_manager.calculate_com_deviation(frame['com2D'])
                if com_deviation is not None:
                    self.results['com']['displacements'].append(com_deviation)
            
            # Convert lists to numpy arrays for calculations
            for category in ['angles', 'symmetry', 'com']:
                for key in self.results[category]:
                    if isinstance(self.results[category][key], list):
                        self.results[category][key] = np.array(self.results[category][key])
            
            # Calculate summary metrics
            self.results['angles']['max_deviation'] = np.max(self.results['angles']['parallel_deviation'])
            self.results['com']['max_deviation'] = np.max(self.results['com']['displacements'])
            self.results['symmetry']['mean_asymmetry'] = np.mean(
                self.results['symmetry']['knee_differences'] + 
                self.results['symmetry']['hip_differences']
            )
            
            # Calculate movement quality scores
            self.calculate_movement_quality()
            
            # Update display
            self.analysis_complete = True
            self.update_metrics()
            self.update_visualization()
            self.export_btn.configure(state="normal")
            
        except Exception as e:
            self.logger.error(f"Error in movement analysis: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def _calculate_spine_angle(self, keypoints):
        """Calculate spine-to-shine angle in sagittal plane"""
        try:
            # Calculate mid-points
            mid_shoulder = np.mean([
                keypoints['rightShoulder'],
                keypoints['leftShoulder']
            ], axis=0)
            
            mid_hip = np.mean([
                keypoints['rightHip'],
                keypoints['leftHip']
            ], axis=0)
            
            # Calculate spine vector
            spine_vector = mid_shoulder - mid_hip
            
            # Calculate angle with vertical
            vertical = np.array([0, 0, 1])
            angle = np.arccos(
                np.dot(spine_vector, vertical) /
                (np.linalg.norm(spine_vector) * np.linalg.norm(vertical))
            )
            
            return np.degrees(angle)
            
        except Exception as e:
            self.logger.error(f"Error calculating spine angle: {str(e)}")
            return None

    def analyze_spine_shin_alignment(self):
        """Analyze spine-to-shin alignment in sagittal plane"""
        try:
            frames = self.sagittal_data['frames']
            spine_angles = []
            shin_angles = []
            
            for frame in frames:
                # Calculate spine angle from 2D joint positions
                shoulder = frame['keypoints2D']['rightShoulder']
                hip = frame['keypoints2D']['rightHip']
                spine_angle = math.atan2(
                    shoulder['y'] - hip['y'],
                    shoulder['x'] - hip['x']
                )
                spine_angles.append(math.degrees(spine_angle))
                
                # Calculate shin angle from 2D joint positions
                knee = frame['keypoints2D']['rightKnee']
                ankle = frame['keypoints2D']['rightAnkle']
                shin_angle = math.atan2(
                    knee['y'] - ankle['y'],
                    knee['x'] - ankle['x']
                )
                shin_angles.append(math.degrees(shin_angle))
            
            # Calculate parallel deviation
            deviations = [abs(s - sh) for s, sh in zip(spine_angles, shin_angles)]
            
            self.results['angles'].update({
                'spine_angles': spine_angles,
                'shin_angles': shin_angles,
                'parallel_deviation': deviations,
                'max_deviation': max(deviations),
                'mean_deviation': np.mean(deviations)
            })
            
        except Exception as e:
            self.logger.error(f"Error analyzing spine-shin alignment: {str(e)}")
            raise

    def analyze_bilateral_symmetry(self):
        """Analyze bilateral symmetry using anterior view data"""
        try:
            frames = self.anterior_data['frames']
            knee_differences = []
            hip_differences = []
            
            for frame in frames:
                # Compare knee angles
                left_knee = frame['angles2D']['leftKneeAngle']
                right_knee = frame['angles2D']['rightKneeAngle']
                knee_diff = abs(left_knee - right_knee)
                knee_differences.append(knee_diff)
                
                # Compare hip angles
                left_hip = frame['angles2D']['leftHipAngle']
                right_hip = frame['angles2D']['rightHipAngle']
                hip_diff = abs(left_hip - right_hip)
                hip_differences.append(hip_diff)
            
            self.results['symmetry'].update({
                'knee_differences': knee_differences,
                'hip_differences': hip_differences,
                'max_knee_asymmetry': max(knee_differences),
                'max_hip_asymmetry': max(hip_differences),
                'mean_asymmetry': np.mean(knee_differences + hip_differences)
            })
            
        except Exception as e:
            self.logger.error(f"Error analyzing bilateral symmetry: {str(e)}")
    
    def _calculate_com_deviation(self, com):
        """Calculate COM deviation from reference position"""
        try:
            if not hasattr(self, 'reference_com'):
                self.reference_com = np.array([com['x'], com['y'], com['z']])
                return 0.0
                
            current_com = np.array([com['x'], com['y'], com['z']])
            deviation = np.linalg.norm(current_com - self.reference_com)
            return deviation
            
        except Exception as e:
            self.logger.error(f"Error calculating COM deviation: {str(e)}")
            return None

    def _calculate_summary_metrics(self):
        """Calculate summary metrics for the squat movement"""
        try:
            # Calculate max spine angle
            max_spine = np.max(self.analysis_results['spine_angles'])
            self.metrics['max_spine_angle'].set(f"Max Spine Angle: {max_spine:.1f}°")
            
            # Calculate bilateral symmetry
            left_knee = np.array(self.analysis_results['knee_angles']['left'])
            right_knee = np.array(self.analysis_results['knee_angles']['right'])
            max_diff = np.max(np.abs(left_knee - right_knee))
            symmetry = 100 * (1 - max_diff / 180)  # Normalize to percentage
            self.metrics['symmetry_index'].set(f"Symmetry Index: {symmetry:.1f}%")
            
            # Calculate squat depth from knee angles
            max_knee = np.max([np.max(left_knee), np.max(right_knee)])
            depth = (max_knee / 180) * 100  # Convert to percentage
            self.metrics['squat_depth'].set(f"Squat Depth: {depth:.1f}%")
            
            # Calculate stability score from COM deviation
            mean_deviation = np.mean(self.analysis_results['com_deviations'])
            max_deviation = np.max(self.analysis_results['com_deviations'])
            stability = 100 * (1 - mean_deviation / max_deviation)
            self.metrics['stability_score'].set(f"Stability Score: {stability:.1f}")
            
        except Exception as e:
            self.logger.error(f"Error calculating summary metrics: {str(e)}")

    def update_visualization(self):
        """Update all visualization plots"""
        try:
            if not self.analysis_complete:
                return
                
            # Clear previous plots
            for ax in self.axes.values():
                ax.clear()
            
            # Plot spine-shin alignment
            self.plot_alignment()
            
            # Plot bilateral symmetry
            self.plot_symmetry()
            
            # Plot COM trajectory
            self.plot_com_trajectory()
            
            # Plot quality metrics
            self.plot_quality_metrics()
            
            # Refresh all canvases
            for canvas in self.canvases.values():
                canvas.draw()
                
        except Exception as e:
            self.logger.error(f"Error updating visualization: {str(e)}")

    def plot_spine_shin_alignment(self):
        """Plot spine and shin angles over time"""
        ax = self.axes['alignment']
        frames = range(len(self.results['angles']['spine_angles']))
        
        # Plot angles
        ax.plot(frames, self.results['angles']['spine_angles'],
                'b-', label='Spine Angle')
        ax.plot(frames, self.results['angles']['shin_angles'],
                'r-', label='Shin Angle')
                
        # Add threshold lines
        threshold = self.params['spine_shin_deviation']
        ax.axhline(y=threshold, color='k', linestyle='--', alpha=0.5)
        ax.axhline(y=-threshold, color='k', linestyle='--', alpha=0.5)
        
        ax.set_title('Spine-Shin Alignment')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Angle (degrees)')
        ax.legend()
        ax.grid(True)

    def plot_vector_relationship(self):
        """Plot shin and trunk vector orientations"""
        ax = self.axes['vectors']
        frame_data = self.results['frame_data']

        # Create time axis based on percentage of squat
        frames = range(len(frame_data))
        movement_percentage = [i / len(frames) * 100 for i in frames]

        # Plot vector angles
        ax.plot(movement_percentage, 
                [d['angles']['shin_angle'] for d in frame_data],
                'b-', label='Shin Angle', linewidth=2)
        ax.plot(movement_percentage,
                [d['angles']['trunk_angle'] for d in frame_data],
                'r-', label='Trunk Angle', linewidth=2)

        # Add reference lines for ideal ranges
        ax.axhline(y=45, color='g', linestyle='--', alpha=0.3, label='Ideal Angle')
        ax.fill_between(movement_percentage, 25, 65, color='g', alpha=0.1,
                       label='Acceptable Range')

        ax.set_title('Shin and Trunk Angles During Squat')
        ax.set_xlabel('Movement Completion (%)')
        ax.set_ylabel('Angle (degrees)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_parallel_deviation(self):
        """Plot deviation from parallel alignment"""
        ax = self.axes['parallel']
        frame_data = self.results['frame_data']

        # Create time axis
        frames = range(len(frame_data))
        movement_percentage = [i / len(frames) * 100 for i in frames]

        # Plot parallel deviation
        deviations = [d['angles']['parallel_deviation'] for d in frame_data]
        ax.plot(movement_percentage, deviations, 'g-', 
                label='Deviation from Parallel', linewidth=2)

        # Add threshold line
        ax.axhline(y=15, color='r', linestyle='--', alpha=0.5,
                   label='Maximum Acceptable Deviation')

        # Color regions based on deviation severity
        ax.fill_between(movement_percentage, 0, 15, color='g', alpha=0.1,
                       label='Good')
        ax.fill_between(movement_percentage, 15, max(max(deviations), 30),
                       color='r', alpha=0.1, label='Poor')

        ax.set_title('Spine-to-Shin Parallel Alignment')
        ax.set_xlabel('Movement Completion (%)')
        ax.set_ylabel('Deviation from Parallel (degrees)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def plot_quality_metrics(self):
        """Plot overall movement quality metrics"""
        ax = self.axes['quality']
        frame_data = self.results['frame_data']

        # Calculate quality metrics for each phase
        movement_phases = ['Descent', 'Bottom', 'Ascent']
        phase_scores = self.calculate_phase_scores(frame_data)

        # Create bar plot
        x_pos = np.arange(len(movement_phases))
        ax.bar(x_pos, phase_scores, align='center', alpha=0.8,
               color=['lightblue', 'lightgreen', 'lightcoral'])

        # Customize plot
        ax.set_title('Movement Quality by Phase')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(movement_phases)
        ax.set_ylabel('Quality Score (%)')
        ax.set_ylim(0, 100)

        # Add threshold line for acceptable quality
        ax.axhline(y=80, color='g', linestyle='--', alpha=0.5,
                   label='Quality Threshold')

        for i, score in enumerate(phase_scores):
            ax.text(i, score + 1, f'{score:.1f}%',
                    ha='center', va='bottom')

        ax.grid(True, alpha=0.3)

    def calculate_phase_scores(self, frame_data):
        """Calculate quality scores for each phase of the movement"""
        # Divide movement into phases
        n_frames = len(frame_data)
        descent = frame_data[:n_frames//3]
        bottom = frame_data[n_frames//3:2*n_frames//3]
        ascent = frame_data[2*n_frames//3:]

        # Calculate average quality score for each phase
        phase_scores = []
        for phase in [descent, bottom, ascent]:
            scores = [frame['quality_score'] for frame in phase]
            phase_scores.append(np.mean(scores))

        return phase_scores

    def analyze_squat_alignment(self, frame):
        """
        Analyze alignment between shin and trunk vectors during squat.
        Calculates the relationship between these vectors throughout the movement.
        """
        try:
            # Calculate shin vector (ankle to knee)
            ankle = np.array([
                frame['keypoints']['rightAnkle']['x'],
                frame['keypoints']['rightAnkle']['y'],
                frame['keypoints']['rightAnkle']['z']
            ])
            knee = np.array([
                frame['keypoints']['rightKnee']['x'],
                frame['keypoints']['rightKnee']['y'],
                frame['keypoints']['rightKnee']['z']
            ])
            shin_vector = knee - ankle
            
            # Calculate trunk vector (hip to shoulder)
            hip = np.array([
                frame['keypoints']['rightHip']['x'],
                frame['keypoints']['rightHip']['y'],
                frame['keypoints']['rightHip']['z']
            ])
            shoulder = np.array([
                frame['keypoints']['rightShoulder']['x'],
                frame['keypoints']['rightShoulder']['y'],
                frame['keypoints']['rightShoulder']['z']
            ])
            trunk_vector = shoulder - hip
            
            # Normalize vectors
            shin_vector = shin_vector / np.linalg.norm(shin_vector)
            trunk_vector = trunk_vector / np.linalg.norm(trunk_vector)
            
            # Calculate angle between vectors (perfect parallel = 0° or 180°)
            angle_between = np.arccos(np.clip(np.abs(np.dot(shin_vector, trunk_vector)), -1.0, 1.0))
            angle_between_deg = np.degrees(angle_between)
            
            # Calculate deviation from parallel (0° = perfect parallel)
            parallel_deviation = min(angle_between_deg, abs(180 - angle_between_deg))
            
            # Calculate reference angles relative to vertical
            vertical = np.array([0, 0, 1])
            shin_vertical = np.degrees(np.arccos(np.clip(np.dot(shin_vector, vertical), -1.0, 1.0)))
            trunk_vertical = np.degrees(np.arccos(np.clip(np.dot(trunk_vector, vertical), -1.0, 1.0)))
            
            return {
                'parallel_deviation': parallel_deviation,
                'shin_vector': shin_vector,
                'trunk_vector': trunk_vector,
                'shin_angle': shin_vertical,
                'trunk_angle': trunk_vertical
            }
        except Exception as e:
            self.logger.error(f"Error analyzing squat alignment: {str(e)}")
            return None

    def calculate_movement_quality(self):
        """Calculate overall movement quality score"""
        try:
            # Component scores (0-100 scale)
            alignment_score = max(0, 100 - (
                self.results['angles']['max_deviation'] * 2
            ))
            
            symmetry_score = max(0, 100 - (
                self.results['symmetry']['mean_asymmetry'] * 2
            ))
            
            stability_score = max(0, 100 - (
                self.results['com']['max_deviation'] / self.params['com_deviation'] * 100
            ))
            
            # Weighted total score
            weights = {
                'alignment': 0.4,
                'symmetry': 0.4,
                'stability': 0.2
            }
            
            total_score = (
                alignment_score * weights['alignment'] +
                symmetry_score * weights['symmetry'] +
                stability_score * weights['stability']
            )
            
            self.results['quality'].update({
                'alignment_score': alignment_score,
                'symmetry_score': symmetry_score,
                'stability_score': stability_score,
                'total_score': total_score
            })
            
        except Exception as e:
            self.logger.error(f"Error calculating movement quality: {str(e)}")
            raise

    def update_metrics(self):
        """Update displayed metrics based on analysis results"""
        try:
            if not self.analysis_complete:
                return
                
            # Update spine-shin alignment metrics
            self.metrics['max_spine_angle'].set(
                f"Max Spine Angle: {np.max(self.results['angles']['spine_angles']):.1f}°"
            )
            self.metrics['parallel_deviation'].set(
                f"Max Parallel Deviation: {self.results['angles']['max_deviation']:.1f}°"
            )
            
            # Update symmetry metrics
            self.metrics['symmetry_index'].set(
                f"Symmetry Index: {100 - self.results['symmetry']['mean_asymmetry']:.1f}%"
            )
            
            # Update stability metrics
            self.metrics['stability_score'].set(
                f"Stability Score: {self.results['quality']['stability_score']:.1f}"
            )
            
            # Update overall quality score
            self.metrics['movement_quality'].set(
                f"Movement Quality: {self.results['quality']['total_score']:.1f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
   
class SquatAnalyzer:
    """Analyzes squat movement using sagittal and anterior view data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define assessment thresholds
        self.thresholds = {
            'spine_shin_deviation': 15.0,  # Maximum deviation from parallel
            'knee_valgus': 15.0,  # Maximum knee valgus angle
            'knee_flexion': 90.0,  # Target knee flexion angle
            'asymmetry': 10.0  # Maximum bilateral difference
        }

    def analyze_squat(self, frame_data):
        """Analyze squat movement from all views"""
        # Sagittal view analysis
        sagittal_results = self.analyze_sagittal_view(frame_data['angles2D'])
        
        # Anterior view analysis  
        anterior_results = self.analyze_anterior_view(frame_data['angles2D'])
        
        # 3D knee analysis
        knee_results = self.analyze_knee_mechanics(frame_data['angles3D'])

    def _validate_data(self, frame_data):
        """Validate required angle data is present"""
        required_2d = ['spine_angle', 'shin_angle']
        required_3d = ['leftKneeAngle', 'rightKneeAngle']
        
        has_2d = all(angle in frame_data.get('angles2D', {}) 
                    for angle in required_2d)
        has_3d = all(angle in frame_data.get('angles3D', {})
                    for angle in required_3d)
                    
        return has_2d and has_3d

    def analyze_sagittal_mechanics(self, angles_2d):
        """Analyze sagittal plane mechanics using 2D angles"""
        try:
            # Calculate spine-to-shin relationship
            spine_angle = angles_2d['spine_angle']
            shin_angle = angles_2d['shin_angle']
            
            parallel_deviation = abs(spine_angle - shin_angle)
            
            # Calculate trunk lean
            trunk_angle = angles_2d.get('trunk_angle', 0.0)
            
            return {
                'spine_angle': spine_angle,
                'shin_angle': shin_angle,
                'parallel_deviation': parallel_deviation,
                'trunk_angle': trunk_angle
            }
            
        except Exception as e:
            self.logger.error(f"Error in sagittal analysis: {str(e)}")
            return None

    def analyze_anterior_mechanics(self, angles_2d):
        """Analyze anterior view mechanics using 2D angles"""
        try:
            # Get knee angles from anterior view
            left_knee = angles_2d.get('leftKneeAngle_frontal', 0.0)
            right_knee = angles_2d.get('rightKneeAngle_frontal', 0.0)
            
            # Calculate bilateral symmetry
            knee_difference = abs(left_knee - right_knee)
            symmetry_index = 100 * (1 - knee_difference / 180)
            
            # Check for valgus
            max_valgus = max(abs(left_knee), abs(right_knee))
            
            return {
                'left_knee': left_knee,
                'right_knee': right_knee,
                'symmetry_index': symmetry_index,
                'max_valgus': max_valgus
            }
            
        except Exception as e:
            self.logger.error(f"Error in anterior analysis: {str(e)}")
            return None

    def analyze_knee_mechanics(self, angles_3d):
        """Analyze knee mechanics using 3D angles"""
        try:
            # Get 3D knee flexion angles
            left_knee = angles_3d['leftKneeAngle']
            right_knee = angles_3d['rightKneeAngle']
            
            # Calculate depth and symmetry
            max_flexion = max(left_knee, right_knee)
            min_flexion = min(left_knee, right_knee)
            depth_symmetry = abs(left_knee - right_knee)
            
            return {
                'left_flexion': left_knee,
                'right_flexion': right_knee,
                'max_flexion': max_flexion,
                'min_flexion': min_flexion,
                'depth_symmetry': depth_symmetry
            }
            
        except Exception as e:
            self.logger.error(f"Error in knee analysis: {str(e)}")
            return None

    def calculate_movement_quality(self, sagittal, anterior, knee):
        """Calculate overall movement quality score"""
        try:
            scores = []
            
            # Score spine-shin alignment (30%)
            spine_shin_score = max(0, 100 - 
                (sagittal['parallel_deviation'] * 2))
            scores.append(spine_shin_score * 0.3)
            
            # Score knee mechanics (40%)
            knee_score = min(100, knee['min_flexion'] / 
                self.thresholds['knee_flexion'] * 100)
            scores.append(knee_score * 0.4)
            
            # Score symmetry (30%)
            sym_score = anterior['symmetry_index']
            scores.append(sym_score * 0.3)
            
            return sum(scores)
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {str(e)}")
            return 0

    def get_movement_feedback(self, analysis_results):
        """Generate coaching cues based on analysis"""
        try:
            feedback = []
            
            # Check spine-shin alignment
            if analysis_results['sagittal']['parallel_deviation'] > self.thresholds['spine_shin_deviation']:
                feedback.append("Maintain parallel shin and spine angles")
                
            # Check depth
            if analysis_results['knee']['min_flexion'] < self.thresholds['knee_flexion']:
                feedback.append("Work on achieving greater depth")
                
            # Check symmetry
            if analysis_results['anterior']['symmetry_index'] < 90:
                feedback.append("Focus on maintaining even weight distribution")
                
            return feedback if feedback else ["Good form maintained"]
            
        except Exception as e:
            self.logger.error(f"Error generating feedback: {str(e)}")
            return ["Unable to generate feedback"]

    def update_visualization(self):
        """Update visualization with current analysis results"""
        try:
            if not self.analysis_results:
                return
                
            # Get data for plotting
            frames = range(len(self.analysis_results))
            spine_angles = [r['spine_angle'] for r in self.analysis_results]
            shin_angles = [r['shin_angle'] for r in self.analysis_results]
            deviations = [r['parallel_deviation'] for r in self.analysis_results]
            quality_scores = [r['quality_score'] for r in self.analysis_results]
            
            # Clear previous plots
            for ax in self.axes.values():
                ax.clear()
            
            # Plot angle relationships
            ax = self.axes['angles']
            ax.plot(frames, spine_angles, 'b-', label='Spine Angle')
            ax.plot(frames, shin_angles, 'r-', label='Shin Angle')
            ax.set_title('Spine and Shin Angles')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Angle (degrees)')
            ax.legend()
            ax.grid(True)
            
            # Plot parallel deviation
            ax = self.axes['deviation']
            ax.plot(frames, deviations, 'g-')
            ax.axhline(y=15, color='r', linestyle='--', alpha=0.5)
            ax.set_title('Spine-to-Shin Parallel Deviation')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Deviation (degrees)')
            ax.grid(True)
            
            # Plot movement quality
            ax = self.axes['quality']
            ax.plot(frames, quality_scores, 'b-')
            ax.set_title('Movement Quality Score')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 100)
            ax.grid(True)
            
            # Refresh canvases
            for canvas in self.canvases.values():
                canvas.draw()
                
        except Exception as e:
            self.logger.error(f"Error updating visualization: {str(e)}")

    def update_metrics(self):
        """Update displayed metrics based on analysis results"""
        try:
            if not self.analysis_results:
                return
                
            # Calculate summary metrics
            max_deviation = max(r['parallel_deviation'] for r in self.analysis_results)
            avg_quality = np.mean([r['quality_score'] for r in self.analysis_results])
            max_asymmetry = max(r['knee_symmetry'] for r in self.analysis_results)
            
            # Update metric displays
            self.metrics['max_deviation'].set(f"Max Spine-Shin Deviation: {max_deviation:.1f}°")
            self.metrics['avg_quality'].set(f"Average Movement Quality: {avg_quality:.1f}")
            self.metrics['symmetry'].set(f"Maximum Asymmetry: {max_asymmetry:.1f}°")
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
          
class HipHingeTab:
    """Tab for analyzing hip hinge movement data"""
    
    def __init__(self, notebook, data_manager):
        self.tab = ttk.Frame(notebook, padding="10")
        self.data_manager = data_manager
        
        # Analysis state
        self.motion_data = None
        self.analysis_complete = False
        self.results = {}
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Create tab layout
        self.create_tab_content()

    def create_tab_content(self):
        """Create the basic tab layout"""
        # Create main layout frames
        self.create_layout_frames()
        
        # Create matplotlib figures
        self.setup_visualization()
        
        # Create control panel
        self.create_controls()

    def create_layout_frames(self):
        """Create the main layout frames"""
        # Left control panel
        self.control_frame = ttk.Frame(self.tab, padding="5")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Right visualization panel
        self.viz_frame = ttk.Frame(self.tab)
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Configure grid for visualization area
        for i in range(2):
            self.viz_frame.grid_rowconfigure(i, weight=1)
            self.viz_frame.grid_columnconfigure(i, weight=1)

    def setup_visualization(self):
        """Set up matplotlib visualization areas"""
        # Create figures
        self.figs = {
            'alignment': Figure(figsize=(6, 4)),  # Head-pelvis alignment plot
            'coordination': Figure(figsize=(6, 4)),  # Phase plots
            'com': Figure(figsize=(6, 4)),  # COM trajectory
            'summary': Figure(figsize=(6, 4))  # Summary metrics
        }
        
        # Create axes
        self.axes = {}
        for key in self.figs:
            self.axes[key] = self.figs[key].add_subplot(111)
        
        # Create canvases
        self.canvases = {}
        for key, fig in self.figs.items():
            self.canvases[key] = FigureCanvasTkAgg(fig, self.viz_frame)
        
        # Grid layout
        self.canvases['alignment'].get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.canvases['coordination'].get_tk_widget().grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.canvases['com'].get_tk_widget().grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.canvases['summary'].get_tk_widget().grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

    def create_controls(self):
        """Create control panel elements"""
        # Data loading section
        load_frame = ttk.LabelFrame(self.control_frame, text="Data Import", padding="5")
        load_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            load_frame, 
            text="Load Motion Data",
            command=self.upload_data
        ).pack(fill=tk.X, pady=2)
        
        # Analysis parameters section
        self.create_metrics_display()
        
        # Analysis buttons
        self.analyze_btn = ttk.Button(
            self.control_frame,
            text="Analyze Movement",
            command=self.analyze_movement,
            state="disabled"
        )
        self.analyze_btn.pack(pady=5)
        
        self.export_btn = ttk.Button(
            self.control_frame,
            text="Export Results",
            command=self.export_results,
            state="disabled"
        )
        self.export_btn.pack(pady=5)

    def create_metrics_display(self):
        """Create metrics display area"""
        self.metrics_frame = ttk.LabelFrame(
            self.control_frame, 
            text="Movement Metrics",
            padding="5"
        )
        self.metrics_frame.pack(fill=tk.X, pady=5)
        
        # Initialize metric variables
        self.metrics = {
            'max_deviation': tk.StringVar(value="Max Head-Pelvis Deviation: --°"),
            'peak_hip_flexion': tk.StringVar(value="Peak Hip Flexion: --°"),
            'movement_symmetry': tk.StringVar(value="Movement Symmetry: --%"),
            'coordination_score': tk.StringVar(value="Coordination Score: --")
        }
        
        # Create metric labels
        for var in self.metrics.values():
            ttk.Label(self.metrics_frame, textvariable=var).pack(anchor=tk.W, pady=2)

    def upload_data(self):
        """Handle motion data file upload"""
        try:
            filetypes = [
                ("Motion data files", "*.json *.xlsx *.csv"),
                ("All files", "*.*")
            ]
            
            filename = filedialog.askopenfilename(
                title="Select Hip Hinge Motion Data",
                filetypes=filetypes
            )
            
            if not filename:
                return
                
            success, data = self.data_manager.validate_motion_data(filename)
            
            if not success:
                messagebox.showerror(
                    "Invalid Data",
                    "The selected file does not contain valid motion capture data."
                )
                return
                
            self.motion_data = data
            self.analyze_btn.configure(state="normal")
            self.clear_visualizations()
            
            messagebox.showinfo(
                "Success",
                "Motion data loaded successfully. Ready for analysis."
            )
            
        except Exception as e:
            self.logger.error(f"Error uploading data: {str(e)}")
            messagebox.showerror("Error", f"Failed to load motion data: {str(e)}")

    def analyze_movement(self):
        """Analyze hip hinge movement patterns"""
        if not self.motion_data:
            return
            
        try:
            # Calculate head-to-pelvis alignment throughout movement
            alignment_data = self.calculate_alignment()
            
            # Analyze COM trajectory
            com_data = self.analyze_com_trajectory()
            
            # Calculate movement velocity and acceleration
            velocity_data = self.calculate_movement_dynamics()
            
            # Calculate movement quality score
            quality_score = self.calculate_movement_quality(
                alignment_data, 
                com_data, 
                velocity_data
            )
            
            # Store results
            self.results = {
                'alignment': alignment_data,
                'com': com_data,
                'velocity': velocity_data,
                'quality': quality_score
            }
            
            self.analysis_complete = True
            self.update_metrics()
            self.update_visualization()
            self.export_btn.configure(state="normal")
            
        except Exception as e:
            self.logger.error(f"Error in movement analysis: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def calculate_movement_quality(self, alignment_data, com_data, velocity_data):
        """Calculate overall movement quality score"""
        try:
            # Initialize quality components
            quality_scores = {
                'alignment': 0,
                'stability': 0,
                'smoothness': 0,
                'total_score': 0
            }
            
            # Score alignment (40% of total)
            max_allowed_deviation = 15.0  # degrees
            alignment_score = max(0, 100 - (
                alignment_data['mean_deviation'] / max_allowed_deviation * 100
            ))
            quality_scores['alignment'] = alignment_score * 0.4
            
            # Score stability (40% of total)
            max_allowed_com_deviation = 50.0  # mm
            stability_score = max(0, 100 - (
                com_data['mean_deviation'] / max_allowed_com_deviation * 100
            ))
            quality_scores['stability'] = stability_score * 0.4
            
            # Score movement smoothness (20% of total)
            smoothness_score = max(0, 100 - (
                np.std(velocity_data['velocity']) / 0.5 * 100
            ))
            quality_scores['smoothness'] = smoothness_score * 0.2
            
            # Calculate total score
            quality_scores['total_score'] = sum([
                quality_scores['alignment'],
                quality_scores['stability'],
                quality_scores['smoothness']
            ])
            
            return quality_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating movement quality: {str(e)}")
            return None

    def calculate_alignment(self):
        """Calculate head-to-pelvis alignment throughout movement"""
        try:
            frames = self.motion_data['frames']
            alignment_data = {
                'angles': [],
                'deviations': [],
                'max_deviation': 0,
                'mean_deviation': 0
            }
            
            for frame in frames:
                # Calculate head position
                head_pos = np.array([
                    frame['keypoints3D']['nose']['x'],
                    frame['keypoints3D']['nose']['y'],
                    frame['keypoints3D']['nose']['z']
                ])
                
                # Calculate mid-pelvis position
                left_hip = np.array([
                    frame['keypoints3D']['leftHip']['x'],
                    frame['keypoints3D']['leftHip']['y'],
                    frame['keypoints3D']['leftHip']['z']
                ])
                right_hip = np.array([
                    frame['keypoints3D']['rightHip']['x'],
                    frame['keypoints3D']['rightHip']['y'],
                    frame['keypoints3D']['rightHip']['z']
                ])
                mid_pelvis = (left_hip + right_hip) / 2
                
                # Calculate alignment vector
                alignment_vector = head_pos - mid_pelvis
                
                # Calculate angle with vertical
                vertical = np.array([0, 0, 1])
                angle = np.arccos(
                    np.dot(alignment_vector, vertical) /
                    (np.linalg.norm(alignment_vector) * np.linalg.norm(vertical))
                )
                angle_deg = np.degrees(angle)
                
                # Calculate deviation from ideal alignment (should be near vertical)
                deviation = abs(angle_deg - 90)
                
                alignment_data['angles'].append(angle_deg)
                alignment_data['deviations'].append(deviation)
            
            # Calculate summary statistics
            alignment_data['max_deviation'] = max(alignment_data['deviations'])
            alignment_data['mean_deviation'] = np.mean(alignment_data['deviations'])
            
            return alignment_data
            
        except Exception as e:
            self.logger.error(f"Error calculating alignment: {str(e)}")
            return None

    def analyze_com_trajectory(self):
        """Analyze center of mass trajectory during hip hinge"""
        try:
            frames = self.motion_data['frames']
            com_data = {
                'trajectory': [],
                'vertical_displacement': [],
                'horizontal_displacement': [],
                'max_deviation': 0,
                'mean_deviation': 0
            }
            
            # Get initial COM position as reference
            initial_com = np.array([
                frames[0]['com2D']['x'],
                frames[0]['com2D']['y']
            ])
            
            for frame in frames:
                current_com = np.array([
                    frame['com2D']['x'],
                    frame['com2D']['y']
                ])
                
                # Calculate displacements
                displacement = current_com - initial_com
                vertical_disp = displacement[1]
                horizontal_disp = displacement[0]
                
                com_data['trajectory'].append(current_com)
                com_data['vertical_displacement'].append(vertical_disp)
                com_data['horizontal_displacement'].append(horizontal_disp)
            
            # Convert lists to numpy arrays
            com_data['trajectory'] = np.array(com_data['trajectory'])
            com_data['vertical_displacement'] = np.array(com_data['vertical_displacement'])
            com_data['horizontal_displacement'] = np.array(com_data['horizontal_displacement'])
            
            # Calculate maximum deviations
            com_data['max_deviation'] = max(abs(com_data['horizontal_displacement']))
            com_data['mean_deviation'] = np.mean(abs(com_data['horizontal_displacement']))
            
            return com_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing COM trajectory: {str(e)}")
            return None
    def calculate_coordination(self):
        """Calculate coordination patterns using CRP"""
        try:
            # Get hip and trunk angles
            hip_angles = self.data_manager.get_angle_data('leftHipAngle')
            trunk_angles = self.data_manager.get_angle_data('trunkAngle')
            
            if hip_angles is None or trunk_angles is None:
                raise ValueError("Required angle data not found")
            
            # Calculate phase angles
            hip_phase = self.calculate_phase_angle(hip_angles)
            trunk_phase = self.calculate_phase_angle(trunk_angles)
            
            # Calculate CRP
            crp = np.abs(hip_phase - trunk_phase)
            crp = np.where(crp > 180, 360 - crp, crp)
            
            return crp
            
        except Exception as e:
            self.logger.error(f"Error calculating coordination: {str(e)}")
            return None

    def calculate_phase_angle(self, signal):
        """Calculate phase angle using Hilbert transform"""
        try:
            analytic_signal = hilbert(signal)
            phase = np.angle(analytic_signal, deg=True)
            return phase
            
        except Exception as e:
            self.logger.error(f"Error calculating phase angle: {str(e)}")
            return None

    def update_metrics(self):
        """Update displayed metrics based on analysis results"""
        try:
            if not self.analysis_complete:
                return
                
            # Update alignment metrics
            self.metrics['max_deviation'].set(
                f"Max Head-Pelvis Deviation: {self.results['alignment']['max_deviation']:.1f}°"
            )
            
            # Update COM metrics
            self.metrics['com_stability'].set(
                f"COM Stability: {100 - self.results['com']['mean_deviation']:.1f}%"
            )
            
            # Update velocity metrics
            self.metrics['movement_smoothness'].set(
                f"Movement Smoothness: {self.results['quality']['smoothness']:.1f}"
            )
            
            # Update overall quality score
            self.metrics['quality_score'].set(
                f"Movement Quality: {self.results['quality']['total_score']:.1f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")

    def update_visualization(self):
        """Update all visualization plots"""
        try:
            if not self.analysis_complete:
                return
                
            # Clear all plots
            for ax in self.axes.values():
                ax.clear()
            
            # Plot head-to-pelvis alignment
            self.plot_alignment()
            
            # Plot COM trajectory
            self.plot_com_path()
            
            # Plot movement velocity
            self.plot_velocity()
            
            # Plot quality metrics
            self.plot_quality_metrics()
            
            # Refresh all canvases
            for canvas in self.canvases.values():
                canvas.draw()
                
        except Exception as e:
            self.logger.error(f"Error updating visualization: {str(e)}")


    def plot_alignment(self):
        """Plot head-to-pelvis alignment analysis"""
        try:
            ax = self.axes['alignment']
            
            # Create time axis (0-100% of movement)
            time = np.linspace(0, 100, len(self.results['alignment']['angles']))
            
            # Plot alignment angle
            ax.plot(time, self.results['alignment']['angles'], 
                    'b-', linewidth=2, label='Head-Pelvis Angle')
            
            # Add threshold lines
            ax.axhline(y=90, color='g', linestyle='--', alpha=0.5, 
                       label='Ideal Alignment')
            ax.axhline(y=75, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=105, color='r', linestyle='--', alpha=0.5)
            
            # Configure plot
            ax.set_title('Head-to-Pelvis Alignment')
            ax.set_xlabel('Movement Completion (%)')
            ax.set_ylabel('Angle (degrees)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        except Exception as e:
            self.logger.error(f"Error plotting alignment: {str(e)}")

    def plot_com_path(self):
        """Plot center of mass trajectory"""
        try:
            ax = self.axes['com_path']
            
            # Plot COM trajectory
            trajectory = self.results['com']['trajectory']
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                    'b-', linewidth=2, label='COM Path')
            
            # Plot start and end points
            ax.plot(trajectory[0, 0], trajectory[0, 1], 
                    'go', markersize=10, label='Start')
            ax.plot(trajectory[-1, 0], trajectory[-1, 1], 
                    'ro', markersize=10, label='End')
            
            # Configure plot
            ax.set_title('Center of Mass Trajectory')
            ax.set_xlabel('Horizontal Position (mm)')
            ax.set_ylabel('Vertical Position (mm)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.axis('equal')
            
        except Exception as e:
            self.logger.error(f"Error plotting COM path: {str(e)}")
    
    def plot_coordination(self):
        """Plot coordination phase relationship"""
        ax = self.axes['coordination']
        time = np.arange(len(self.results['coordination']))
        ax.plot(time, self.results['coordination'], 'r-', label='CRP')
        ax.set_title('Hip-Trunk Coordination')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Relative Phase (degrees)')
        ax.grid(True)
        ax.legend()

    def plot_com_trajectory(self):
        """Plot COM trajectory"""
        ax = self.axes['com']
        com_data = self.results['com']
        ax.plot(com_data[:, 1], com_data[:, 2], 'k-', label='COM Path')
        ax.set_title('Center of Mass Trajectory')
        ax.set_xlabel('Anterior-Posterior (mm)')
        ax.set_ylabel('Vertical (mm)')
        ax.grid(True)
        ax.legend()

    def plot_summary(self):
        """Plot summary metrics"""
        ax = self.axes['summary']
        metrics = [
            np.max(np.abs(self.results['alignment'])),
            np.mean(self.results['coordination']),
            np.std(self.results['coordination'])
        ]
        labels = ['Max Deviation', 'Mean CRP', 'CRP Variability']
        ax.bar(labels, metrics)
        ax.set_title('Summary Metrics')
        ax.set_ylabel('Value')
        plt.setp(ax.get_xticklabels(), rotation=45)

    def plot_quality_metrics(self):
        """Plot movement quality metrics"""
        try:
            ax = self.axes['quality']
            
            # Prepare data
            metrics = [
                self.results['quality']['alignment'],
                self.results['quality']['stability'],
                self.results['quality']['smoothness']
            ]
            labels = ['Alignment', 'Stability', 'Smoothness']
            
            # Create bar plot
            x_pos = np.arange(len(labels))
            ax.bar(x_pos, metrics, align='center', alpha=0.8)
            
            # Configure plot
            ax.set_title('Movement Quality Analysis')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Score')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            
            # Add score labels
            for i, v in enumerate(metrics):
                ax.text(i, v + 1, f'{v:.1f}', 
                        ha='center', va='bottom')
            
        except Exception as e:
            self.logger.error(f"Error plotting quality metrics: {str(e)}")

    def clear_visualizations(self):
        """Clear all visualization plots"""
        for ax in self.axes.values():
            ax.clear()
        for canvas in self.canvases.values():
            canvas.draw()

    def export_results(self):
        """Export analysis results"""
        if not self.analysis_complete:
            messagebox.showwarning(
                "No Results",
                "Please analyze the movement before exporting results."
            )
            return
            
        try:
            filename = filedialog.asksaveasfilename(
                title="Export Analysis Results",
                defaultextension=".xlsx",
                filetypes=[
                    ("Excel files", "*.xlsx"),
                    ("CSV files", "*.csv")
                ]
            )
            
            if not filename:
                return
                
            # Prepare results for export
            results_df = pd.DataFrame({
                'Frame': range(len(self.results['alignment'])),
                'Head_Pelvis_Alignment': self.results['alignment'],
                'Coordination_CRP': self.results['coordination'],
                'COM_X': self.results['com'][:, 0],
                'COM_Y': self.results['com'][:, 1],
                'COM_Z': self.results['com'][:, 2]
            })
            
            # Add summary metrics
            summary_df = pd.DataFrame({
                'Metric': [
                    'Maximum Head-Pelvis Deviation',
                    'Mean Coordination Phase',
                    'Phase Variability',
                    'Movement Quality Score'
                ],
                'Value': [
                    np.max(np.abs(self.results['alignment'])),
                    np.mean(self.results['coordination']),
                    np.std(self.results['coordination']),
                    self.calculate_quality_score()
                ]
            })
            
            # Export with multiple sheets if Excel
            if filename.endswith('.xlsx'):
                with pd.ExcelWriter(filename) as writer:
                    results_df.to_excel(writer, sheet_name='Detailed Data', index=False)
                    summary_df.to_excel(writer, sheet_name='Summary Metrics', index=False)
            else:
                results_df.to_csv(filename, index=False)
                summary_df.to_csv(filename.replace('.csv', '_summary.csv'), index=False)
                
            messagebox.showinfo(
                "Success",
                "Analysis results exported successfully."
            )
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            messagebox.showerror(
                "Error",
                f"Failed to export results: {str(e)}"
            )

    def calculate_quality_score(self):
        """Calculate overall movement quality score"""
        try:
            # Calculate component scores
            alignment_score = 100 - np.mean(np.abs(self.results['alignment']))
            coordination_score = 100 - np.mean(self.results['coordination'])
            variability_score = 100 - np.std(self.results['coordination'])
            
            # Weight and combine scores
            weights = [0.4, 0.4, 0.2]  # Emphasis on alignment and coordination
            total_score = (
                weights[0] * alignment_score +
                weights[1] * coordination_score +
                weights[2] * variability_score
            )
            
            return np.clip(total_score, 0, 100)
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {str(e)}")
            return 0

    def on_closing(self):
        """Handle cleanup when tab is closed"""
        try:
            # Clear matplotlib figures
            for fig in self.figs.values():
                plt.close(fig)
                
            # Clear data
            self.motion_data = None
            self.results = {}
            self.analysis_complete = False
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    def reset_analysis(self):
        """Reset the analysis state and clear visualizations"""
        try:
            # Reset state
            self.motion_data = None
            self.results = {}
            self.analysis_complete = False
            
            # Reset UI
            self.analyze_btn.configure(state="disabled")
            self.export_btn.configure(state="disabled")
            
            # Reset metrics
            for var in self.metrics.values():
                var.set(var.get().split(':')[0] + ": --")
                
            # Clear visualizations
            self.clear_visualizations()
            
            messagebox.showinfo(
                "Reset Complete",
                "Analysis has been reset. Please load new data to continue."
            )
            
        except Exception as e:
            self.logger.error(f"Error resetting analysis: {str(e)}")
            messagebox.showerror(
                "Error",
                f"Failed to reset analysis: {str(e)}"
            )

class WalkingGaitTab:
    """Tab for analyzing walking gait coordination patterns"""
    
    def __init__(self, notebook, data_manager):
        self.tab = ttk.Frame(notebook, padding="10")
        self.data_manager = data_manager
        
        # Analysis state
        self.motion_data = None
        self.analysis_complete = False
        self.results = {}
        
        # Initialize analysis parameters
        self.params = {
            'stride_threshold': 0.1,  # For detecting gait events
            'filter_cutoff': 6.0,  # Hz for smoothing
            'phase_bins': 20  # For coordination histograms
        }
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Create tab layout
        self.create_tab_content()

    def create_tab_content(self):
        """Create the tab layout and controls"""
        # Configure grid
        self.tab.grid_columnconfigure(1, weight=1)
        self.tab.grid_rowconfigure(0, weight=1)
        
        # Create control panel
        self.control_frame = ttk.Frame(self.tab, padding="5")
        self.control_frame.grid(row=0, column=0, sticky="ns", padx=5)
        
        # Create visualization area
        self.viz_frame = ttk.Frame(self.tab)
        self.viz_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        
        # Setup visualization plots
        self.setup_visualization()
        
        # Create controls
        self.create_controls()

    def setup_visualization(self):
        """Setup matplotlib visualization areas"""
        # Create figures for different analyses
        self.figs = {
            'phase_plot': Figure(figsize=(6, 4)),  # Relative phase plot
            'coordination': Figure(figsize=(6, 4)),  # Coordination patterns
            'symmetry': Figure(figsize=(6, 4)),  # Bilateral symmetry
            'variability': Figure(figsize=(6, 4))  # Coordination variability
        }
        
        # Create axes
        self.axes = {}
        for key in self.figs:
            self.axes[key] = self.figs[key].add_subplot(111)
        
        # Create canvases
        self.canvases = {}
        for key, fig in self.figs.items():
            self.canvases[key] = FigureCanvasTkAgg(fig, self.viz_frame)
            
        # Grid layout
        for i, (key, canvas) in enumerate(self.canvases.items()):
            row, col = divmod(i, 2)
            canvas.get_tk_widget().grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
        # Configure grid
        self.viz_frame.grid_columnconfigure(0, weight=1)
        self.viz_frame.grid_columnconfigure(1, weight=1)
        self.viz_frame.grid_rowconfigure(0, weight=1)
        self.viz_frame.grid_rowconfigure(1, weight=1)

    def create_controls(self):
        """Create control panel elements"""
        # Data loading section
        load_frame = ttk.LabelFrame(self.control_frame, text="Data Import", padding="5")
        load_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            load_frame,
            text="Load Gait Data",
            command=self.load_motion_data
        ).pack(fill=tk.X, pady=2)
        
        # Analysis parameters
        param_frame = ttk.LabelFrame(self.control_frame, text="Analysis Parameters", padding="5")
        param_frame.pack(fill=tk.X, pady=5)
        
        # Parameter controls
        for param, value in self.params.items():
            frame = ttk.Frame(param_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(
                frame,
                text=param.replace('_', ' ').title()
            ).pack(side=tk.LEFT)
            
            var = tk.DoubleVar(value=value)
            ttk.Entry(
                frame,
                textvariable=var,
                width=10
            ).pack(side=tk.RIGHT)
            
            setattr(self, f"{param}_var", var)
        
        # Metrics display
        self.create_metrics_display()
        
        # Analysis buttons
        self.analyze_btn = ttk.Button(
            self.control_frame,
            text="Analyze Gait",
            command=self.analyze_gait,
            state="disabled"
        )
        self.analyze_btn.pack(pady=5)
        
        self.export_btn = ttk.Button(
            self.control_frame,
            text="Export Results",
            command=self.export_results,
            state="disabled"
        )
        self.export_btn.pack(pady=5)

    def create_metrics_display(self):
        """Create metrics display area"""
        metrics_frame = ttk.LabelFrame(
            self.control_frame,
            text="Gait Metrics",
            padding="5"
        )
        metrics_frame.pack(fill=tk.X, pady=5)
        
        # Initialize metric variables
        self.metrics = {
            'mean_crp': tk.StringVar(value="Mean CRP: --°"),
            'crp_variability': tk.StringVar(value="CRP Variability: --°"),
            'symmetry_index': tk.StringVar(value="Symmetry Index: --%"),
            'coordination_stability': tk.StringVar(value="Coordination Stability: --")
        }
        
        # Create metric labels
        for var in self.metrics.values():
            ttk.Label(metrics_frame, textvariable=var).pack(anchor=tk.W, pady=2)

    def load_motion_data(self):
        """Load and validate gait motion data"""
        try:
            filename = filedialog.askopenfilename(
                title="Select Gait Motion Data",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("Excel files", "*.xlsx"),
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ]
            )
            
            if not filename:
                return
                
            success, data = self.data_manager.validate_motion_data(filename)
            
            if not success:
                messagebox.showerror(
                    "Invalid Data",
                    "The selected file does not contain valid gait motion data."
                )
                return
                
            self.motion_data = data
            self.analyze_btn.configure(state="normal")
            self.clear_visualizations()
            
            messagebox.showinfo(
                "Success",
                "Gait data loaded successfully. Ready for analysis."
            )
            
        except Exception as e:
            self.logger.error(f"Error loading gait data: {str(e)}")
            messagebox.showerror("Error", f"Failed to load gait data: {str(e)}")

    def analyze_gait(self):
        """Perform gait analysis focusing on coordination patterns"""
        if not self.motion_data:
            return
            
        try:
            # Calculate intralimb coordination (within limb)
            hip_knee_crp = self.calculate_intralimb_coordination('right')
            knee_ankle_crp = self.calculate_intralimb_coordination('left')
            
            # Calculate interlimb coordination (between limbs)
            bilateral_crp = self.calculate_interlimb_coordination()
            
            # Calculate coordination variability
            variability = self.calculate_coordination_variability(
                [hip_knee_crp, knee_ankle_crp, bilateral_crp]
            )
            
            # Store results
            self.results = {
                'hip_knee_crp': hip_knee_crp,
                'knee_ankle_crp': knee_ankle_crp,
                'bilateral_crp': bilateral_crp,
                'variability': variability
            }
            
            self.analysis_complete = True
            self.update_metrics()
            self.update_visualization()
            self.export_btn.configure(state="normal")
            
        except Exception as e:
            self.logger.error(f"Error in gait analysis: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def calculate_intralimb_coordination(self, side):
        """Calculate continuous relative phase for joints within a limb"""
        try:
            # Get joint angle data
            hip_angles = self.data_manager.get_angle_data(f'{side}HipAngle')
            knee_angles = self.data_manager.get_angle_data(f'{side}KneeAngle')
            
            if hip_angles is None or knee_angles is None:
                raise ValueError("Required joint angle data not found")
            
            # Calculate phase angles
            hip_phase = self.calculate_phase_angle(hip_angles)
            knee_phase = self.calculate_phase_angle(knee_angles)
            
            # Calculate CRP
            crp = np.abs(hip_phase - knee_phase)
            crp = np.where(crp > 180, 360 - crp, crp)
            
            return crp
            
        except Exception as e:
            self.logger.error(f"Error calculating intralimb coordination: {str(e)}")
            return None

    def calculate_interlimb_coordination(self):
        """Calculate continuous relative phase between left and right limbs"""
        try:
            # Get bilateral joint angles
            right_hip = self.data_manager.get_angle_data('rightHipAngle')
            left_hip = self.data_manager.get_angle_data('leftHipAngle')
            
            if right_hip is None or left_hip is None:
                raise ValueError("Required bilateral data not found")
            
            # Calculate phase angles
            right_phase = self.calculate_phase_angle(right_hip)
            left_phase = self.calculate_phase_angle(left_hip)
            
            # Calculate CRP
            crp = np.abs(right_phase - left_phase)
            crp = np.where(crp > 180, 360 - crp, crp)
            
            return crp
            
        except Exception as e:
            self.logger.error(f"Error calculating interlimb coordination: {str(e)}")
            return None

    def calculate_phase_angle(self, signal):
        """Calculate phase angle using Hilbert transform"""
        try:
            analytic_signal = hilbert(signal)
            phase = np.angle(analytic_signal, deg=True)
            return phase
            
        except Exception as e:
            self.logger.error(f"Error calculating phase angle: {str(e)}")
            return None

    def calculate_coordination_variability(self, crp_signals):
        """Calculate variability in coordination patterns"""
        try:
            variability = []
            for crp in crp_signals:
                if crp is not None:
                    # Calculate circular standard deviation
                    var = np.std(crp)
                    variability.append(var)
            
            return np.mean(variability) if variability else None
            
        except Exception as e:
            self.logger.error(f"Error calculating coordination variability: {str(e)}")
            return None

    def update_metrics(self):
        """Update displayed metrics"""
        try:
            if not self.analysis_complete:
                return
                
            # Calculate summary metrics
            mean_crp = np.mean([
                np.mean(self.results['hip_knee_crp']),
                np.mean(self.results['knee_ankle_crp'])
            ])
            
            variability = self.results['variability']
            symmetry = 100 * (1 - np.mean(self.results['bilateral_crp']) / 180)
            stability = 100 * (1 - variability / 180)
            
            # Update metric displays
            self.metrics['mean_crp'].set(f"Mean CRP: {mean_crp:.1f}°")
            self.metrics['crp_variability'].set(f"CRP Variability: {variability:.1f}°")
            self.metrics['symmetry_index'].set(f"Symmetry Index: {symmetry:.1f}%")
            self.metrics['coordination_stability'].set(f"Coordination Stability: {stability:.1f}")
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")

    def update_visualization(self):
        """Update all visualization plots"""
        try:
            if not self.analysis_complete:
                return
                
            # Clear all plots
            for ax in self.axes.values():
                ax.clear()
            
            # Plot phase relationships
            self.plot_phase_relationships()
            
            # Plot coordination patterns
            self.plot_coordination_patterns()
            
            # Plot symmetry
            self.plot_symmetry()
            
            # Plot variability
            self.plot_variability()
            
            # Refresh all canvases
            for canvas in self.canvases.values():
                canvas.draw()
                
        except Exception as e:
            self.logger.error(f"Error updating visualization: {str(e)}")

    def plot_phase_relationships(self):
        """Plot continuous relative phase relationships"""
        ax = self.axes['phase_plot']
        time = np.arange(len(self.results['hip_knee_crp']))
        
        ax.plot(time, self.results['hip_knee_crp'], 'b-', label='Hip-Knee')
        ax.plot(time, self.results['knee_ankle_crp'], 'r-', label='Knee-Ankle')
        
        ax.set_title('Joint Coordination Patterns')
        ax.set_xlabel('% Gait Cycle')
        ax.set_ylabel('Relative Phase (degrees)')
        ax.grid(True)
        ax.legend()

    def plot_coordination_patterns(self):
        """Plot coordination pattern distribution"""
        ax = self.axes['coordination']
        
        # Create histogram of coordination patterns
        bins = np.linspace(0, 180, self.params['phase_bins'])
        ax.hist(self.results['hip_knee_crp'], bins, alpha=0.5, label='Hip-Knee')
        ax.hist(self.results['knee_ankle_crp'], bins, alpha=0.5, label='Knee-Ankle')
        
        ax.set_title('Coordination Pattern Distribution')
        ax.set_xlabel('Relative Phase (degrees)')
        ax.set_ylabel('Frequency')
        ax.legend()

    def plot_symmetry(self):
        """Plot bilateral symmetry analysis"""
        ax = self.axes['symmetry']
        time = np.arange(len(self.results['bilateral_crp']))
        
        ax.plot(time, self.results['bilateral_crp'], 'k-')
        ax.set_title('Bilateral Symmetry')
        ax.set_xlabel('% Gait Cycle')
        ax.set_ylabel('Relative Phase (degrees)')
        ax.grid(True)

    def plot_variability(self):
        """Plot coordination variability across gait cycles"""
        try:
            ax = self.axes['variability']
            
            # Calculate variability metrics
            variability_metrics = [
                np.std(self.results['hip_knee_crp']),
                np.std(self.results['knee_ankle_crp']),
                np.std(self.results['bilateral_crp'])
            ]
            
            # Define labels and positions for bar plot
            labels = ['Hip-Knee', 'Knee-Ankle', 'Bilateral']
            x_pos = np.arange(len(labels))
            
            # Create bar plot
            bars = ax.bar(x_pos, variability_metrics, 
                         align='center', 
                         alpha=0.8,
                         color=['lightblue', 'lightgreen', 'salmon'])
            
            # Customize plot appearance
            ax.set_title('Coordination Variability')
            ax.set_xlabel('Joint Coupling')
            ax.set_ylabel('Standard Deviation (degrees)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}°',
                       ha='center', va='bottom')
            
            # Add grid for better readability
            ax.grid(True, axis='y', alpha=0.3)
            
            # Adjust layout to prevent label cutoff
            self.figs['variability'].tight_layout()
            
        except Exception as e:
            self.logger.error(f"Error plotting variability: {str(e)}")
            messagebox.showerror("Error", "Failed to plot coordination variability")

    def export_results(self):
        """Export analysis results to Excel file"""
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                initialfile=f"gait_analysis_{timestamp}.xlsx"
            )
            
            if not filename:
                return
                
            # Create Excel writer
            with pd.ExcelWriter(filename) as writer:
                # Export CRP data
                crp_data = pd.DataFrame({
                    'Hip_Knee_CRP': self.results['hip_knee_crp'],
                    'Knee_Ankle_CRP': self.results['knee_ankle_crp'],
                    'Bilateral_CRP': self.results['bilateral_crp']
                })
                crp_data.to_excel(writer, sheet_name='CRP_Data', index=False)
                
                # Export summary metrics
                metrics_data = pd.DataFrame({
                    'Metric': list(self.metrics.keys()),
                    'Value': [var.get() for var in self.metrics.values()]
                })
                metrics_data.to_excel(writer, sheet_name='Summary_Metrics', index=False)
                
            messagebox.showinfo(
                "Export Complete",
                f"Analysis results have been exported to:\n{filename}"
            )
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            messagebox.showerror("Error", "Failed to export analysis results")

    def clear_visualizations(self):
        """Clear all visualization plots"""
        try:
            for ax in self.axes.values():
                ax.clear()
            
            for canvas in self.canvases.values():
                canvas.draw()
                
            # Reset metrics display
            for var in self.metrics.values():
                var.set(var.get().split(':')[0] + ': --')
                
        except Exception as e:
            self.logger.error(f"Error clearing visualizations: {str(e)}")

class FukudaStepTab:
    """Handles analysis of the Fukuda Step test data"""
    
    def __init__(self, notebook, data_manager):
        self.tab = ttk.Frame(notebook, padding="10")
        self.data_manager = data_manager
        
        # Analysis state
        self.motion_data = None
        self.analysis_complete = False
        self.results = {}
        
        # Analysis parameters
        self.params = {
            'rotation_threshold': 15.0,  # Degrees of rotation considered significant
            'com_deviation_threshold': 50.0,  # mm of lateral deviation considered significant
            'step_detection_threshold': 0.1  # For detecting individual steps
        }
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Create tab layout
        self.create_tab_content()

    def create_tab_content(self):
        """Create the tab layout and controls"""
        # Configure grid
        self.tab.grid_columnconfigure(1, weight=1)
        self.tab.grid_rowconfigure(0, weight=1)
        
        # Create frames
        self.control_frame = ttk.Frame(self.tab, padding="5")
        self.control_frame.grid(row=0, column=0, sticky="ns", padx=5)
        
        self.viz_frame = ttk.Frame(self.tab)
        self.viz_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        
        # Setup visualization plots
        self.setup_visualization()
        
        # Create controls
        self.create_controls()

    def setup_visualization(self):
        """Setup matplotlib visualization areas"""
        self.figs = {
            'top_view': Figure(figsize=(6, 4)),  # Movement trajectory from above
            'com_path': Figure(figsize=(6, 4)),  # Center of mass path
            'rotation': Figure(figsize=(6, 4)),  # Rotation over time
            'step_analysis': Figure(figsize=(6, 4))  # Step-by-step metrics
        }
        
        self.axes = {}
        for key in self.figs:
            self.axes[key] = self.figs[key].add_subplot(111)
            
        self.canvases = {}
        for key, fig in self.figs.items():
            self.canvases[key] = FigureCanvasTkAgg(fig, self.viz_frame)
            
        # Grid layout for visualization
        for i, (key, canvas) in enumerate(self.canvases.items()):
            row, col = divmod(i, 2)
            canvas.get_tk_widget().grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
        # Configure grid
        self.viz_frame.grid_columnconfigure(0, weight=1)
        self.viz_frame.grid_columnconfigure(1, weight=1)
        self.viz_frame.grid_rowconfigure(0, weight=1)
        self.viz_frame.grid_rowconfigure(1, weight=1)

    def create_controls(self):
        """Create control panel elements"""
        # Data loading section
        load_frame = ttk.LabelFrame(self.control_frame, text="Data Import", padding="5")
        load_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            load_frame,
            text="Load Test Data",
            command=self.load_motion_data
        ).pack(fill=tk.X, pady=2)
        
        # Analysis parameters
        param_frame = ttk.LabelFrame(self.control_frame, text="Analysis Parameters", padding="5")
        param_frame.pack(fill=tk.X, pady=5)
        
        for param, value in self.params.items():
            frame = ttk.Frame(param_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(
                frame,
                text=param.replace('_', ' ').title()
            ).pack(side=tk.LEFT)
            
            var = tk.DoubleVar(value=value)
            ttk.Entry(
                frame,
                textvariable=var,
                width=10
            ).pack(side=tk.RIGHT)
            
            setattr(self, f"{param}_var", var)
        
        # Create metrics display
        self.create_metrics_display()
        
        # Analysis buttons
        self.analyze_btn = ttk.Button(
            self.control_frame,
            text="Analyze Test",
            command=self.analyze_test,
            state="disabled"
        )
        self.analyze_btn.pack(pady=5)
        
        self.export_btn = ttk.Button(
            self.control_frame,
            text="Export Results",
            command=self.export_results,
            state="disabled"
        )
        self.export_btn.pack(pady=5)

    def create_metrics_display(self):
        """Create metrics display area"""
        metrics_frame = ttk.LabelFrame(
            self.control_frame,
            text="Test Metrics",
            padding="5"
        )
        metrics_frame.pack(fill=tk.X, pady=5)
        
        # Initialize metric variables
        self.metrics = {
            'total_rotation': tk.StringVar(value="Total Rotation: --°"),
            'lateral_deviation': tk.StringVar(value="Lateral Deviation: -- mm"),
            'step_symmetry': tk.StringVar(value="Step Symmetry: --%"),
            'stability_score': tk.StringVar(value="Stability Score: --")
        }
        
        # Create metric labels
        for var in self.metrics.values():
            ttk.Label(metrics_frame, textvariable=var).pack(anchor=tk.W, pady=2)

    def load_motion_data(self):
        """Load and validate Fukuda test data"""
        try:
            filename = filedialog.askopenfilename(
                title="Select Fukuda Test Data",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("Excel files", "*.xlsx"),
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ]
            )
            
            if not filename:
                return
                
            success, data = self.data_manager.validate_motion_data(filename)
            
            if not success:
                messagebox.showerror(
                    "Invalid Data",
                    "The selected file does not contain valid Fukuda test data."
                )
                return
                
            self.motion_data = data
            self.analyze_btn.configure(state="normal")
            self.clear_visualizations()
            
            messagebox.showinfo(
                "Success",
                "Test data loaded successfully. Ready for analysis."
            )
            
        except Exception as e:
            self.logger.error(f"Error loading test data: {str(e)}")
            messagebox.showerror("Error", f"Failed to load test data: {str(e)}")

    def analyze_test(self):
        """Perform Fukuda Step test analysis"""
        if not self.motion_data:
            return
            
        try:
            # Analyze rotation
            rotation_data = self.analyze_rotation()
            
            # Analyze COM trajectory
            com_data = self.analyze_com_trajectory()
            
            # Analyze step patterns
            step_data = self.analyze_step_patterns()
            
            # Calculate stability metrics
            stability_metrics = self.calculate_stability_metrics(
                rotation_data, com_data, step_data
            )
            
            # Store results
            self.results = {
                'rotation': rotation_data,
                'com': com_data,
                'steps': step_data,
                'stability': stability_metrics
            }
            
            self.analysis_complete = True
            self.update_metrics()
            self.update_visualization()
            self.export_btn.configure(state="normal")
            
        except Exception as e:
            self.logger.error(f"Error in test analysis: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def analyze_rotation(self):
        """Analyze body rotation during test"""
        try:
            # Get shoulder orientation data
            shoulder_points = []
            for frame in self.motion_data['frames']:
                left = frame['keypoints3D']['leftShoulder']
                right = frame['keypoints3D']['rightShoulder']
                shoulder_points.append([left, right])
                
            shoulder_points = np.array(shoulder_points)
            
            # Calculate rotation angles relative to start
            angles = []
            reference = shoulder_points[0]
            for points in shoulder_points:
                angle = self.calculate_angle_change(reference, points)
                angles.append(angle)
                
            return {
                'angles': np.array(angles),
                'total_rotation': np.abs(angles[-1]),
                'max_rotation': np.max(np.abs(angles))
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing rotation: {str(e)}")
            return None

    def analyze_com_trajectory(self):
        """Analyze center of mass trajectory"""
        try:
            frames = self.sagittal_data['frames']
            com_positions = []
            
            # Get initial COM position as reference
            initial_com = frames[0]['com2D']
            ref_x = initial_com['x']
            ref_y = initial_com['y']
            
            for frame in frames:
                com = frame['com2D']
                # Calculate deviation from reference
                deviation = math.sqrt(
                    (com['x'] - ref_x)**2 + 
                    (com['y'] - ref_y)**2
                )
                com_positions.append({
                    'x': com['x'],
                    'y': com['y'],
                    'deviation': deviation
                })
            
            self.results['com'].update({
                'trajectory': com_positions,
                'max_deviation': max(p['deviation'] for p in com_positions),
                'mean_deviation': np.mean([p['deviation'] for p in com_positions])
            })
            
        except Exception as e:
            self.logger.error(f"Error analyzing COM trajectory: {str(e)}")
            raise

    def analyze_step_patterns(self):
        """Analyze individual step patterns"""
        try:
            # Get ankle vertical positions
            right_ankle = []
            left_ankle = []
            
            for frame in self.motion_data['frames']:
                right_ankle.append(frame['keypoints3D']['rightAnkle'][2])
                left_ankle.append(frame['keypoints3D']['leftAnkle'][2])
                
            # Detect steps using height threshold
            right_steps = self.detect_steps(np.array(right_ankle))
            left_steps = self.detect_steps(np.array(left_ankle))
            
            # Calculate step metrics
            step_lengths = self.calculate_step_lengths(right_steps, left_steps)
            step_durations = self.calculate_step_durations(right_steps, left_steps)
            
            return {
                'right_steps': right_steps,
                'left_steps': left_steps,
                'step_lengths': step_lengths,
                'step_durations': step_durations
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing step patterns: {str(e)}")
            return None

    def calculate_stability_metrics(self, rotation_data, com_data, step_data):
        """Calculate overall stability metrics"""
        try:
            # Calculate rotation stability
            rotation_stability = 100 * (1 - rotation_data['total_rotation'] / 360)
            
            # Calculate COM stability
            com_stability = 100 * (1 - com_data['lateral_deviation'] / self.params['com_deviation_threshold'])
            
            # Calculate step symmetry
            step_lengths = step_data['step_lengths']
            symmetry = 100 * (1 - np.std(step_lengths) / np.mean(step_lengths))
            
            # Calculate overall stability score
            stability_score = np.mean([rotation_stability, com_stability, symmetry])
            
            return {
                'rotation_stability': rotation_stability,
                'com_stability': com_stability,
                'step_symmetry': symmetry,
                'overall_score': stability_score
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating stability metrics: {str(e)}")
            return None

    def update_metrics(self):
        """Update displayed metrics"""
        try:
            if not self.analysis_complete:
                return
                
            # Update metric displays
            self.metrics['total_rotation'].set(
                f"Total Rotation: {self.results['rotation']['total_rotation']:.1f}°"
            )
            self.metrics['lateral_deviation'].set(
                f"Lateral Deviation: {self.results['com']['lateral_deviation']:.1f} mm"
            )
            self.metrics['step_symmetry'].set(
                f"Step Symmetry: {self.results['stability']['step_symmetry']:.1f}%"
            )
            self.metrics['stability_score'].set(
                f"Stability Score: {self.results['stability']['overall_score']:.1f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")

    def update_visualization(self):
        """Update all visualization plots"""
        try:
            if not self.analysis_complete:
                return
                
            # Clear all plots
            for ax in self.axes.values():
                ax.clear()
                
            # Plot top view trajectory
            self.plot_top_view()
            
            # Plot COM path
            self.plot_com_path()
            
            # Plot rotation over time
            self.plot_rotation()
            
            # Plot step analysis
            self.plot_step_analysis()
            
            # Refresh all canvases
            for canvas in self.canvases.values():
                canvas.draw()
                
        except Exception as e:
            self.logger.error(f"Error updating visualization: {str(e)}")

    def export_results(self):
        """Export analysis results"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Export Analysis Results",
                defaultextension=".xlsx",
                filetypes=[
                    ("Excel files", "*.xlsx"),
                    ("CSV files", "*.csv")
                ]
            )
            
            if not filename:
                return
                
            # Create results dataframe
            results_df = pd.DataFrame({
                'Frame': range(len(self.results['rotation']['angles'])),
                'Rotation': self.results['rotation']['angles'],
                'COM_X': self.results['com']['displacements'][:, 0],
                'COM_Y': self.results['com']['displacements'][:, 1],
                'COM_Z': self.results['com']['displacements'][:, 2]
            })
            
            # Create summary dataframe
            summary_df = pd.DataFrame({
                'Metric': list(self.metrics.keys()),
                'Value': [var.get().split(': ')[1] for var in self.metrics.values()]
            })
            
            # Add step analysis data
            step_df = pd.DataFrame({
                'Step_Number': range(1, len(self.results['steps']['step_lengths']) + 1),
                'Step_Length': self.results['steps']['step_lengths'],
                'Step_Duration': self.results['steps']['step_durations']
            })
            
            # Export to file
            if filename.endswith('.xlsx'):
                with pd.ExcelWriter(filename) as writer:
                    results_df.to_excel(writer, sheet_name='Detailed Data', index=False)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    step_df.to_excel(writer, sheet_name='Step Analysis', index=False)
                    
                    # Add metadata sheet
                    metadata = {
                        'Analysis Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Analysis Parameters': self.params,
                        'Data Source': os.path.basename(self.motion_data['filename']) if 'filename' in self.motion_data else 'Unknown'
                    }
                    pd.DataFrame([metadata]).to_excel(writer, sheet_name='Metadata', index=False)
                    
            else:
                # For CSV, save multiple files with descriptive names
                base_name = os.path.splitext(filename)[0]
                results_df.to_csv(f"{base_name}_detailed.csv", index=False)
                summary_df.to_csv(f"{base_name}_summary.csv", index=False)
                step_df.to_csv(f"{base_name}_steps.csv", index=False)
                
            # Save visualization plots
            plot_filename = os.path.splitext(filename)[0] + '_plots.pdf'
            with PdfPages(plot_filename) as pdf:
                for fig in self.figs.values():
                    pdf.savefig(fig)
                    
            messagebox.showinfo(
                "Export Complete",
                f"Results successfully exported to:\n{filename}\n\nPlots saved to:\n{plot_filename}"
            )
            
            # Log export
            self.logger.info(f"Analysis results exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            messagebox.showerror(
                "Export Error",
                f"Failed to export results: {str(e)}\n\nPlease check the log file for details."
            )

        def clear_visualizations(self):
            """Clear all visualization plots"""
            for ax in self.axes.values():
                ax.clear()
            for canvas in self.canvases.values():
                canvas.draw()

        def plot_top_view(self):
            """Plot bird's eye view of movement trajectory"""
            try:
                ax = self.axes['top_view']
                com_data = self.results['com']['trajectory']
                
                # Plot COM path
                ax.plot(com_data[:, 0], com_data[:, 1], 'b-', linewidth=2, label='Path')
                
                # Plot start and end points
                ax.plot(com_data[0, 0], com_data[0, 1], 'go', markersize=10, label='Start')
                ax.plot(com_data[-1, 0], com_data[-1, 1], 'ro', markersize=10, label='End')
                
                # Plot initial orientation
                shoulder_points = self.motion_data['frames'][0]['keypoints3D']
                left = shoulder_points['leftShoulder']
                right = shoulder_points['rightShoulder']
                ax.plot([left[0], right[0]], [left[1], right[1]], 'k-', linewidth=2, label='Initial Orientation')
                
                # Configure plot
                ax.set_title('Top View Trajectory')
                ax.set_xlabel('Lateral Position (mm)')
                ax.set_ylabel('Anterior-Posterior Position (mm)')
                ax.grid(True)
                ax.axis('equal')
                ax.legend()
                
            except Exception as e:
                self.logger.error(f"Error plotting top view: {str(e)}")

        def plot_com_path(self):
            """Plot center of mass path with stability metrics"""
            try:
                ax = self.axes['com_path']
                com_data = self.results['com']['displacements']
                
                # Plot lateral deviation over time
                time = np.arange(len(com_data)) / len(com_data) * 100
                ax.plot(time, com_data[:, 0], 'b-', label='Lateral')
                ax.plot(time, com_data[:, 1], 'r-', label='AP')
                
                # Add threshold lines
                threshold = self.params['com_deviation_threshold']
                ax.axhline(y=threshold, color='k', linestyle='--', alpha=0.5)
                ax.axhline(y=-threshold, color='k', linestyle='--', alpha=0.5)
                
                # Configure plot
                ax.set_title('Center of Mass Deviation')
                ax.set_xlabel('Test Completion (%)')
                ax.set_ylabel('Displacement (mm)')
                ax.grid(True)
                ax.legend()
                
            except Exception as e:
                self.logger.error(f"Error plotting COM path: {str(e)}")

        def plot_rotation(self):
            """Plot rotation analysis"""
            try:
                ax = self.axes['rotation']
                rotation_data = self.results['rotation']['angles']
                time = np.arange(len(rotation_data)) / len(rotation_data) * 100
                
                # Plot rotation angle over time
                ax.plot(time, rotation_data, 'b-', label='Rotation')
                
                # Add threshold lines
                threshold = self.params['rotation_threshold']
                ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
                ax.axhline(y=-threshold, color='r', linestyle='--', alpha=0.5)
                
                # Configure plot
                ax.set_title('Body Rotation')
                ax.set_xlabel('Test Completion (%)')
                ax.set_ylabel('Rotation Angle (degrees)')
                ax.grid(True)
                ax.legend()
                
            except Exception as e:
                self.logger.error(f"Error plotting rotation: {str(e)}")

        def plot_step_analysis(self):
            """Plot step-by-step analysis"""
            try:
                ax = self.axes['step_analysis']
                step_data = self.results['steps']
                
                # Plot step lengths
                step_numbers = range(1, len(step_data['step_lengths']) + 1)
                ax.bar(step_numbers, step_data['step_lengths'], alpha=0.6, label='Step Length')
                
                # Add step symmetry line
                mean_length = np.mean(step_data['step_lengths'])
                ax.axhline(y=mean_length, color='r', linestyle='--', label='Mean Length')
                
                # Configure plot
                ax.set_title('Step Analysis')
                ax.set_xlabel('Step Number')
                ax.set_ylabel('Step Length (mm)')
                ax.grid(True)
                ax.legend()
                
            except Exception as e:
                self.logger.error(f"Error plotting step analysis: {str(e)}")

        def calculate_angle_change(self, reference_points, current_points):
            """Calculate angle change between two sets of points"""
            try:
                # Calculate vectors
                ref_vector = reference_points[1] - reference_points[0]
                curr_vector = current_points[1] - current_points[0]
                
                # Calculate angle between vectors
                dot_product = np.dot(ref_vector[:2], curr_vector[:2])
                ref_mag = np.linalg.norm(ref_vector[:2])
                curr_mag = np.linalg.norm(curr_vector[:2])
                
                angle = np.arccos(dot_product / (ref_mag * curr_mag))
                angle_deg = np.degrees(angle)
                
                # Determine rotation direction using cross product
                cross_product = np.cross(ref_vector[:2], curr_vector[:2])
                if cross_product < 0:
                    angle_deg = -angle_deg
                    
                return angle_deg
                
            except Exception as e:
                self.logger.error(f"Error calculating angle change: {str(e)}")
                return 0.0

        def detect_steps(self, ankle_heights):
            """Detect individual steps from ankle height data"""
            try:
                # Calculate height threshold for step detection
                threshold = self.params['step_detection_threshold']
                mean_height = np.mean(ankle_heights)
                step_threshold = mean_height + threshold
                
                # Find peaks above threshold
                steps = []
                in_step = False
                
                for i, height in enumerate(ankle_heights):
                    if height > step_threshold and not in_step:
                        steps.append(i)
                        in_step = True
                    elif height < step_threshold:
                        in_step = False
                        
                return np.array(steps)
                
            except Exception as e:
                self.logger.error(f"Error detecting steps: {str(e)}")
                return np.array([])

        def calculate_step_lengths(self, right_steps, left_steps):
            """Calculate lengths of individual steps"""
            try:
                step_lengths = []
                
                # Combine steps and sort by time
                all_steps = np.sort(np.concatenate([right_steps, left_steps]))
                
                # Calculate distance between consecutive steps
                for i in range(len(all_steps) - 1):
                    step_start = self.results['com']['trajectory'][all_steps[i]]
                    step_end = self.results['com']['trajectory'][all_steps[i + 1]]
                    length = np.linalg.norm(step_end[:2] - step_start[:2])
                    step_lengths.append(length)
                    
                return np.array(step_lengths)
                
            except Exception as e:
                self.logger.error(f"Error calculating step lengths: {str(e)}")
                return np.array([])

        def calculate_step_durations(self, right_steps, left_steps):
            """Calculate durations of individual steps"""
            try:
                step_durations = []
                
                # Combine steps and sort by time
                all_steps = np.sort(np.concatenate([right_steps, left_steps]))
                
                # Calculate time between consecutive steps
                for i in range(len(all_steps) - 1):
                    duration = all_steps[i + 1] - all_steps[i]
                    step_durations.append(duration)
                    
                return np.array(step_durations)
                
            except Exception as e:
                self.logger.error(f"Error calculating step durations: {str(e)}")
                return np.array([])

        def calculate_coordination_metrics(self, step_data):
            """Calculate coordination metrics for Fukuda step test"""
            try:
                if not step_data or len(step_data['right_steps']) < 2:
                    raise ValueError("Insufficient step data for coordination analysis")
                    
                # Initialize coordination metrics
                metrics = {
                    'temporal_symmetry': 0.0,
                    'spatial_symmetry': 0.0,
                    'step_regularity': 0.0,
                    'rotational_stability': 0.0
                }
                
                # Calculate temporal symmetry
                right_durations = np.diff(step_data['right_steps'])
                left_durations = np.diff(step_data['left_steps'])
                metrics['temporal_symmetry'] = 100 * (1 - abs(
                    np.mean(right_durations) - np.mean(left_durations)
                ) / np.mean(right_durations))
                
                # Calculate spatial symmetry using step lengths
                right_lengths = step_data['step_lengths'][::2]  # Even indices
                left_lengths = step_data['step_lengths'][1::2]  # Odd indices
                metrics['spatial_symmetry'] = 100 * (1 - abs(
                    np.mean(right_lengths) - np.mean(left_lengths)
                ) / np.mean(right_lengths))
                
                # Calculate step regularity using coefficient of variation
                metrics['step_regularity'] = 100 * (1 - np.std(
                    step_data['step_lengths']
                ) / np.mean(step_data['step_lengths']))
                
                # Calculate rotational stability
                rotation_angles = np.diff(self.calculate_rotation_angles())
                metrics['rotational_stability'] = 100 * (1 - np.std(
                    rotation_angles
                ) / 45.0)  # Normalize to 45 degrees
                
                return metrics
                
            except Exception as e:
                self.logger.error(f"Error calculating coordination metrics: {str(e)}")
                return None

        def calculate_rotation_angles(self):
            """Calculate rotation angles throughout the test"""
            try:
                angles = []
                
                for frame in self.motion_data['frames']:
                    # Get shoulder points 
                    left_shoulder = np.array(frame['keypoints3D']['leftShoulder'])
                    right_shoulder = np.array(frame['keypoints3D']['rightShoulder'])
                    
                    # Calculate shoulder vector in horizontal plane
                    shoulder_vector = right_shoulder - left_shoulder
                    shoulder_vector = shoulder_vector[:2]  # Use only x,y components
                    
                    # Calculate angle with anterior direction
                    anterior = np.array([0, 1])
                    angle = np.arctan2(
                        np.cross(anterior, shoulder_vector),
                        np.dot(anterior, shoulder_vector)
                    )
                    angles.append(np.degrees(angle))
                    
                return np.array(angles)
                
            except Exception as e:
                self.logger.error(f"Error calculating rotation angles: {str(e)}")
                return np.array([])

        def update_visualization(self):
            """Update all visualization plots with current analysis results"""
            try:
                if not self.analysis_complete:
                    return
                    
                # Clear previous plots
                for ax in self.axes.values():
                    ax.clear()
                    
                # Plot rotation pattern
                self.plot_rotation_pattern()
                
                # Plot step trajectories
                self.plot_step_trajectories()
                
                # Plot coordination metrics
                self.plot_coordination_metrics()
                
                # Plot stability analysis
                self.plot_stability_analysis()
                
                # Refresh all canvases
                for canvas in self.canvases.values():
                    canvas.draw()
                    
            except Exception as e:
                self.logger.error(f"Error updating visualization: {str(e)}")

        def plot_rotation_pattern(self):
            """Plot rotation pattern throughout the test"""
            try:
                ax = self.axes['rotation']
                
                # Get rotation data
                angles = self.calculate_rotation_angles()
                time = np.linspace(0, 100, len(angles))
                
                # Plot rotation angle over time
                ax.plot(time, angles, 'b-', linewidth=2, label='Rotation')
                
                # Add threshold lines
                threshold = self.params['rotation_threshold']
                ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
                ax.axhline(y=-threshold, color='r', linestyle='--', alpha=0.5)
                
                # Configure plot
                ax.set_title('Rotation Pattern Analysis')
                ax.set_xlabel('Test Progress (%)')
                ax.set_ylabel('Rotation Angle (degrees)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
            except Exception as e:
                self.logger.error(f"Error plotting rotation pattern: {str(e)}")

class MovementAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movement Analysis System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)  # Set minimum window size
        
        # Configure logging
        logging.basicConfig(
            filename='movement_analysis.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize data manager
        self.data_manager = DataManager()
        
        # Create style configuration
        self.create_styles()
        
        # Create main interface
        self.create_gui()
        
    def create_styles(self):
        """Create custom styles for widgets"""
        style = ttk.Style()
        style.configure('Analysis.TButton', padding=5)
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Metric.TLabel', padding=3)
        
    def create_gui(self):
        """Create main application interface"""
        # Create and configure main notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create menu bar
        self.create_menu()
        
        # Create tabs
        self.create_tabs()
        
    def create_menu(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def create_tabs(self):
        """Create assessment tabs"""
        # Create tab instances 
        self.standing_posture = StandingPostureTab(self.notebook, self.data_manager)  # Add data_manager parameter
        self.body_weight_squat = BodyWeightSquatTab(self.notebook, self.data_manager)
        self.hip_hinge = HipHingeTab(self.notebook, self.data_manager)
        self.walking_gait = WalkingGaitTab(self.notebook, self.data_manager)
        self.fukuda_step = FukudaStepTab(self.notebook, self.data_manager)
        
        # Add tabs to notebook
        self.notebook.add(self.standing_posture.tab, text="Standing Posture")
        self.notebook.add(self.body_weight_squat.tab, text="Body Weight Squat")
        self.notebook.add(self.hip_hinge.tab, text="Hip Hinge")
        self.notebook.add(self.walking_gait.tab, text="Walking Gait")
        self.notebook.add(self.fukuda_step.tab, text="Fukuda Step")
        
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About Movement Analysis System",
            "Movement Analysis System v1.0\n\n"
            "A tool for analyzing human movement patterns\n"
            "using motion capture data."
        )
        
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = MovementAnalysisApp(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        messagebox.showerror(
            "Error",
            f"An error occurred while running the application:\n{str(e)}"
        )
