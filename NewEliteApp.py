import os
import json
import uuid
import random
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from typing import Dict, List, Set, Optional, Union
from dataclasses import dataclass, asdict


"START OF CORE STORAGE MODULE"
# Custom Exceptions
class JsonStorageError(Exception):
    """Custom exception for JSON storage operations"""
    pass

class ValidationError(JsonStorageError):
    """Exception for data validation failures"""
    pass

class StorageOperationError(JsonStorageError):
    """Exception for storage operation failures"""
    pass

@dataclass
class StorageMetadata:
    """Metadata for storage files"""
    version: str = "1.0.0"
    last_modified: str = ""
    backup_path: str = ""

class BaseJsonStorage:
    """Base class for JSON storage operations with built-in validation and backup"""
    
    def __init__(self, filepath: str, backup_dir: str = "backups"):
        self.filepath = filepath
        self.backup_dir = os.path.join(os.path.dirname(filepath), backup_dir)
        self.metadata = StorageMetadata()
        self.ensure_directories()

    def ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

    def create_backup(self) -> str:
        """Create a backup of the current data file"""
        if not os.path.exists(self.filepath):
            return ""
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(
            self.backup_dir,
            f"{os.path.basename(self.filepath)}.{timestamp}.bak"
        )
        
        try:
            shutil.copy2(self.filepath, backup_path)
            self.metadata.backup_path = backup_path
            return backup_path
        except Exception as e:
            raise StorageOperationError(f"Failed to create backup: {str(e)}")

    def validate_data(self, data: dict) -> bool:
        """Validate data structure before saving
        Override this method in subclasses to implement specific validation"""
        return True

    def read(self) -> dict:
        """Read and validate JSON data from file"""
        try:
            if not os.path.exists(self.filepath):
                return {}
                
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                
            if not self.validate_data(data):
                raise ValidationError(f"Invalid data structure in {self.filepath}")
                
            return data
        except json.JSONDecodeError as e:
            raise StorageOperationError(f"Invalid JSON in {self.filepath}: {str(e)}")
        except Exception as e:
            raise StorageOperationError(f"Failed to read {self.filepath}: {str(e)}")

    def write(self, data: dict):
        """Write JSON data to file with validation and backup"""
        if not self.validate_data(data):
            raise ValidationError(f"Invalid data structure for {self.filepath}")
            
        try:
            # Create backup before writing
            self.create_backup()
            
            # Update metadata
            self.metadata.last_modified = datetime.now().isoformat()
            
            # Combine data with metadata
            full_data = {
                "metadata": asdict(self.metadata),
                "data": data
            }
            
            # Write to temporary file first
            temp_path = f"{self.filepath}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(full_data, f, indent=2)
            
            # Replace original file with temporary file
            os.replace(temp_path, self.filepath)
            
        except Exception as e:
            raise StorageOperationError(f"Failed to write to {self.filepath}: {str(e)}")

    def restore_from_backup(self, backup_path: str = None):
        """Restore data from a backup file"""
        if not backup_path:
            backup_path = self.metadata.backup_path
            
        if not backup_path or not os.path.exists(backup_path):
            raise StorageOperationError("No valid backup file found")
            
        try:
            shutil.copy2(backup_path, self.filepath)
            return self.read()
        except Exception as e:
            raise StorageOperationError(f"Failed to restore from backup: {str(e)}")

class StorageManager:
    """Central manager for all storage operations"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # Initialize storage components
        self.exercise_storage = ExerciseStorage(
            os.path.join(data_dir, "exercises.json")
        )
        self.workout_storage = WorkoutStorage(
            os.path.join(data_dir, "workouts.json")
        )
        self.program_storage = ProgramStorage(
            os.path.join(data_dir, "programs.json")
        )

    def ensure_data_directory(self):
        """Ensure the data directory exists"""
        os.makedirs(self.data_dir, exist_ok=True)

    def save_all(self):
        """Save all storage components"""
        try:
            self.exercise_storage.save()
            self.workout_storage.save()
            self.program_storage.save()
        except Exception as e:
            raise StorageOperationError(f"Failed to save all data: {str(e)}")

    def load_all(self):
        """Load all storage components"""
        try:
            self.exercise_storage.load()
            self.workout_storage.load()
            self.program_storage.load()
        except Exception as e:
            raise StorageOperationError(f"Failed to load all data: {str(e)}")

    def create_backup(self):
        """Create backups of all storage components"""
        backups = {
            "exercises": self.exercise_storage.create_backup(),
            "workouts": self.workout_storage.create_backup(),
            "programs": self.program_storage.create_backup()
        }
        return backups

    def restore_from_backup(self, component: str = None):
        """Restore specific or all components from backup"""
        if component:
            if component == "exercises":
                self.exercise_storage.restore_from_backup()
            elif component == "workouts":
                self.workout_storage.restore_from_backup()
            elif component == "programs":
                self.program_storage.restore_from_backup()
            else:
                raise ValueError(f"Invalid component: {component}")
        else:
            # Restore all components
            self.exercise_storage.restore_from_backup()
            self.workout_storage.restore_from_backup()
            self.program_storage.restore_from_backup()

class DataValidator:
    """Utility class for data validation"""
    
    @staticmethod
    def validate_exercise_data(data: dict) -> bool:
        """Validate exercise data structure"""
        if not isinstance(data, dict):
            return False
            
        required_keys = {"metadata", "data"}
        if not all(key in data for key in required_keys):
            return False
            
        exercise_data = data["data"]
        if not isinstance(exercise_data, dict):
            return False
            
        # Validate exercise categories
        required_categories = {
            "squat", "hinge", "total_body",
            "upper_push_horizontal", "upper_push_vertical",
            "upper_pull_horizontal", "upper_pull_vertical"
        }
        
        if not all(category in exercise_data for category in required_categories):
            return False
            
        # Validate category structure
        for category, info in exercise_data.items():
            if not isinstance(info, dict):
                return False
            if "exercises" not in info or "prefix" not in info:
                return False
                
        return True

    @staticmethod
    def validate_workout_data(data: dict) -> bool:
        """Validate workout data structure"""
        if not isinstance(data, dict):
            return False
            
        required_keys = {"metadata", "data"}
        if not all(key in data for key in required_keys):
            return False
            
        workout_data = data["data"]
        if not isinstance(workout_data, dict):
            return False
            
        # Validate workout categories
        required_categories = {"squat_push", "hinge_pull", "total_body"}
        if not all(category in workout_data for category in required_categories):
            return False
            
        # Validate split workout structure
        for category in ["squat_push", "hinge_pull"]:
            if not isinstance(workout_data[category], dict):
                return False
            if not all(plane in workout_data[category] 
                      for plane in ["horizontal", "vertical"]):
                return False
                
        return True

    @staticmethod
    def validate_program_data(data: dict) -> bool:
        """Validate program data structure"""
        if not isinstance(data, dict):
            return False
            
        required_keys = {"metadata", "data"}
        if not all(key in data for key in required_keys):
            return False
            
        program_data = data["data"]
        if not isinstance(program_data, dict):
            return False
            
        if "programs" not in program_data:
            return False
            
        # Validate program structure
        programs = program_data["programs"]
        if not isinstance(programs, list):
            return False
            
        for program in programs:
            if not isinstance(program, dict):
                return False
            if not all(key in program for key in ["id", "date_created", "weeks"]):
                return False
                
        return True
"END OF CORE STORAG MODULE"



"START OF EXERCISE MANAGEMENT MODULE"    
class Exercise:
    """Represents an exercise with its properties"""
    def __init__(self, name: str, exercise_id: str, pattern: str, plane: str = None):
        self.name = name
        self.id = exercise_id
        self.pattern = pattern
        self.plane = plane
        
    def to_dict(self) -> dict:
        """Convert exercise to dictionary representation"""
        return {
            "name": self.name,
            "id": self.id,
            "pattern": self.pattern,
            "plane": self.plane
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'Exercise':
        """Create exercise instance from dictionary"""
        return cls(
            name=data["name"],
            exercise_id=data["id"],
            pattern=data["pattern"],
            plane=data.get("plane")
        )
        
    def __eq__(self, other):
        if not isinstance(other, Exercise):
            return False
        return self.id == other.id
        
    def __hash__(self):
        return hash(self.id)
    
class ExerciseStorage(BaseJsonStorage):
    """Handles exercise data storage with validation and backup capabilities"""
    
    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.categories = {
            "squat": {"prefix": "SQ_", "exercises": set()},
            "hinge": {"prefix": "HG_", "exercises": set()},
            "total_body": {"prefix": "TB_", "exercises": set()},
            "upper_push_horizontal": {"prefix": "UPH_", "exercises": set()},
            "upper_push_vertical": {"prefix": "UPV_", "exercises": set()},
            "upper_pull_horizontal": {"prefix": "ULH_", "exercises": set()},
            "upper_pull_vertical": {"prefix": "ULV_", "exercises": set()}
        }
        self.exercise_names = set()
        self.load_data()

    def validate_data(self, data: dict) -> bool:
        """Override BaseJsonStorage validation with exercise-specific checks"""
        try:
            if not isinstance(data, dict):
                return False

            # Check metadata structure if present
            if "metadata" in data:
                metadata = data["metadata"]
                if not all(key in metadata for key in ["version", "last_modified"]):
                    return False
                data = data.get("data", {})

            # Verify required categories exist
            required_categories = set(self.categories.keys())
            if not all(category in data for category in required_categories):
                return False

            # Validate structure of each category
            for category, info in data.items():
                if not isinstance(info, dict):
                    return False
                if not all(key in info for key in ["exercises", "prefix"]):
                    return False
                if not isinstance(info["exercises"], list):
                    return False
                if not isinstance(info["prefix"], str):
                    return False

                # Validate individual exercises
                for exercise in info["exercises"]:
                    if not isinstance(exercise, dict):
                        return False
                    if not all(key in exercise for key in ["name", "id", "pattern"]):
                        return False

            return True
        except Exception:
            return False

    def load_data(self):
        """Load and process exercise data from storage"""
        try:
            data = self.read()
            # Extract actual data from metadata wrapper if present
            exercise_data = data.get("data", data)

            # Clear existing data
            self.exercise_names.clear()
            for category in self.categories.values():
                category["exercises"].clear()

            # Load new data
            for category, info in exercise_data.items():
                if category in self.categories:
                    self.categories[category]["exercises"] = {
                        Exercise.from_dict(ex) for ex in info.get("exercises", [])
                    }
                    self.categories[category]["prefix"] = info["prefix"]
                    self.exercise_names.update(
                        ex.name.lower() for ex in self.categories[category]["exercises"]
                    )
        except Exception as e:
            raise StorageOperationError(f"Failed to load exercise data: {str(e)}")

    def save_data(self):
        """Save exercise data with validation and backup"""
        try:
            data = {
                category: {
                    "exercises": [ex.to_dict() for ex in info["exercises"]],
                    "prefix": info["prefix"]
                }
                for category, info in self.categories.items()
            }
            self.write(data)
        except Exception as e:
            raise StorageOperationError(f"Failed to save exercise data: {str(e)}")            

    def get_exercises_by_pattern(self, pattern: str, plane: str = None) -> List[Exercise]:
        """Get exercises matching the specified pattern and plane"""
        try:
            exercises = []
            for category, info in self.categories.items():
                if pattern.lower() in category.lower():
                    if plane:
                        if plane.lower() in category.lower():
                            exercises.extend(info["exercises"])
                    else:
                        exercises.extend(info["exercises"])
            return exercises
        except Exception as e:
            raise StorageOperationError(f"Error retrieving exercises: {str(e)}")

    def restore_from_backup(self, backup_path: str = None):
        """Restore exercise data from a backup file"""
        try:
            data = super().restore_from_backup(backup_path)
            self.load_data()  # Reload data from restored file
            return data
        except Exception as e:
            raise StorageOperationError(f"Failed to restore from backup: {str(e)}")

class ExercisePair:
    """Represents a pair of exercises"""
    def __init__(self, exercises: List[Exercise], pair_id: str):
        self.exercises = sorted(exercises, key=lambda x: x.name)
        self.id = pair_id
        
    def to_dict(self) -> dict:
        """Convert pair to dictionary representation"""
        return {
            "exercises": [ex.to_dict() for ex in self.exercises],
            "id": self.id
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'ExercisePair':
        """Create pair instance from dictionary"""
        exercises = [Exercise.from_dict(ex) for ex in data["exercises"]]
        return cls(exercises, data["id"])
        
    def __eq__(self, other):
        if not isinstance(other, ExercisePair):
            return False
        return self.id == other.id
        
    def __hash__(self):
        return hash(self.id)

class PairingStorage(BaseJsonStorage):
    """Handles exercise pairing data storage"""
    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.categories = {
            "squat_push": {
                "horizontal": {"prefix": "SP_H", "pairs": set()},
                "vertical": {"prefix": "SP_V", "pairs": set()}
            },
            "hinge_pull": {
                "horizontal": {"prefix": "HP_H", "pairs": set()},
                "vertical": {"prefix": "HP_V", "pairs": set()}
            }
        }
        self.load_data()
        
    def load_data(self):
        """Load pairing data from storage"""
        data = self.read()
        for category, planes in data.items():
            if category in self.categories:
                for plane, info in planes.items():
                    self.categories[category][plane]["pairs"] = {
                        ExercisePair.from_dict(pair) for pair in info.get("pairs", [])
                    }

    def save_data(self):
        """Save pairing data to storage"""
        data = {
            category: {
                plane: {
                    "pairs": [pair.to_dict() for pair in info["pairs"]],
                    "prefix": info["prefix"]
                }
                for plane, info in planes.items()
            }
            for category, planes in self.categories.items()
        }
        self.write(data)
"END OF EXERCISE MANAGEMENT MODULE"



"START OF WORKOUT MANAGEMENT MODULE"
@dataclass
class WorkoutMetadata:
    """Metadata for workout storage"""
    version: str = "1.0.0"
    last_modified: str = ""
    total_workouts: int = 0
    workout_types: Dict[str, int] = None
    
    def __post_init__(self):
        if self.workout_types is None:
            self.workout_types = {
                "squat_push": 0,
                "hinge_pull": 0,
                "total_body": 0
            }
            
class Workout:
    """Represents a complete workout"""
    def __init__(self, workout_type: str, exercises: Dict[str, Exercise], 
                 workout_id: str = None, plane: str = None):
        self.type = workout_type
        self.exercises = exercises
        self.id = workout_id or str(uuid.uuid4())
        self.plane = plane
        
    def to_dict(self) -> dict:
        """Convert workout to dictionary representation"""
        return {
            "id": self.id,
            "type": self.type,
            "plane": self.plane,
            "exercises": {
                position: ex.to_dict() 
                for position, ex in self.exercises.items()
            }
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'Workout':
        """Create workout instance from dictionary"""
        exercises = {
            position: Exercise.from_dict(ex_data)
            for position, ex_data in data["exercises"].items()
        }
        return cls(
            workout_type=data["type"],
            exercises=exercises,
            workout_id=data["id"],
            plane=data.get("plane")
        )

class WorkoutStorage(BaseJsonStorage):
    """Handles workout data storage with validation and backup capabilities"""
    
    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.metadata = WorkoutMetadata()
        self.workouts = {
            "squat_push": {"horizontal": [], "vertical": []},
            "hinge_pull": {"horizontal": [], "vertical": []},
            "total_body": []
        }
        self.load_data()
        
    def validate_data(self, data: dict) -> bool:
        """Validate workout data structure and content"""
        try:
            if not isinstance(data, dict):
                return False
                
            # Validate metadata if present
            if "metadata" in data:
                metadata = data["metadata"]
                if not all(key in metadata for key in ["version", "last_modified", "total_workouts", "workout_types"]):
                    return False
                data = data.get("data", {})
                
            # Check required categories
            required_categories = {"squat_push", "hinge_pull", "total_body"}
            if not all(category in data for category in required_categories):
                return False
                
            # Validate split workout structure
            for category in ["squat_push", "hinge_pull"]:
                if not isinstance(data[category], dict):
                    return False
                    
                # Check plane categories
                if not all(plane in data[category] for plane in ["horizontal", "vertical"]):
                    return False
                    
                # Validate workout lists
                for plane_workouts in data[category].values():
                    if not isinstance(plane_workouts, list):
                        return False
                    
                    # Validate individual workouts
                    for workout in plane_workouts:
                        if not self._validate_workout_structure(workout):
                            return False
            
            # Validate total body workouts
            if not isinstance(data["total_body"], list):
                return False
                
            for workout in data["total_body"]:
                if not self._validate_workout_structure(workout):
                    return False
                    
            return True
            
        except Exception:
            return False
            
    def _validate_workout_structure(self, workout: dict) -> bool:
        """Validate individual workout data structure"""
        try:
            required_fields = {"id", "type", "exercises"}
            if not all(field in workout for field in required_fields):
                return False
                
            if not isinstance(workout["exercises"], dict):
                return False
            
            # Validate exercise positions
            if workout["type"] == "total_body":
                required_positions = {f"{i}{c}" for i in range(1, 4) for c in 'abc'}
            else:
                required_positions = (
                    {f"{i}a" for i in range(1, 4)} |
                    {f"{i}b" for i in range(1, 4)} |
                    {f"4{c}" for c in 'ab'}
                )
                
            if set(workout["exercises"].keys()) != required_positions:
                return False
                
            # Validate exercise structure
            for exercise in workout["exercises"].values():
                required_exercise_fields = {"name", "id", "pattern"}
                if not all(field in exercise for field in required_exercise_fields):
                    return False
                    
                # Validate exercise properties
                if not isinstance(exercise["name"], str) or not exercise["name"]:
                    return False
                if not isinstance(exercise["id"], str) or not exercise["id"]:
                    return False
                if not isinstance(exercise["pattern"], str) or not exercise["pattern"]:
                    return False
                    
            return True
            
        except Exception:
            return False
        
    def _update_metadata(self):
        """Update storage metadata"""
        self.metadata.last_modified = datetime.now().isoformat()
        self.metadata.total_workouts = sum(
            len(workouts) if isinstance(workouts, list) else
            sum(len(plane_workouts) for plane_workouts in workouts.values())
            for workouts in self.workouts.values()
        )
        
        # Update workout type counts
        self.metadata.workout_types = {
            "squat_push": sum(len(workouts) for workouts in self.workouts["squat_push"].values()),
            "hinge_pull": sum(len(workouts) for workouts in self.workouts["hinge_pull"].values()),
            "total_body": len(self.workouts["total_body"])
        }
        
    def load_data(self):
        """Load workout data from storage with validation"""
        try:
            data = self.read()
            
            # Load metadata if present
            if "metadata" in data:
                metadata = data["metadata"]
                self.metadata = WorkoutMetadata(
                    version=metadata.get("version", "1.0.0"),
                    last_modified=metadata.get("last_modified", ""),
                    total_workouts=metadata.get("total_workouts", 0),
                    workout_types=metadata.get("workout_types", {
                        "squat_push": 0,
                        "hinge_pull": 0,
                        "total_body": 0
                    })
                )
                data = data.get("data", {})
            
            # Clear existing data
            for category in self.workouts:
                if category == "total_body":
                    self.workouts[category] = []
                else:
                    self.workouts[category] = {"horizontal": [], "vertical": []}
            
            # Load new data
            for category, type_data in data.items():
                if category == "total_body":
                    self.workouts[category] = [
                        Workout.from_dict(w) for w in type_data
                    ]
                else:
                    for plane in ["horizontal", "vertical"]:
                        self.workouts[category][plane] = [
                            Workout.from_dict(w) for w in type_data.get(plane, [])
                        ]
                        
            # Update metadata
            self._update_metadata()
                        
        except Exception as e:
            raise StorageOperationError(f"Failed to load workout data: {str(e)}")
            
    def save_data(self):
        """Save workout data with validation and backup"""
        try:
            # Update metadata
            self._update_metadata()
            
            # Prepare workout data
            workout_data = {}
            for category, type_data in self.workouts.items():
                if category == "total_body":
                    workout_data[category] = [w.to_dict() for w in type_data]
                else:
                    workout_data[category] = {
                        plane: [w.to_dict() for w in workouts]
                        for plane, workouts in type_data.items()
                    }
            
            # Combine with metadata using asdict
            data = {
                "metadata": asdict(self.metadata),
                "data": workout_data
            }
                    
            self.write(data)
            
        except Exception as e:
            raise StorageOperationError(f"Failed to save workout data: {str(e)}")
            
    def add_workout(self, workout: Workout) -> tuple[bool, str]:
        """Add a new workout to storage"""
        try:
            # Validate workout type
            if workout.type not in self.workouts:
                return False, f"Invalid workout type: {workout.type}"
                
            # Set ID if not present
            if not workout.id:
                workout.id = str(uuid.uuid4())
                
            if workout.type == "total_body":
                self.workouts[workout.type].append(workout)
            else:
                if not workout.plane:
                    return False, "Plane focus required for split workouts"
                if workout.plane not in ["horizontal", "vertical"]:
                    return False, f"Invalid plane: {workout.plane}"
                self.workouts[workout.type][workout.plane].append(workout)
                
            self.save_data()
            return True, f"Workout saved with ID: {workout.id}"
            
        except Exception as e:
            return False, f"Error saving workout: {str(e)}"
            
    def remove_workout(self, workout_id: str) -> tuple[bool, str]:
        """Remove a workout from storage"""
        try:
            for workout_type, type_data in self.workouts.items():
                if workout_type == "total_body":
                    for workout in type_data:
                        if workout.id == workout_id:
                            type_data.remove(workout)
                            self.save_data()
                            return True, f"Removed workout: {workout_id}"
                else:
                    for plane_workouts in type_data.values():
                        for workout in plane_workouts:
                            if workout.id == workout_id:
                                plane_workouts.remove(workout)
                                self.save_data()
                                return True, f"Removed workout: {workout_id}"
                                
            return False, "Workout not found"
            
        except Exception as e:
            return False, f"Error removing workout: {str(e)}"
            
    def clear_workouts(self, workout_type: str = None, plane: str = None) -> tuple[bool, str]:
        """Clear workouts with optional type and plane filtering"""
        try:
            if workout_type:
                if workout_type == "total_body":
                    self.workouts[workout_type] = []
                elif workout_type in self.workouts:
                    if plane:
                        if plane in self.workouts[workout_type]:
                            self.workouts[workout_type][plane] = []
                        else:
                            return False, f"Invalid plane: {plane}"
                    else:
                        self.workouts[workout_type] = {
                            "horizontal": [],
                            "vertical": []
                        }
                else:
                    return False, f"Invalid workout type: {workout_type}"
            else:
                # Clear all workouts
                self.workouts = {
                    "squat_push": {"horizontal": [], "vertical": []},
                    "hinge_pull": {"horizontal": [], "vertical": []},
                    "total_body": []
                }
                
            self.save_data()
            return True, "Workouts cleared successfully"
            
        except Exception as e:
            return False, f"Error clearing workouts: {str(e)}"
            
    def get_workouts(self, workout_type: str = None, plane: str = None) -> List[Workout]:
        """Retrieve workouts with optional type and plane filtering"""
        try:
            if workout_type:
                if workout_type == "total_body":
                    return self.workouts[workout_type]
                elif workout_type in self.workouts:
                    if plane:
                        if plane in self.workouts[workout_type]:
                            return self.workouts[workout_type][plane]
                        else:
                            raise ValueError(f"Invalid plane: {plane}")
                    else:
                        return [
                            workout
                            for plane_workouts in self.workouts[workout_type].values()
                            for workout in plane_workouts
                        ]
                else:
                    raise ValueError(f"Invalid workout type: {workout_type}")
            else:
                # Return all workouts
                all_workouts = []
                for type_data in self.workouts.values():
                    if isinstance(type_data, list):
                        all_workouts.extend(type_data)
                    else:
                        for plane_workouts in type_data.values():
                            all_workouts.extend(plane_workouts)
                return all_workouts
                
        except Exception as e:
            raise StorageOperationError(f"Error retrieving workouts: {str(e)}")
            
    def restore_from_backup(self, backup_path: str = None):
        """Restore workout data from backup"""
        try:
            data = super().restore_from_backup(backup_path)
            self.load_data()  # Reload data from restored file
            return data
        except Exception as e:
            raise StorageOperationError(f"Failed to restore from backup: {str(e)}")

class WorkoutGenerator:
    """Handles workout generation logic"""
    def __init__(self, storage_manager: StorageManager):
        self.storage = storage_manager
        self.reset_tracking()
        
    def reset_tracking(self):
        """Reset exercise and pairing tracking"""
        self.used_exercises = set()
        self.program_used_exercises = set()
        
    def _select_unique_exercise(self, pattern: str, plane: str = None) -> Exercise:
        """Select a unique exercise that hasn't been used in the current workout"""
        available_exercises = self.storage.exercise_storage.get_exercises_by_pattern(pattern, plane)
        
        # Filter out already used exercises
        available_exercises = [ex for ex in available_exercises 
                             if ex.name not in self.used_exercises]
        
        if not available_exercises:
            raise ValueError(f"No available {pattern} exercises" + 
                           f" in {plane} plane" if plane else "")
        
        selected = random.choice(available_exercises)
        self.used_exercises.add(selected.name)
        self.program_used_exercises.add(selected.name)
        
        return selected
        
    def _select_total_body_exercise(self) -> Exercise:
        """Select a unique total body exercise"""
        return self._select_unique_exercise("total_body")
        
    def generate_workout(self, workout_type: str, plane: str = None) -> Workout:
        """Generate a workout based on type and plane"""
        self.reset_tracking()
        
        if workout_type == "total_body":
            return self._generate_total_body_workout()
        elif workout_type in ["squat_push", "hinge_pull"]:
            return self._generate_split_workout(workout_type, plane)
        else:
            raise ValueError(f"Invalid workout type: {workout_type}")
            
    def _generate_total_body_workout(self) -> Workout:
        """Generate a total body workout"""
        exercises = {}
        
        # First Triplet: Lower Body Focus
        exercises["1a"] = self._select_unique_exercise("squat")
        exercises["1b"] = self._select_unique_exercise("hinge")
        exercises["1c"] = self._select_unique_exercise("total_body")
        
        # Second Triplet: Upper Body Focus
        exercises["2a"] = self._select_unique_exercise("push", "horizontal")
        exercises["2b"] = self._select_unique_exercise("pull", "horizontal")
        exercises["2c"] = self._select_unique_exercise("total_body")
        
        # Third Triplet: Mixed Focus
        exercises["3a"] = self._select_unique_exercise("push", "vertical")
        exercises["3b"] = self._select_unique_exercise("pull", "vertical")
        exercises["3c"] = self._select_unique_exercise("total_body")
        
        return Workout("total_body", exercises)
        
    def _generate_split_workout(self, workout_type: str, plane: str) -> Workout:
        """Generate a split workout (squat/push or hinge/pull)"""
        exercises = {}
        
        if workout_type == "squat_push":
            # Select three squats for the 'a' positions
            for i in range(3):
                exercises[f"{i+1}a"] = self._select_unique_exercise("squat")
            
            # Select two pushes in the focused plane
            for i in range(2):
                exercises[f"{i+1}b"] = self._select_unique_exercise("push", plane)
            
            # Select one pull in the same plane
            exercises["3b"] = self._select_unique_exercise("pull", plane)
            
        else:  # hinge_pull
            # Select three hinges for the 'a' positions
            for i in range(3):
                exercises[f"{i+1}a"] = self._select_unique_exercise("hinge")
            
            # Select two pulls in the focused plane
            for i in range(2):
                exercises[f"{i+1}b"] = self._select_unique_exercise("pull", plane)
            
            # Select one push in the same plane
            exercises["3b"] = self._select_unique_exercise("push", plane)
        
        # Add total body finishers
        exercises["4a"] = self._select_total_body_exercise()
        exercises["4b"] = self._select_total_body_exercise()
        
        return Workout(workout_type, exercises, plane=plane)
"END OF WOORKOUT MANAGMENT MODULE"


"START OF PROGRAM MANAGEMENT MODULE"
class Program:
    """Represents a complete training program cycle"""
    def __init__(self, weeks: List[Dict], program_id: str = None):
        self.weeks = weeks
        self.id = program_id or str(uuid.uuid4())
        self.date_created = datetime.now().isoformat()
        
    def to_dict(self) -> dict:
        """Convert program to dictionary representation"""
        return {
            "id": self.id,
            "date_created": self.date_created,
            "weeks": [
                {
                    "workouts": [workout.to_dict() for workout in week["workouts"]]
                }
                for week in self.weeks
            ]
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'Program':
        """Create program instance from dictionary"""
        weeks = [
            {
                "workouts": [
                    Workout.from_dict(workout_data)
                    for workout_data in week["workouts"]
                ]
            }
            for week in data["weeks"]
        ]
        
        program = cls(
            weeks=weeks,
            program_id=data["id"]
        )
        program.date_created = data["date_created"]
        return program

class ProgramStorage(BaseJsonStorage):
    """Handles program cycle data storage"""
    def __init__(self, filepath: str):
        super().__init__(filepath)
        self.programs = []
        self.load_data()
        
    def load_data(self):
        """Load program data from storage"""
        data = self.read()
        self.programs = [Program.from_dict(p) for p in data.get("programs", [])]
        
    def save_data(self):
        """Save program data to storage"""
        data = {"programs": [p.to_dict() for p in self.programs]}
        self.write(data)
        
    def add_program(self, program: Program) -> tuple[bool, str]:
        """Add a new program to storage"""
        try:
            self.programs.append(program)
            self.save_data()
            return True, f"Program saved with ID: {program.id}"
        except Exception as e:
            return False, f"Error saving program: {str(e)}"
            
    def remove_program(self, program_id: str) -> tuple[bool, str]:
        """Remove a program from storage"""
        program = next((p for p in self.programs if p.id == program_id), None)
        if program:
            self.programs.remove(program)
            self.save_data()
            return True, f"Removed program: {program_id}"
        return False, "Program not found"
        
    def clear_programs(self) -> tuple[bool, str]:
        """Remove all stored programs"""
        try:
            self.programs.clear()
            self.save_data()
            return True, "All programs cleared"
        except Exception as e:
            return False, f"Error clearing programs: {str(e)}"

class WorkoutProgramGenerator:
    """Handles program cycle generation"""
    def __init__(self, storage_manager: StorageManager):
        self.storage = storage_manager
        self.workout_generator = WorkoutGenerator(storage_manager)
        
    def check_workout_availability(self) -> tuple[bool, str]:
        """Check if sufficient workouts are available for program generation"""
        required = {
            'squat_push': {'vertical': 1, 'horizontal': 1},
            'hinge_pull': {'vertical': 1, 'horizontal': 1},
            'total_body': 2  # One for each week
        }
        
        missing_workouts = []
        
        for workout_type, plane_reqs in required.items():
            if workout_type == 'total_body':
                available = len(self.storage.workout_storage.workouts[workout_type])
                if available < plane_reqs:
                    missing_workouts.append(
                        f"Total Body: need {plane_reqs}, have {available}"
                    )
            else:
                for plane, count in plane_reqs.items():
                    available = len(self.storage.workout_storage.workouts[workout_type][plane])
                    if available < count:
                        missing_workouts.append(
                            f"{workout_type.replace('_', '/')} {plane}: need {count}, have {available}"
                        )
        
        if missing_workouts:
            return False, "Insufficient workouts:\n" + "\n".join(missing_workouts)
        return True, "Sufficient workouts available"
        
    def generate_program(self) -> Program:
        """Generate a complete program cycle"""
        sufficient, message = self.check_workout_availability()
        if not sufficient:
            raise ValueError(message)
            
        self.workout_generator.reset_tracking()
        
        program_structure = {
            1: [  # Week 1
                ("squat_push", "vertical"),
                ("hinge_pull", "vertical"),
                ("total_body", None)
            ],
            2: [  # Week 2
                ("squat_push", "horizontal"),
                ("hinge_pull", "horizontal"),
                ("total_body", None)
            ]
        }
        
        weeks = []
        for week_num in [1, 2]:
            week_workouts = []
            for workout_type, plane in program_structure[week_num]:
                workout = self.workout_generator.generate_workout(workout_type, plane)
                week_workouts.append(workout)
            
            weeks.append({
                "workouts": week_workouts
            })
        
        return Program(weeks)
"END OF PROGRAM MANAGEMENT MODULE"

"START OF UI MODULE"
class StatusDisplay:
    """Displays storage and program statistics with automatic updates"""
    def __init__(self, parent: ttk.Frame, storage_manager: StorageManager):
        self.frame = ttk.LabelFrame(parent, text="System Status", padding="5")
        self.storage = storage_manager
        
        # Create text display
        self.text = tk.Text(self.frame, height=6, width=40)
        self.text.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Initial update
        self.update_display()
        
    def update_display(self):
        """Update status information"""
        try:
            # Enable text widget for updating
            self.text.config(state='normal')
            self.text.delete(1.0, tk.END)
            
            # Exercise counts
            total_exercises = sum(
                len(info["exercises"]) 
                for info in self.storage.exercise_storage.categories.values()
            )
            
            # Workout counts
            total_workouts = sum(
                len(workouts) if isinstance(workouts, list) else
                sum(len(plane_workouts) for plane_workouts in workouts.values())
                for workouts in self.storage.workout_storage.workouts.values()
            )
            
            # Program count
            total_programs = len(self.storage.program_storage.programs)
            
            # Format display text
            status = (
                f"Total Exercises: {total_exercises}\n"
                f"Total Workouts: {total_workouts}\n"
                f"Total Programs: {total_programs}\n\n"
            )
            
            # Add workout breakdown
            status += "Workout Distribution:\n"
            for workout_type, workouts in self.storage.workout_storage.workouts.items():
                if workout_type == "total_body":
                    count = len(workouts)
                    status += f"- Total Body: {count}\n"
                else:
                    type_display = workout_type.replace("_", "/")
                    for plane, plane_workouts in workouts.items():
                        count = len(plane_workouts)
                        status += f"- {type_display} ({plane}): {count}\n"
            
            self.text.insert(tk.END, status)
            self.text.config(state='disabled')
        except Exception as e:
            print(f"Error updating status display: {str(e)}")
        
    def pack(self, **kwargs):
        """Position the frame using pack"""
        self.frame.pack(**kwargs)

class HelpWindow:
    """Help window showing application documentation"""
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("How to Use Elite Personal Training")
        self.window.geometry("800x600")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Add tabs
        self.add_requirements_tab()
        self.add_exercises_tab()
        self.add_workouts_tab()
        self.add_programs_tab()
        self.add_tips_tab()
        
    def create_text_area(self, parent: ttk.Frame) -> tk.Text:
        """Create a standard text area widget"""
        text = tk.Text(parent, wrap=tk.WORD, padx=10, pady=5)
        text.pack(expand=True, fill='both')
        return text
        
    def add_requirements_tab(self):
        """Add the exercise requirements tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Exercise Requirements")
        
        text = self.create_text_area(frame)
        content = """Minimum Exercise Requirements:

Before generating workouts, you need these exercises in your library:

SQUATS (Minimum 6):
- Used in Squat/Push workouts (3 per workout)
- Used in Total Body workouts

HINGES (Minimum 6):
- Used in Hinge/Pull workouts (3 per workout)
- Used in Total Body workouts

PUSH EXERCISES:
- Horizontal Push (Minimum 7)
- Vertical Push (Minimum 7)
- Used in respective plane workouts

PULL EXERCISES:
- Horizontal Pull (Minimum 7)
- Vertical Pull (Minimum 7)
- Used in respective plane workouts

TOTAL BODY (Minimum 25-30):
- 20 exercises needed for Total Body workouts (5 exercises × 4 workouts)
- Additional 8-10 exercises needed for finishers in other workouts
- No exercise repetition allowed within a program cycle
- Used as finishers in all workouts
- Used heavily in Total Body workouts

If you get errors about "no available exercises", check these minimums first!"""
        text.insert("1.0", content)
        text.config(state='disabled')
        
    def add_exercises_tab(self):
        """Add the exercise management tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Adding Exercises")
        
        text = self.create_text_area(frame)
        content = """STEP 1: ADDING EXERCISES

1. Go to "Library Manager" tab
2. Use "Category:" dropdown to select:
   - squat (for squat exercises)
   - hinge (for hinge exercises)
   - total_body (for total body exercises)
   - upper_push_horizontal (for horizontal pushes)
   - upper_push_vertical (for vertical pushes)
   - upper_pull_horizontal (for horizontal pulls)
   - upper_pull_vertical (for vertical pulls)

3. For each exercise:
   - Type name in the "Name:" box
   - Click "Add Exercise"
   - Verify it appears in the list

IMPORTANT:
- Add enough exercises in each category (see Requirements tab)
- Use clear, consistent naming
- Don't duplicate exercise names
- Save your library when done"""
        text.insert("1.0", content)
        text.config(state='disabled')
        
    def add_workouts_tab(self):
        """Add the workout generation tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Creating Workouts")
        
        text = self.create_text_area(frame)
        content = """STEP 2: CREATING WORKOUTS

You need exercises available to create:
- Squat/Push workouts (horizontal and vertical)
- Hinge/Pull workouts (horizontal and vertical)
- Total Body workouts

To create a workout:
1. Go to "Workout Generator" tab
2. Select workout type and plane (if applicable)
3. Click "Generate Workout"
4. Review the generated workout
5. Workout saves automatically when generated

The program requires:
- Balanced exercise selection
- No exercise repetition within workouts
- Appropriate plane focus
- Total body finishers for split workouts

Common Errors:
- "No available exercises": Add more exercises to that category
- "Failed to generate": Check minimum exercise requirements
- If stuck, try clearing workouts and starting over"""
        text.insert("1.0", content)
        text.config(state='disabled')
        
    def add_programs_tab(self):
        """Add the program generation tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Generating Programs")
        
        text = self.create_text_area(frame)
        content = """STEP 3: GENERATING PROGRAMS

Before generating:
- Ensure sufficient exercises in all categories
- Review the exercise requirements
- Consider exercise variety needs

To generate a program:
1. Go to "Program Generator" tab
2. Click "Generate New Program"
3. Review the generated program
4. Program saves automatically

Program Structure:
Week 1:
- Day 1: Squat/Push (Vertical)
- Day 2: Hinge/Pull (Vertical)
- Day 3: Total Body

Week 2:
- Day 1: Squat/Push (Horizontal)
- Day 2: Hinge/Pull (Horizontal)
- Day 3: Total Body

Key Points:
- No exercise repetition across entire program
- Balanced progression between weeks
- Alternating vertical/horizontal focus
- Total body workouts cap each week"""
        text.insert("1.0", content)
        text.config(state='disabled')
        
    def add_tips_tab(self):
        """Add the tips and troubleshooting tab"""
        frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(frame, text="Tips & Troubleshooting")
        
        text = self.create_text_area(frame)
        content = """COMMON ISSUES AND SOLUTIONS

1. Program Won't Generate
   - Check exercise minimums in each category
   - Verify exercises are in correct categories
   - Clear existing programs and try again
   - Add more exercise variety if needed

2. Exercise Issues
   - Use unique, descriptive names
   - Assign to correct categories
   - Check pattern and plane selections
   - Save library after making changes

3. Workout Generation Fails
   - Review exercise requirements
   - Check for sufficient variety
   - Verify pattern/plane assignments
   - Try clearing stored workouts

4. Data Management
   - Save library changes regularly
   - Back up your exercise library
   - Clear old programs periodically
   - Keep exercise names consistent

5. Best Practices
   - Add exercises methodically
   - Test workouts individually
   - Review generated programs fully
   - Plan exercise variety carefully"""
        text.insert("1.0", content)
        text.config(state='disabled')

class ElitePersonalTrainingApp:
    """Main application class handling UI and interaction"""
    def __init__(self, root):
        self.root = root
        self.root.title("Elite Personal Training")
        
        # Initialize storage and generators
        self.storage_manager = StorageManager(os.path.join(os.getcwd(), "elite_pt_data"))
        self.workout_generator = WorkoutGenerator(self.storage_manager)
        self.program_generator = WorkoutProgramGenerator(self.storage_manager)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create main tabs
        self.setup_library_tab()
        self.setup_unified_workout_tab()
        self.setup_program_cycle_tab()
        
        # Show help window on startup
        self.show_help_window()

    def setup_library_tab(self):
        """Setup the exercise library management tab"""
        library_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(library_frame, text="Library Manager")
        
        # Category Selection
        category_frame = ttk.Frame(library_frame)
        category_frame.pack(fill='x', pady=5)
        
        ttk.Label(category_frame, text="Category:").pack(side='left')
        self.category_var = tk.StringVar()
        self.category_combo = ttk.Combobox(
            category_frame,
            textvariable=self.category_var,
            values=list(self.storage_manager.exercise_storage.categories.keys())
        )
        self.category_combo.pack(side='left', expand=True, fill='x', padx=5)
        self.category_combo.bind('<<ComboboxSelected>>', self.update_exercise_list)
        
        # Exercise List
        list_frame = ttk.LabelFrame(library_frame, text="Exercises", padding="5")
        list_frame.pack(expand=True, fill='both', pady=5)
        
        self.exercise_list = tk.Listbox(list_frame, height=10)
        self.exercise_list.pack(side='left', expand=True, fill='both')
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, 
                                command=self.exercise_list.yview)
        scrollbar.pack(side='right', fill='y')
        self.exercise_list['yscrollcommand'] = scrollbar.set
        
        # Add Exercise Frame
        add_frame = ttk.LabelFrame(library_frame, text="Add Exercise", padding="5")
        add_frame.pack(fill='x', pady=5)
        
        ttk.Label(add_frame, text="Name:").pack(side='left')
        self.new_exercise_name = ttk.Entry(add_frame)
        self.new_exercise_name.pack(side='left', expand=True, fill='x', padx=5)
        
        ttk.Button(add_frame, text="Add Exercise", 
                  command=self.add_exercise).pack(side='right')
        
        # Control Buttons
        button_frame = ttk.Frame(library_frame)
        button_frame.pack(fill='x', pady=5)
        
        ttk.Button(button_frame, text="Remove Selected Exercise",
                  command=self.remove_exercise).pack(side='left', expand=True, fill='x', padx=2)
        
        ttk.Button(button_frame, text="Save Library",
                  command=self.save_library).pack(side='left', expand=True, fill='x', padx=2)
        
        ttk.Button(button_frame, text="Load Library",
                  command=self.load_library).pack(side='left', expand=True, fill='x', padx=2)

    def setup_unified_workout_tab(self):
        """Setup the unified workout generation and management tab"""
        unified_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(unified_frame, text="Workout Generator")
        
        # Create left and right frames for side-by-side layout
        left_frame = ttk.Frame(unified_frame)
        right_frame = ttk.Frame(unified_frame)
        
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # Configure grid weights for equal spacing
        unified_frame.grid_columnconfigure(0, weight=1)
        unified_frame.grid_columnconfigure(1, weight=1)
        
        # === LEFT SIDE (Generator) ===
        # Workout Type Selection
        type_frame = ttk.LabelFrame(left_frame, text="Workout Type", padding="5")
        type_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.workout_type = tk.StringVar(value="squat_push")
        ttk.Radiobutton(type_frame, text="Squat/Push", variable=self.workout_type, 
                       value="squat_push").grid(row=0, column=0, padx=5)
        ttk.Radiobutton(type_frame, text="Hinge/Pull", variable=self.workout_type, 
                       value="hinge_pull").grid(row=0, column=1, padx=5)
        ttk.Radiobutton(type_frame, text="Total Body", variable=self.workout_type, 
                       value="total_body").grid(row=0, column=2, padx=5)
        
        # Plane Focus Selection
        self.plane_focus = tk.StringVar(value="horizontal")
        plane_frame = ttk.LabelFrame(left_frame, text="Plane Focus", padding="5")
        plane_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(plane_frame, text="Horizontal", variable=self.plane_focus, 
                       value="horizontal").grid(row=0, column=0, padx=5)
        ttk.Radiobutton(plane_frame, text="Vertical", variable=self.plane_focus, 
                       value="vertical").grid(row=0, column=1, padx=5)

        # Count Input
        count_frame = ttk.Frame(left_frame)
        count_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(count_frame, text="Number of Workouts:").pack(side='left', padx=5)
        self.workout_count = ttk.Entry(count_frame, width=10)
        self.workout_count.pack(side='left', padx=5)
        self.workout_count.insert(0, "1")
        
        # Generate Button
        ttk.Button(left_frame, text="Generate Workout(s)", 
                  command=self.generate_workouts).grid(row=3, column=0, 
                  sticky=(tk.W, tk.E), pady=5)
        
        # Results Area
        results_frame = ttk.LabelFrame(left_frame, text="Generated Workout", padding="5")
        results_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.results_text = tk.Text(results_frame, height=20, width=50)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, 
                                command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text['yscrollcommand'] = scrollbar.set

        # === RIGHT SIDE (Library) ===
        # Same right side code as before...
        list_frame = ttk.LabelFrame(right_frame, text="Stored Workouts", padding="5")
        list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.workout_list = tk.Listbox(list_frame, height=25, width=50)
        self.workout_list.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, 
                                command=self.workout_list.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.workout_list['yscrollcommand'] = scrollbar.set
        
        # Buttons
        button_frame = ttk.Frame(right_frame, padding="5")
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="View Selected", 
                  command=self.view_selected_workout).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Remove Selected",
                  command=self.remove_selected_workout).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Clear All",
                  command=self.clear_all_workouts).grid(row=0, column=2, padx=5)
    
    def update_status(self):
        """Update the status display"""
        if self.status_display:
            self.status_display.update_display()
            
    def generate_workouts(self):
        """Generate multiple workouts based on user selections"""
        try:
            workout_type = self.workout_type.get()
            plane = None if workout_type == "total_body" else self.plane_focus.get()
            count = int(self.workout_count.get())
            
            if count < 1:
                messagebox.showerror("Error", "Please enter a positive number")
                return
                
            generated_workouts = []
            for _ in range(count):
                try:
                    workout = self.workout_generator.generate_workout(workout_type, plane)
                    success, message = self.storage_manager.workout_storage.add_workout(workout)
                    if success:
                        generated_workouts.append(workout)
                    else:
                        messagebox.showerror("Error", message)
                        break
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to generate workout: {str(e)}")
                    break
            
            if generated_workouts:
                # Display the last generated workout
                self.display_workout(generated_workouts[-1])
                self.update_workout_list()
                self.update_status()
                
                # Show success message
                if len(generated_workouts) == 1:
                    messagebox.showinfo("Success", "Workout generated and saved successfully")
                else:
                    messagebox.showinfo("Success", 
                                      f"{len(generated_workouts)} workouts generated and saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate workouts: {str(e)}")
                  
    def setup_program_cycle_tab(self):
        """Setup the program cycle generation tab"""
        cycle_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(cycle_frame, text="Program Generator")
        
        # Split frame into left and right sections
        left_frame = ttk.Frame(cycle_frame)
        right_frame = ttk.Frame(cycle_frame)
        
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        cycle_frame.grid_columnconfigure(0, weight=1)
        cycle_frame.grid_columnconfigure(1, weight=1)
        
        # Left side - Program Controls and List
        controls_frame = ttk.LabelFrame(left_frame, text="Program Controls", padding="5")
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(controls_frame, text="Generate New Program",
                  command=self.generate_program).grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Clear All Programs",
                  command=self.clear_all_programs).grid(row=0, column=1, padx=5, pady=5)
        
        list_frame = ttk.LabelFrame(left_frame, text="Stored Programs", padding="5")
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.program_list = tk.Listbox(list_frame, height=20, width=40)
        self.program_list.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, 
                                command=self.program_list.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.program_list['yscrollcommand'] = scrollbar.set
        
        # Right side - Program View
        view_frame = ttk.LabelFrame(right_frame, text="Program Details", padding="5")
        view_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.program_text = tk.Text(view_frame, height=30, width=70)
        self.program_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(view_frame, orient=tk.VERTICAL, 
                                command=self.program_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.program_text['yscrollcommand'] = scrollbar.set
        
        # Bottom buttons
        button_frame = ttk.Frame(right_frame, padding="5")
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="View Selected",
                  command=self.view_selected_program).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Remove Selected",
                  command=self.remove_selected_program).grid(row=0, column=1, padx=5)

    def update_exercise_list(self, event=None):
        """Update the exercise list display"""
        self.exercise_list.delete(0, tk.END)
        category = self.category_var.get()
        
        if category in self.storage_manager.exercise_storage.categories:
            exercises = sorted(self.storage_manager.exercise_storage.categories[category]["exercises"],
                            key=lambda x: x.name.lower())
            for exercise in exercises:
                self.exercise_list.insert(tk.END, f"{exercise.name} ({exercise.id})")

    def add_exercise(self, category: str, name: str) -> tuple[bool, str]:
        """Add a new exercise to the specified category"""
        try:
            if name.lower() in self.exercise_names:
                return False, "Exercise with this name already exists"

            if category not in self.categories:
                return False, "Invalid category"

            # Generate new ID
            prefix = self.categories[category]["prefix"]
            existing_ids = {ex.id for ex in self.categories[category]["exercises"]}
            counter = 1
            while f"{prefix}{counter}" in existing_ids:
                counter += 1
            new_id = f"{prefix}{counter}"

            # Create and add new exercise
            pattern = category.split('_')[0]
            plane = category.split('_')[-1] if '_' in category else None
            new_exercise = Exercise(name, new_id, pattern, plane)
            self.categories[category]["exercises"].add(new_exercise)
            self.exercise_names.add(name.lower())

            # Save changes
            self.save_data()
            return True, f"Exercise added with ID: {new_id}"
        except Exception as e:
            return False, f"Error adding exercise: {str(e)}"
            
    def remove_exercise(self):
        """Remove exercise with status update"""
        selection = self.exercise_list.curselection()
        if not selection:
            messagebox.showerror("Error", "Please select an exercise to remove")
            return
        
        exercise_text = self.exercise_list.get(selection[0])
        exercise_id = exercise_text.split("(")[-1].strip(")")
        
        category = self.category_var.get()
        success, message = self.storage_manager.exercise_storage.remove_exercise(category, exercise_id)
        
        if success:
            messagebox.showinfo("Success", message)
            self.update_exercise_list()
            self.update_status()
        else:
            messagebox.showerror("Error", message)

    def save_library(self):
        """Save the exercise library to storage"""
        try:
            self.storage_manager.exercise_storage.save_data()
            messagebox.showinfo("Success", "Exercise library saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save library: {str(e)}")

    def load_library(self):
        """Load the exercise library from storage"""
        try:
            self.storage_manager.exercise_storage.load_data()
            self.update_exercise_list()
            messagebox.showinfo("Success", "Exercise library loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load library: {str(e)}")

    def update_workout_list(self):
        """Update the workout list display"""
        self.workout_list.delete(0, tk.END)
        
        workouts = []
        for workout_type, type_data in self.storage_manager.workout_storage.workouts.items():
            if workout_type == "total_body":
                workouts.extend(type_data)
            else:
                for plane_workouts in type_data.values():
                    workouts.extend(plane_workouts)
        
        for workout in sorted(workouts, key=lambda w: w.id):
            display_text = f"{workout.type.replace('_', '/')} "
            if workout.plane:
                display_text += f"({workout.plane}) "
            display_text += f"- {workout.id}"
            self.workout_list.insert(tk.END, display_text)
            
    def generate_workout(self):
        """Generate a new workout based on user selections"""
        try:
            workout_type = self.workout_type.get()
            plane = None if workout_type == "total_body" else self.plane_focus.get()
            
            workout = self.workout_generator.generate_workout(workout_type, plane)
            success, message = self.storage_manager.workout_storage.add_workout(workout)
            
            if success:
                self.display_workout(workout)
                self.update_workout_list()
                messagebox.showinfo("Success", "Workout generated and saved successfully")
            else:
                messagebox.showerror("Error", message)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate workout: {str(e)}")

    def display_workout(self, workout: Workout):
        """Display a workout in the results text area"""
        self.results_text.delete(1.0, tk.END)
        
        header = f"Elite Personal Training Workout\n"
        header += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        header += f"Type: {workout.type.replace('_', '/')}\n"
        if workout.plane:
            header += f"Plane Focus: {workout.plane}\n"
        header += f"Workout ID: {workout.id}\n\nExercises:\n"
        self.results_text.insert(tk.END, header)
        
        sorted_positions = sorted(workout.exercises.keys())
        for pos in sorted_positions:
            exercise = workout.exercises[pos]
            exercise_text = f"{pos}: {exercise.name}"
            if exercise.plane:
                exercise_text += f" ({exercise.pattern} - {exercise.plane})"
            else:
                exercise_text += f" ({exercise.pattern})"
            exercise_text += "\n"
            
            if workout.type == "total_body" and pos in ["1c", "2c", "3c"]:
                exercise_text += "\n"
            elif workout.type != "total_body" and pos == "4a":
                self.results_text.insert(tk.END, "\nFinishers:\n")
            
            self.results_text.insert(tk.END, exercise_text)

    def view_selected_workout(self):
        """Display the selected workout details"""
        selection = self.workout_list.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a workout to view")
            return
            
        workout_id = self.workout_list.get(selection[0]).split(" - ")[-1]
        
        # Search for workout in storage
        workout = None
        for type_data in self.storage_manager.workout_storage.workouts.values():
            if isinstance(type_data, list):
                workout = next((w for w in type_data if w.id == workout_id), None)
            else:
                for plane_workouts in type_data.values():
                    workout = next((w for w in plane_workouts if w.id == workout_id), None)
                    if workout:
                        break
            if workout:
                break
                
        if workout:
            self.display_workout(workout)
        else:
            messagebox.showerror("Error", "Workout not found")

    def remove_selected_workout(self):
        """Remove the selected workout from storage"""
        selection = self.workout_list.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a workout to remove")
            return
            
        if messagebox.askyesno("Confirm Remove", "Are you sure you want to remove this workout?"):
            workout_id = self.workout_list.get(selection[0]).split(" - ")[-1]
            success, message = self.storage_manager.workout_storage.remove_workout(workout_id)
            
            if success:
                self.update_workout_list()
                self.results_text.delete(1.0, tk.END)
                messagebox.showinfo("Success", message)
            else:
                messagebox.showerror("Error", message)

    def clear_all_workouts(self):
        """Clear all stored workouts"""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all workouts?"):
            self.storage_manager.workout_storage.workouts = {
                "squat_push": {"horizontal": [], "vertical": []},
                "hinge_pull": {"horizontal": [], "vertical": []},
                "total_body": []
            }
            self.storage_manager.workout_storage.save_data()
            self.update_workout_list()
            self.results_text.delete(1.0, tk.END)
            self.update_status()
            messagebox.showinfo("Success", "All workouts cleared")

    def update_program_list(self):
        """Update the program list display"""
        self.program_list.delete(0, tk.END)
        for program in self.storage_manager.program_storage.programs:
            created_date = datetime.fromisoformat(program.date_created).strftime("%Y-%m-%d")
            self.program_list.insert(tk.END, f"Program {program.id} ({created_date})")

    def generate_program(self):
        """Generate a new program cycle"""
        try:
            program = self.program_generator.generate_program()
            success, message = self.storage_manager.program_storage.add_program(program)
            
            if success:
                self.display_program(program)
                self.update_program_list()
                self.update_status()
                messagebox.showinfo("Success", "Program generated and saved successfully")
            else:
                messagebox.showerror("Error", message)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate program: {str(e)}")

    def display_program(self, program: Program):
        """Display a program in the program text area"""
        self.program_text.delete(1.0, tk.END)
        
        header = f"Elite Personal Training Program\n"
        header += f"Generated: {program.date_created}\n"
        header += f"Program ID: {program.id}\n\n"
        self.program_text.insert(tk.END, header)
        
        for week_num, week in enumerate(program.weeks, 1):
            self.program_text.insert(tk.END, f"=== WEEK {week_num} ===\n\n")
            
            for day_num, workout in enumerate(week["workouts"], 1):
                self.program_text.insert(tk.END, f"Day {day_num}: ")
                if workout.type == "total_body":
                    self.program_text.insert(tk.END, "Total Body\n")
                else:
                    type_display = "Squat/Push" if workout.type == "squat_push" else "Hinge/Pull"
                    self.program_text.insert(tk.END, f"{type_display} ({workout.plane})\n")
                
                sorted_positions = sorted(workout.exercises.keys())
                for pos in sorted_positions:
                    exercise = workout.exercises[pos]
                    exercise_text = f"{pos}: {exercise.name}"
                    if exercise.plane:
                        exercise_text += f" ({exercise.pattern} - {exercise.plane})"
                    else:
                        exercise_text += f" ({exercise.pattern})"
                    exercise_text += "\n"
                    
                    if workout.type == "total_body" and pos in ["1c", "2c", "3c"]:
                        exercise_text += "\n"
                    elif workout.type != "total_body" and pos == "4a":
                        self.program_text.insert(tk.END, "\nFinishers:\n")
                    
                    self.program_text.insert(tk.END, exercise_text)
                
                self.program_text.insert(tk.END, "\n")

    def view_selected_program(self):
        """Display the selected program details"""
        selection = self.program_list.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a program to view")
            return
            
        program_id = self.program_list.get(selection[0]).split()[1]
        program = next((p for p in self.storage_manager.program_storage.programs 
                       if p.id == program_id), None)
        
        if program:
            self.display_program(program)
        else:
            messagebox.showerror("Error", "Program not found")

    def remove_selected_program(self):
        """Remove the selected program from storage"""
        selection = self.program_list.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a program to remove")
            return
            
        if messagebox.askyesno("Confirm Remove", "Are you sure you want to remove this program?"):
            program_id = self.program_list.get(selection[0]).split()[1]
            success, message = self.storage_manager.program_storage.remove_program(program_id)
            
            if success:
                self.update_program_list()
                self.program_text.delete(1.0, tk.END)
                messagebox.showinfo("Success", message)
            else:
                messagebox.showerror("Error", message)

    def clear_all_programs(self):
        """Clear all stored programs"""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all programs?"):
            success, message = self.storage_manager.program_storage.clear_programs()
            if success:
                self.update_program_list()
                self.program_text.delete(1.0, tk.END)
                self.update_status()
                messagebox.showinfo("Success", message)
            else:
                messagebox.showerror("Error", message)
    def show_help_window(self):
        """Show the application help window"""
        help_window = HelpWindow(self.root)
        
        # Add help button to main window if it doesn't exist
        if not hasattr(self, 'help_button'):
            self.help_button = ttk.Button(
                self.root,
                text="Help",
                command=self.show_help_window
            )
            self.help_button.pack(side=tk.TOP, anchor=tk.NE, padx=10, pady=5)
"END OF UI MODULE"

    
if __name__ == "__main__":
    # Create root window
    root = tk.Tk()
    
    # Set window size and position
    window_width = 1200
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    # Create the application instance
    app = ElitePersonalTrainingApp(root)
    
    # Create top frame for status and help
    top_frame = tk.Frame(root)
    top_frame.pack(fill='x', padx=10, pady=5)
    
    # Add status display to top frame and store reference
    app.status_display = StatusDisplay(top_frame, app.storage_manager)
    app.status_display.pack(side='left', expand=True, fill='x')
    
    # Add help button to top frame
    help_button = tk.Button(
        top_frame,
        text="Help",
        command=app.show_help_window
    )
    help_button.pack(side='right', padx=5)
    
    # Start application
    root.mainloop()
