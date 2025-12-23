from typing import TypedDict, Literal, Optional, Union, Dict, List, Any
from typing_extensions import NotRequired

class Point(TypedDict):
    """Represents a clicked point with x, y coordinates."""
    x: int
    y: int

class TileShape(TypedDict):
    """Represents the shape of a tile as [width, height]."""
    # This is actually a list of two integers, but we'll type it as such
    pass

# We'll use List[int] for tile_shape since it's always [width, height]

class UserOptions(TypedDict):
    """User information and preferences."""
    email: NotRequired[Optional[str]]
    name: NotRequired[Optional[str]]
    anonymize: NotRequired[bool]
    experience: NotRequired[Optional[int]]

class Progress(TypedDict):
    """Progress information for a session."""
    name: str
    step: Union[int, float]
    total_steps: Union[int, float]

class ProcessResult(TypedDict):
    """Result data from a completed process."""
    tile_shape: NotRequired[List[int]]  # [width, height]

class ProcessInfo(TypedDict):
    """Information about a process state."""
    state: Literal["Processing", "Done", "Cancel"]
    result: NotRequired[ProcessResult]

class Processes(TypedDict):
    """Container for different process states."""
    initial_image_setup: NotRequired[ProcessInfo]

class SessionOptions(TypedDict, total=False):
    """All possible session options. Uses flexible typing due to mixed flat/nested structure."""
    # The actual JSON has mixed structures, so we use Any for flexibility
    user: UserOptions
    
    # These are used as dictionary keys, so we need string keys with Any values
    # to handle both "image_select.filename" and nested {"image_select": {"filename": "..."}}
    
# More flexible approach for the actual data structure
SessionOptionsFlexible = Dict[str, Any]

class SessionData(TypedDict):
    """Complete session data structure."""
    counter: int
    progress: Progress
    processes: Processes
    options: NotRequired[SessionOptionsFlexible]
    
    # Legacy field that appears in some sessions
    clicked_points: NotRequired[List[Point]]

# The main type for the entire JSON file
SessionsData = Dict[str, SessionData]  # UUID -> SessionData

# Utility functions for type-safe access
def get_min_pore_size(session: SessionData) -> Optional[int]:
    """Extract min_pore_size from session options."""
    if "options" not in session:
        return None
    
    options = session["options"]
    
    # Check flattened key first
    if hasattr(options, 'get'):
        flat_key = "initial_image_setup.min_pore_size"
        if flat_key in options:
            value = options[flat_key]
            return value if isinstance(value, int) else None
    
    # Check nested structure
    initial_setup = options.get("initial_image_setup", {})
    if isinstance(initial_setup, dict):
        value = initial_setup.get("min_pore_size")
        return value if isinstance(value, int) else None
    
    return None

def get_tile_shape(session: SessionData) -> Optional[List[int]]:
    """Extract tile_shape from session options."""
    if "options" not in session:
        return None
    
    options = session["options"]
    
    # Check flattened key first
    if hasattr(options, 'get'):
        flat_key = "initial_image_setup.tile_shape"
        if flat_key in options:
            value = options[flat_key]
            return value if isinstance(value, list) else None
    
    # Check nested structure
    initial_setup = options.get("initial_image_setup", {})
    if isinstance(initial_setup, dict):
        value = initial_setup.get("tile_shape")
        return value if isinstance(value, list) else None
    
    return None

def get_clicked_points(session: SessionData) -> List[Point]:
    """Extract clicked points from session, checking multiple locations."""
    points: List[Point] = []
    
    # Check root level (legacy)
    if "clicked_points" in session:
        points.extend(session["clicked_points"])
    
    # Check options
    if "options" in session:
        options = session["options"]
        
        # Check flat structure
        if "params_select.clicked_points" in options:
            value = options["params_select.clicked_points"]
            if isinstance(value, list):
                points.extend(value)
        
        # Check nested structure
        params_select = options.get("params_select", {})
        if isinstance(params_select, dict) and "clicked_points" in params_select:
            value = params_select["clicked_points"]
            if isinstance(value, list):
                points.extend(value)
    
    return points

def get_user_experience(session: SessionData) -> Optional[int]:
    """Extract user experience level from session."""
    if "options" not in session:
        return None
    
    user_options = session["options"].get("user", {})
    if isinstance(user_options, dict):
        return user_options.get("experience")
    
    return None

def get_selected_filename(session: SessionData) -> Optional[str]:
    """Extract selected image filename from session."""
    if "options" not in session:
        return None
    
    options = session["options"]
    if "image_select.filename" in options:
        return options["image_select.filename"]
    
    # Check nested structure
    image_select = options.get("image_select", {})
    if isinstance(image_select, dict):
        return image_select.get("filename")
    
    return None

# Example usage and validation
def validate_session_data(data: Any) -> SessionsData:
    """Validate and return properly typed session data."""
    if not isinstance(data, dict):
        raise ValueError("Session data must be a dictionary")
    
    # Basic validation - could be expanded
    for session_id, session_data in data.items():
        if not isinstance(session_id, str):
            raise ValueError(f"Session ID must be string, got {type(session_id)}")
        
        if not isinstance(session_data, dict):
            raise ValueError(f"Session data must be dict, got {type(session_data)}")
        
        required_fields = ["counter", "progress", "processes"]
        for field in required_fields:
            if field not in session_data:
                raise ValueError(f"Missing required field '{field}' in session {session_id}")
    
    return data  # type: ignore

# Example usage
if __name__ == "__main__":
    import json
    
    # Load and validate session data
    def load_session_data(file_path: str) -> SessionsData:
        """Load session data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return validate_session_data(data)
    
    # Example of analyzing session data
    def analyze_sessions(sessions: SessionsData) -> None:
        """Analyze session data and print statistics."""
        total_sessions = len(sessions)
        completed_sessions = 0
        experience_levels = []
        
        for session_id, session in sessions.items():
            # Check if session is completed
            if session["processes"].get("initial_image_setup", {}).get("state") == "Done":
                completed_sessions += 1
            
            # Collect experience levels
            experience = get_user_experience(session)
            if experience is not None:
                experience_levels.append(experience)
            
            # Print session summary
            filename = get_selected_filename(session)
            tile_shape = get_tile_shape(session)
            min_pore_size = get_min_pore_size(session)
            clicked_points = get_clicked_points(session)
            
            print(f"Session {session_id[:8]}...")
            print(f"  File: {filename}")
            print(f"  Tile shape: {tile_shape}")
            print(f"  Min pore size: {min_pore_size}")
            print(f"  Clicked points: {len(clicked_points)}")
            print(f"  Experience: {experience}")
            print()
        
        print(f"Total sessions: {total_sessions}")
        print(f"Completed sessions: {completed_sessions}")
        if experience_levels:
            avg_experience = sum(experience_levels) / len(experience_levels)
            print(f"Average experience: {avg_experience:.1f}")

    # Uncomment to test with actual file:
    # sessions = load_session_data("session.json")
    # analyze_sessions(sessions)
