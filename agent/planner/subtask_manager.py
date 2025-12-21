"""Subtask Manager for tracking subtask execution state and progress."""

from typing import Any, Dict, List, Optional
from enum import Enum


class SubtaskStatus(Enum):
    """Status of a subtask."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVISED = "revised"  # Subtask was revised and replaced


class SubtaskManager:
    """Manages subtask state tracking and progression.
    
    This class provides explicit state tracking for subtasks,
    eliminating the need for LLM to infer the current subtask from observation.
    
    Key responsibilities:
    1. Track which subtask is currently being executed (via index)
    2. Mark subtasks as completed when Reflector confirms
    3. Handle subtask revision when needed
    4. Provide current subtask information to other agents
    """

    def __init__(self) -> None:
        """Initialize the subtask manager."""
        self.subtasks: List[str] = []
        self.subtask_statuses: List[SubtaskStatus] = []
        self.current_index: int = 0
        self.revision_history: List[Dict[str, Any]] = []
        self.initialized: bool = False

    def initialize(self, subtasks: List[str]) -> None:
        """Initialize with a list of subtasks from task decomposition.
        
        Args:
            subtasks: List of decomposed subtasks
        """
        self.subtasks = subtasks.copy()
        self.subtask_statuses = [SubtaskStatus.PENDING] * len(subtasks)
        self.current_index = 0
        self.revision_history = []
        self.initialized = True
        
        # Mark the first subtask as in progress
        if self.subtasks:
            self.subtask_statuses[0] = SubtaskStatus.IN_PROGRESS
            print(f"📋 SubtaskManager: Initialized with {len(subtasks)} subtasks")
            print(f"📋 Current subtask: {self.get_current_subtask()}")

    def get_current_subtask(self) -> str:
        """Get the current subtask being executed.
        
        Returns:
            Current subtask string, or empty string if no subtasks
        """
        if not self.subtasks or self.current_index >= len(self.subtasks):
            return ""
        return self.subtasks[self.current_index]

    def get_current_index(self) -> int:
        """Get the current subtask index.
        
        Returns:
            Current subtask index
        """
        return self.current_index

    def get_all_subtasks(self) -> List[str]:
        """Get all subtasks.
        
        Returns:
            List of all subtasks
        """
        return self.subtasks.copy()

    def get_subtask_statuses(self) -> List[SubtaskStatus]:
        """Get status of all subtasks.
        
        Returns:
            List of subtask statuses
        """
        return self.subtask_statuses.copy()

    def mark_current_completed(self) -> bool:
        """Mark the current subtask as completed and advance to next.
        
        Called by Coordinator when Reflector indicates subtask_completed=True.
        
        Returns:
            True if advanced to next subtask, False if all subtasks completed
        """
        if not self.subtasks or self.current_index >= len(self.subtasks):
            return False

        # Mark current as completed
        self.subtask_statuses[self.current_index] = SubtaskStatus.COMPLETED
        completed_subtask = self.subtasks[self.current_index]
        print(f"✅ SubtaskManager: Completed subtask {self.current_index + 1}: {completed_subtask[:50]}...")

        # Advance to next subtask
        self.current_index += 1

        # Mark next subtask as in progress if exists
        if self.current_index < len(self.subtasks):
            self.subtask_statuses[self.current_index] = SubtaskStatus.IN_PROGRESS
            print(f"📋 SubtaskManager: Now working on subtask {self.current_index + 1}: {self.get_current_subtask()[:50]}...")
            return True
        else:
            print(f"🎉 SubtaskManager: All {len(self.subtasks)} subtasks completed!")
            return False

    def revise_current_subtask(self, revised_subtask: str) -> None:
        """Revise the current subtask with a new version.
        
        Called when Reflector indicates subtask_needs_revision=True
        and provides a revised subtask.
        
        Args:
            revised_subtask: The revised subtask to replace current one
        """
        if not self.subtasks or self.current_index >= len(self.subtasks):
            return

        old_subtask = self.subtasks[self.current_index]
        
        # Record revision history
        self.revision_history.append({
            "index": self.current_index,
            "original": old_subtask,
            "revised": revised_subtask,
        })

        # Update the subtask
        self.subtasks[self.current_index] = revised_subtask
        self.subtask_statuses[self.current_index] = SubtaskStatus.IN_PROGRESS  # Keep in progress
        
        print(f"🔄 SubtaskManager: Revised subtask {self.current_index + 1}")
        print(f"   Old: {old_subtask[:50]}...")
        print(f"   New: {revised_subtask[:50]}...")

    def insert_subtask_after_current(self, new_subtask: str) -> None:
        """Insert a new subtask after the current one.
        
        Useful when revision determines additional steps are needed.
        
        Args:
            new_subtask: New subtask to insert
        """
        if not self.subtasks:
            self.subtasks.append(new_subtask)
            self.subtask_statuses.append(SubtaskStatus.PENDING)
            return

        insert_index = self.current_index + 1
        self.subtasks.insert(insert_index, new_subtask)
        self.subtask_statuses.insert(insert_index, SubtaskStatus.PENDING)
        
        print(f"➕ SubtaskManager: Inserted new subtask at position {insert_index + 1}: {new_subtask[:50]}...")

    def all_completed(self) -> bool:
        """Check if all subtasks are completed.
        
        Returns:
            True if all subtasks are completed
        """
        if not self.subtasks:
            return False
        return self.current_index >= len(self.subtasks)

    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information.
        
        Returns:
            Dictionary with progress details
        """
        completed_count = sum(1 for s in self.subtask_statuses if s == SubtaskStatus.COMPLETED)
        
        return {
            "current_index": self.current_index,
            "total_subtasks": len(self.subtasks),
            "completed_count": completed_count,
            "current_subtask": self.get_current_subtask(),
            "all_completed": self.all_completed(),
            "all_subtasks": self.subtasks,
            "statuses": [s.value for s in self.subtask_statuses],
        }

    def reset(self) -> None:
        """Reset the subtask manager for a new task."""
        self.subtasks = []
        self.subtask_statuses = []
        self.current_index = 0
        self.revision_history = []
        self.initialized = False

    def __repr__(self) -> str:
        """String representation for debugging."""
        if not self.subtasks:
            return "SubtaskManager(empty)"
        
        current = self.get_current_subtask()[:30] + "..." if len(self.get_current_subtask()) > 30 else self.get_current_subtask()
        return f"SubtaskManager(index={self.current_index}/{len(self.subtasks)}, current='{current}')"

