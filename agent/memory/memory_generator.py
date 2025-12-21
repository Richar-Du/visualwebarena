"""Memory Generator for creating experiences from trajectories."""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json

import numpy as np
from PIL import Image

from browser_env.utils import Observation, pil_to_b64

from llms import lm_config, call_llm
from ..prompts.prompt_loader import load_prompt_template
from browser_env import (
    Action,
    action2str,
)


class MemoryGenerator:
    """Memory Generator for creating experiences from trajectories.

    Implements iterative memory generation with two-phase approach:
    1. Initial memory generation from early trajectory
    2. Iterative refinement with full trajectory
    """

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config

    def generate_memory_from_trajectory(self,
                                       user_goal: str,
                                       observations: List[Observation],
                                       intentions: List[str],
                                       actions: List[Action],
                                       task_completed: bool,
                                       window_size: int = 3
                                    ) -> Dict[str, Any]:
        """Generate memory from complete trajectory.

        Args:
            user_goal: Original user goal
            intentions: List of intentions generated
            actions: List of actions taken
            task_completed: Whether the task was completed successfully

        Returns:
            Generated memory dictionary
        """
        if len(actions) == 0:
            return {}


        assert(len(intentions) == len(actions))
        assert( (len(observations)-1) == len(actions))


        # Phase 1: Initial memory generation from first part of trajectory
        memory = self._generate_initial_memory(
            user_goal,
            observations[:window_size+1], # 多包涵一个初始页
            intentions[:window_size],
            actions[:window_size],
            task_completed
        )

        memory.update({
            "success_rate": 1.0 if task_completed else 0.0,
            "intentions_count": len(intentions),
            "actions_count": len(actions),
        })

        # Phase 2: Iterative refinement with full trajectory
        start_idx = window_size
        while start_idx < len(actions):
            end_idx = min(start_idx + window_size, len(actions))

            partial_observations = observations[start_idx:end_idx+1] # 多包涵一个结束页
            partial_intentions = intentions[start_idx:end_idx]
            partial_actions = actions[start_idx:end_idx]
            history_intentions = intentions[:start_idx]
            history_actions = actions[:start_idx]

            memory = self._refine_memory_with_trajectory(
                memory,
                user_goal,
                partial_observations,
                partial_intentions,
                partial_actions,
                history_intentions,
                history_actions,
                start_idx,
                task_completed
            )
            start_idx = end_idx

        return memory

    def _generate_initial_memory(self,
                                 user_goal: str,
                                 partial_observations: List[Observation],
                                 partial_intentions: List[str],
                                 partial_actions: List[Action],
                                 task_completed: bool
                                 ) -> Dict[str, Any]:
        """Generate initial memory from partial trajectory."""
        try:
            # Generate initial memory using LLM
            prompt = load_prompt_template(
                "memory_generator",
                "memory_initial_generation",
                user_goal=user_goal,
                task_completed=task_completed
            )

            messages = self._get_trajectory_message(
                prompt,
                partial_observations,
                partial_intentions,
                partial_actions,
                start_idx=0
            )


            response = call_llm(
                self.lm_config, messages
            ).replace("```json", "").replace("```", "").strip()

            # Parse response
            try:
                memory_data = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
                # Fallback parsing if response is not valid JSON
                memory_data = self._parse_memory_response(response)

            # Ensure required fields
            return {
                "title": memory_data.get("title", f"Task: {user_goal[:50]}..."),
                "description": memory_data.get("description", f"Experience with task: {user_goal}"),
                "content": "",
                "phase": "initial"
            }

        except Exception as e:
            print(f"Error in initial memory generation: {e}")
            # Fallback memory
            return {
                "title": f"Task Experience: {user_goal[:50]}...",
                "description": f"Automatically generated memory for: {user_goal}",
                "content": f"Experience with task: {user_goal}",
                "phase": "initial_fallback"
            }

    def _refine_memory_with_trajectory(self,
                                           initial_memory: Dict[str, Any],
                                           user_goal: str,
                                           partial_observations: List[Observation],
                                           partial_intentions: List[str],
                                           partial_actions: List[Action],
                                           history_intentions: List[str],
                                           history_actions: List[Action],
                                           start_idx: int,
                                           task_completed: bool) -> Dict[str, Any]:
        """Refine initial memory with full trajectory data."""


        try:

            history_intention_text = self._get_history_intention_text(history_intentions)
            history_action_text = self._get_history_action_text(history_actions)

            # Generate refined memory using LLM
            prompt = load_prompt_template(
                "memory_generator",
                "memory_refinement",
                initial_memory=initial_memory,
                user_goal=user_goal,
                history_intentions=history_intention_text,
                history_actions=history_action_text,
                task_completed=task_completed
            )

            messages = self._get_trajectory_message(
                prompt,
                partial_observations,
                partial_intentions,
                partial_actions,
                start_idx=start_idx
            )

            response = call_llm(
                self.lm_config, messages
            ).replace("```json", "").replace("```", "").strip()

            # Parse refined memory data
            try:
                refined_data = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
                refined_data = self._parse_memory_response(response)

            # Merge with initial memory
            refined_memory = initial_memory.copy()
            refined_memory.update({
                "title": refined_data.get("title", initial_memory["title"]),
                "description": refined_data.get("description", initial_memory["description"]),
                "content": refined_data.get("content", initial_memory["content"]),
                "phase": "refined"
            })

            return refined_memory

        except Exception as e:
            print(f"Error in memory refinement: {e}")
            # Return initial memory with minimal refinement
            refined_memory = initial_memory.copy()
            refined_memory.update({
                "phase": "refined_fallback",
            })
            return refined_memory

    def _get_obs_message(self, obs: Observation, idx: int):
        """Extract text from observation, truncating if necessary."""
        obs_text = obs.get("text", "")
        if obs_text:
            truncated_text = obs_text[:800] + "..." if len(obs_text) > 800 else obs_text
            obs_text = f"Page {idx} content: \n{truncated_text}\n"
        else:
            obs_text = f"Page {idx}: None\n"

        # Prefer image_raw (pure screenshot) for memory generation visual analysis
        obs_image = obs.get("image_raw")
        if obs_image is None:
            obs_image = obs.get("image")

        if isinstance(obs_image, np.ndarray):
            obs_image = Image.fromarray(obs_image)

        content = [
            {"type": "text", "text": f"Page {idx} screenshot: "},
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(obs_image)}
            },
            {"type": "text", "text": obs_text},
        ]

        return content

    def _get_intention_message(self, intention: str, idx: int):
        """Extract text from intention, truncating if necessary."""
        content = [
            {"type": "text", "text": self._get_intention_text(intention, idx)},
        ]
        return content

    def _get_action_message(self, action: Action, idx: int):
        """Extract text from action, truncating if necessary."""
        content = [
            {"type": "text", "text": self._get_action_text(action, idx)},
        ]

        return content

    def _get_trajectory_message(self,
                        prompt_text: str,
                        partial_observations: List[Observation],
                        partial_intentions: List[str],
                        partial_actions: List[Action],
                        start_idx: int
        ) -> List[Dict[str, Any]]:
        """Generate trajectory message for LLM input."""
        content = []
        obs_num = len(partial_observations)
        content += self._get_obs_message(partial_observations[0], start_idx)
        if obs_num > 1:
            for i, (intention, action, obs) in enumerate(zip(partial_intentions, partial_actions, partial_observations[1:]), 1):
                content += self._get_intention_message(intention, start_idx+i)
                content += self._get_action_message(action, start_idx+i)
                content += self._get_obs_message(obs, start_idx+i)

        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        return messages

    def _get_intention_text(self, intention: str, idx: int) -> str:
        """Extract text from intention, truncating if necessary."""
        if intention:
            # Truncate for brevity but keep meaningful content
            truncated_intention = intention[:200] + "..." if len(intention) > 200 else intention
            return f"Intention {idx}: {truncated_intention}"
        return f"Intention {idx}: None"

    def _get_action_text(self, action: Action, idx: int) -> str:
        """Extract text from action, truncating if necessary."""
        action_str = action2str(action, "som", "")
        action_text = f"Action {idx}: {action_str}"
        return action_text

    def _get_history_intention_text(self, intentions: List[str]) -> str:
        """Extract text from intentions, truncating if necessary."""
        intention_texts = [self._get_intention_text(intention, i+1) for i, intention in enumerate(intentions)]
        return "\n".join(intention_texts)

    def _get_history_action_text(self, actions: List[Action]) -> str:
        """Extract text from actions, truncating if necessary."""
        action_texts = [self._get_action_text(action, i+1) for i, action in enumerate(actions)]
        return "\n".join(action_texts)

    def _parse_memory_response(self, response: str) -> Dict[str, Any]:
        """Parse memory response when JSON parsing fails with a simplified approach.

        Args:
            response: Raw text response from LLM

        Returns:
            Dictionary with parsed memory components
        """
        import re

        # Initialize result with default values
        result = {
            'title': '',
            'description': '',
            'content': '',
        }

        if not response:
            return result

        # Normalize response
        response = response.strip()
        lines = response.split('\n')

        # Check for possible JSON fragment first
        json_match = re.search(r'(\{[^}]*\})', response, re.DOTALL)
        if json_match:
            try:
                json_data = json.loads(json_match.group(1))
                for key in result.keys():
                    if key in json_data:
                        result[key] = json_data[key]
            except json.JSONDecodeError:
                pass

        # Process each line for key-value pairs or section headers
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for key:value format
            if ':' in line and not line.startswith('-') and not line.startswith('*'):
                parts = line.split(':', 1)
                key = parts[0].strip().lower()
                value = parts[1].strip() if len(parts) > 1 else ''

                if 'title' in key and not result['title']:
                    result['title'] = value
                elif 'description' in key and not result['description']:
                    result['description'] = value
                elif 'content' in key and not result['content']:
                    result['content'] = value


        # Set defaults if still empty
        if not result['title'] and lines:
            result['title'] = lines[0][:200].strip()

        if not result['content']:
            result['content'] = response[:500].strip() + ('...' if len(response) > 500 else '')

        return result
