"""Context summary generation for maintaining agent awareness."""

from typing import Any, Dict, List, Optional
import numpy as np

from PIL import Image

from browser_env import Action, action2str
from browser_env.utils import Observation, pil_to_b64
from llms import lm_config, call_llm
from ..prompts.prompt_loader import load_prompt_template
from ..utils import is_multimodal_model


class SummaryGenerator:
    """Generates context summaries for agent coordination."""

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        # Check if the model supports multimodal inputs
        self.is_multimodal = is_multimodal_model(lm_config.model)

    def generate_summary(
        self,
        user_goal: str,
        current_summary: Optional[str],
        observations: List[Observation],
        actions: List[Action],
        intentions: List[str],
    ) :
        """Generate a comprehensive context summary.

        Args:
            user_goal: Original user goal
            observations: List of all observations
            actions: List of all actions taken
            intentions: List of all intentions

        Returns:
            Generated context summary string
        """
        # Extract key information
        total_steps = len(actions)

        all_his_intentions = self._get_history_intention_text(intentions)
        all_his_actions = self._get_history_action_text(actions)

        # Get recent context O_{t-2}, I_{t-1}, A_{t-1}, O_{t-1}, I_{t}, A_{t}, O_{t}
        recent_observations = observations[-3:] if len(observations) >= 3 else observations
        recent_actions = actions[-2:] if len(actions) >= 2 else actions
        recent_intentions = intentions[-2:] if len(intentions) >= 2 else intentions

        # Generate summary using prompt template
        if current_summary is None:
            prompt = load_prompt_template(
                "context_agent",
                "init_summary_generation",
                user_goal=user_goal,
            )
        else:
            prompt = load_prompt_template(
                "context_agent",
                "summary_generation",
                user_goal=user_goal,
                current_summary=current_summary,
            )

        messages = self._build_messages(
            prompt, recent_observations, recent_actions, recent_intentions
        )

        try:
            summary = call_llm(
                self.lm_config, messages
            ).strip()
        except Exception as e:
            # Fallback summary
            print(f"caught exception {e}")
            print("Context Agent call_llm failed, use fallback summary")
            summary = f"""Task execution after {total_steps} steps."""

        return summary, all_his_intentions, all_his_actions

    def _get_obs_message(self, obs: Observation, idx: int):
        """Extract text from observation, truncating if necessary."""
        obs_text = obs.get("text", "")
        if obs_text:
            truncated_text = obs_text[:800] + "..." if len(obs_text) > 800 else obs_text
            obs_text = f"Page {idx} content: \n{truncated_text}\n"
        else:
            obs_text = f"Page {idx}: None\n"

        # Prefer image_raw (pure screenshot) for Context Agent visual analysis
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

    def _build_messages(
            self,
            prompt_text: str,
            recent_observations: List[Observation],
            recent_actions: List[Action],
            recent_intentions: List[str]
    ):
        """Build recent context messages for summarization.

        Args:
            prompt_text: The base prompt text
            recent_observations: List of recent observations
            recent_actions: List of recent actions
            recent_intentions: List of recent intentions

        Returns:
            List of message dictionaries for LLM input
        """
        content = []
        obs_num = len(recent_observations)
        content += self._get_obs_message(recent_observations[0], 1)
        if obs_num > 1:
            for idx, (intention, act, obs) in enumerate(zip(recent_intentions, recent_actions, recent_observations[1:])):
                content += self._get_intention_message(intention, idx+2)
                content += self._get_action_message(act, idx+2)
                content += self._get_obs_message(obs, idx+2)

        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        return messages

