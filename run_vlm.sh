#!/bin/bash

# Simple Multi-Agent Runner Script
# Usage: ./run_ma_simple.sh

export DATASET=visualwebarena

export DATASET=visualwebarena

export CLASSIFIEDS="<your_classifieds_domain>:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"  # Default reset token for classifieds site, change if you edited its docker-compose.yml
export SHOPPING="<your_shopping_site_domain>:7770"
export REDDIT="<your_reddit_domain>:9999"
export WIKIPEDIA="<your_wikipedia_domain>:8888"
export HOMEPAGE="<your_homepage_domain>:4399"

export SHOPPING_ADMIN="<your_e_commerce_cms_domain>:7780/admin"
export GITLAB="<your_gitlab_domain>:8023"
export MAP="<your_map_domain>:3000"

# Environment variables (modify as needed)
# export OPENAI_API_KEY=sk-ba12564bebdb4f129f91944b55147971
# export OPENAI_BASE_URL=https://api.deepseek.com/v1

# export OPENAI_API_KEY=sk-80a5eac8321a4aa694552fd0f147437e
# export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

export OPENAI_API_KEY=sk-KJhxTaZUuCyi92U1I4ZOOpK173c94Qlbs9uGvc7BdCEro9wr
export OPENAI_API_KEY=sk-QqeaDJ148kH7R5s2dBT1W5nCHu9PGXAKl8vD3VBYiBe5j45d
export OPENAI_BASE_URL=https://aigc.x-see.cn/v1

export HF_ENDPOINT=https://hf-mirror.com

# Run the multi-agent script
python run_multi_agent.py \
  --start_url "http://18.216.88.140:7770/" \
  --intent "Show me the first item with round cookies in the \"ice cream sandwiches\" search results by descending relevance." \
  --max_steps 10 \
  --config_file config_vlm.json


# python run_multi_agent.py \
#   --start_url "https://www.baidu.com/" \
#   --intent "Please search for the term 'Large Model', navigate to its details page, extract the first sentence of the introduction, and send it to me." \
#   --max_steps 10 \
#   --config_file config_vlm.json