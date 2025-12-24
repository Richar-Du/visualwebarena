export DATASET=visualwebarena

export CLASSIFIEDS="http://18.216.88.140:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export SHOPPING="http://18.216.88.140:7770"
export REDDIT="http://18.216.88.140:9999"
export WIKIPEDIA="http://18.216.88.140:8888"

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

# python scripts/generate_test_data.py

# 获取网页自动登录的cookie
# bash prepare.sh

python run_multi_agent_eval.py --config_file ./config_vlm.json --test_config_base_dir config_files/vwa/test_shopping  --test_start_idx 10 --test_end_idx 20 --result_dir result/test10 --max_steps 10