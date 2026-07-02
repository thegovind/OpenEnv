# OpenEnv examples

Runnable examples for building, using, evaluating, and collecting data from
OpenEnv environments. Most scripts accept `--help`. Examples that need a running
environment (Docker image or HF Space) say so in their module docstring.

> New here? Start with `openenv_introduction_quickstart.ipynb`, then
> `local_echo_env.py`, then `echo_mcp_demo.py`.

## Environment quickstarts and demos

| Script | What it shows |
| --- | --- |
| `local_echo_env.py` | The simplest usage: `EchoEnv.from_docker_image()` |
| `local_coding_env.py` | `CodingEnv.from_docker_image()` |
| `local_git_env.py` | `GitEnv.from_docker_image()` |
| `echo_mcp_demo.py` | MCP tool usage via the canonical simulation-mode pattern |
| `opencode_env_simple.py` | End-to-end opencode task: write `binary_search.py` and verify |
| `atari_simple.py` | Atari environment usage |
| `connect4.py` | Connect Four game environment |
| `snake_simple.py` | Interactive Snake game player |
| `openspiel_simple.py` | Minimal OpenSpiel environment usage |
| `openspiel_all_games.py` | All six OpenSpiel games integrated with OpenEnv |
| `textarena_simple.py` | Quickstart for the generic TextArena environment |
| `sumo_rl_simple.py` | SUMO-RL environment usage |
| `finrl_simple.py` | FinRL environment usage |
| `unity_simple.py` | Unity ML-Agents environment usage |
| `wildfire.py` | Wildfire environment usage |
| `repl_oolong_simple.py` | REPL + Oolong with recursive LLM calls (RLM paradigm) |
| `repl_with_llm.py` | REPL environment with LLM integration |
| `tbench2_env_simple.py` | Terminal-Bench 2 runner (local mode) |
| `daytona_tbench2_simple.py` | Terminal-Bench 2 runner (Daytona mode) |
| `daytona_tbench2_concurrent.py` | Spawn N environments concurrently on Daytona |
| `openapp_example.py` | OpenApp environment usage |
| `openapp_recording_demo.py` | OpenApp interactions optimized for screen recording |

## LLM inference examples

Drive an environment with a hosted or OpenAI-compatible LLM.

| Script | What it shows |
| --- | --- |
| `coding_env_inference.py` | Solve a coding task with a hosted LLM (HF Inference) |
| `finqa_inference.py` | Play FinQA with any OpenAI-compatible API |
| `kernrl_inference.py` | Optimize a GPU kernel with a hosted LLM |
| `poker_inference.py` | Compare multiple LLMs playing Kuhn Poker |
| `textarena_wordle_inference.py` | Play TextArena Wordle with a hosted LLM |
| `browsergym_example.py` | BrowserGym MiniWoB with a model deciding the next action |

## Harness and evaluation (RFC 005)

Drive rollouts through the `openenv.core.harness` runtime. White-box adapters
let the trainer own model sampling; black-box adapters wrap an opaque agent and
are evaluation-only.

| Script | Mode | What it shows |
| --- | --- | --- |
| `browsergym_harness.py` | white-box | Harness-oriented BrowserGym rollout for TRL |
| `browsergym_harness_eval.py` | white-box | White-box BrowserGym evaluation via the harness runtime |
| `browsergym_codex_eval.py` | black-box | Evaluate Codex CLI as an opaque agent over MCP |
| `copilot_sdk_harness_eval.py` | black-box | Evaluate the GitHub Copilot SDK as an opaque agent |
| `browsergym_harness_eval_common.py` | — | Shared helpers for the BrowserGym harness examples |

## Dataset collection

| Script | What it shows |
| --- | --- |
| `ttt_collect_demo.py` | Smoke-test the harness `collect` pipeline (scripted teacher) |
| `ttt_collect_with_llm.py` | Collect Tic-Tac-Toe rollouts with a real LLM teacher |

## Notebooks

| Notebook | What it covers |
| --- | --- |
| `openenv_introduction_quickstart.ipynb` | Start here: the core concepts and first rollout |
| `OpenEnv_Tutorial.ipynb` | Guided tutorial |
| `openenv_using_environments.ipynb` | Consuming existing environments |
| `openenv_building_environments.ipynb` | Building a new environment |
| `mcp_environment.ipynb` | MCP environments end to end |
| `rubrics.ipynb` | Rubric-based grading |
| `evaluation_inspect.ipynb` | Inspecting evaluation results |
| `sft_warmup.ipynb` | SFT warm-up before RL |
| `end_to_end_walkthrough.ipynb` | A full end-to-end walkthrough |

## Contributing an example

Keep examples self-contained and runnable. Put a module docstring at the top
that states what it shows, any manual prerequisites (Docker image, HF Space, API
keys, extra `pip install`s), and the exact run command. Make optional
dependencies import-guarded so the file still imports without them. Format with
`uv run ruff format examples/<your_example>.py`.
