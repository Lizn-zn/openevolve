# openrouter api key
# export OPENROUTER_API_KEY="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"
export OPENROUTER_API_KEY="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"
# export OPENAI_API_BASE="https://openrouter.ai/api/v1"

# Start with nohup and setsid to prevent SIGHUP from killing the process
# Use setsid to create a new session, making the process independent of the terminal
# This ensures the process survives even if the parent shell exits
python openevolve-run.py problems/verina_advanced_1/initial_program.lean \
  problems/verina_advanced_1/evaluator.py \
  --target-score 1.0 \
  --config problems/verina_advanced_1/config_stage_1.yaml 
  # --checkpoint problems/verina_advanced_1/openevolve_output/checkpoints/checkpoint_500