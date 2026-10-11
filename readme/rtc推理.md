一.评估：
cd /home/lixiang/PROJECT/LingBotVLA2.0

unset ROBOTWIN_REALTIME_VISUALIZATION

bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh

结果目录保存在：
myscripts/simulation_rtc_tools/results/clean/

查看成功率：
RUN=/home/lixiang/PROJECT/LingBotVLA2.0/myscripts/simulation_rtc_tools/results/clean/_0k_demo_clean_20261009_194445

sed -r 's/\x1B\[[0-9;]*[mK]//g' "$RUN/eval_logs/lift_pot.log" \
  | tr '\r' '\n' \
  | grep 'Success rate:' \
  | tail -1


根据上述结果目录断点续跑评估：
cd /home/lixiang/PROJECT/LingBotVLA2.0

unset ROBOTWIN_REALTIME_VISUALIZATION

export QWEN3VL_PATH="$PWD/models/Qwen3-VL-4B-Instruct"
export LINGBOT_RTC_TOOL_ROOT="$PWD/myscripts/simulation_rtc_tools"
export LINGBOT_RTC_PROJECT_ROOT="$PWD"
export LINGBOT_RTC_CONFIG="$PWD/myscripts/simulation_rtc_tools/configs/lingbot_action_step2000_rtc_fast.yaml"

bash "$PWD/myscripts/simulation_rtc_tools/scripts/start_robotwin_rtc_infer_and_eval.sh" \
  --model_path "$PWD/models/lingbot_action_step2000" \
  --inference_script myscripts/simulation_rtc_tools/server/policy_server_rtc.py \
  --inference_workdir "$PWD/" \
  --eval_workdir "$PWD/RoboTwin_lingbot" \
  --output_base "$PWD/myscripts/simulation_rtc_tools/results/clean" \
  --resume_run_dir "$PWD/myscripts/simulation_rtc_tools/results/clean/_0k_demo_clean_20261009_194445" \
  --conda_sh /home/lixiang/miniconda3/etc/profile.d/conda.sh \
  --inference_env lingbotvla \
  --sim_env RoboTwin \
  --task_config demo_clean \
  --num_tasks 50 \
  --num_episodes 50 \
  --num_gpus 1 \
  --num_per_gpu 1 \
  --use_length 50 \
  --use_bf16 True \
  --use_fp32 False \
  --use_compile True \
  --no_video

二.可视化：
（1）现实时间rtc：
cd /home/lixiang/PROJECT/LingBotVLA2.0

export QWEN3VL_PATH="$PWD/models/Qwen3-VL-4B-Instruct"

export LINGBOT_RTC_TOOL_ROOT="$PWD/myscripts/simulation_rtc_tools"
export LINGBOT_RTC_PROJECT_ROOT="$PWD"
export LINGBOT_RTC_CONFIG="$PWD/myscripts/simulation_rtc_tools/configs/lingbot_action_step2000_rtc_fast.yaml"

export LINGBOT_RTC_REQUEST_INDEX=20
export LINGBOT_RTC_FIXED_DELAY_STEPS=20
export LINGBOT_RTC_EXECUTION_HORIZON=20
export LINGBOT_RTC_CROSSFADE_STEPS=4
export LINGBOT_RTC_EMA_ALPHA=0.36

export ROBOTWIN_REALTIME_VISUALIZATION=1
export PYTHONPATH="$PWD/evaluation/lingbot_action_step2000/realtime_live:$PWD${PYTHONPATH:+:$PYTHONPATH}"

bash myscripts/simulation_rtc_tools/scripts/start_robotwin_rtc_infer_and_eval.sh \
  --model_path "$PWD/models/lingbot_action_step2000" \
  --inference_script myscripts/simulation_rtc_tools/server/policy_server_rtc.py \
  --inference_workdir "$PWD/" \
  --eval_workdir "$PWD/RoboTwin_lingbot" \
  --output_base "$PWD/myscripts/simulation_rtc_tools/results/realtime_visualization" \
  --conda_sh /home/lixiang/miniconda3/etc/profile.d/conda.sh \
  --inference_env lingbotvla \
  --sim_env RoboTwin \
  --task_config demo_clean \
  --task_name click_alarmclock \
  --num_tasks 1 \
  --num_episodes 1 \
  --num_gpus 1 \
  --num_per_gpu 1 \
  --use_length 50 \
  --use_bf16 True \
  --use_fp32 False \
  --use_compile True \
  --render_freq 1 \
  --no_video

（2）仿真时间rtc：
cd /home/lixiang/PROJECT/LingBotVLA2.0

unset ROBOTWIN_REALTIME_VISUALIZATION

export QWEN3VL_PATH="$PWD/models/Qwen3-VL-4B-Instruct"
export LINGBOT_RTC_TOOL_ROOT="$PWD/myscripts/simulation_rtc_tools"
export LINGBOT_RTC_PROJECT_ROOT="$PWD"
export LINGBOT_RTC_CONFIG="$PWD/myscripts/simulation_rtc_tools/configs/lingbot_action_step2000_rtc_fast.yaml"

export LINGBOT_RTC_REQUEST_INDEX=20
export LINGBOT_RTC_FIXED_DELAY_STEPS=20
export LINGBOT_RTC_EXECUTION_HORIZON=20
export LINGBOT_RTC_CROSSFADE_STEPS=4
export LINGBOT_RTC_EMA_ALPHA=0.36

bash myscripts/simulation_rtc_tools/scripts/start_robotwin_rtc_infer_and_eval.sh \
  --model_path "$PWD/models/lingbot_action_step2000" \
  --inference_script myscripts/simulation_rtc_tools/server/policy_server_rtc.py \
  --inference_workdir "$PWD/" \
  --eval_workdir "$PWD/RoboTwin_lingbot" \
  --output_base "$PWD/myscripts/simulation_rtc_tools/results/fast_visualization" \
  --conda_sh /home/lixiang/miniconda3/etc/profile.d/conda.sh \
  --inference_env lingbotvla \
  --sim_env RoboTwin \
  --task_config demo_clean \
  --task_name click_alarmclock \
  --num_tasks 1 \
  --num_episodes 1 \
  --num_gpus 1 \
  --num_per_gpu 1 \
  --use_length 50 \
  --use_bf16 True \
  --use_fp32 False \
  --use_compile True \
  --render_freq 1 \
  --no_video

（3）可用任务：
01. adjust_bottle
02. beat_block_hammer
03. blocks_ranking_rgb
04. blocks_ranking_size
05. click_alarmclock
06. click_bell
07. dump_bin_bigbin
08. grab_roller
09. handover_block
10. handover_mic

11. hanging_mug
12. lift_pot
13. move_can_pot
14. move_pillbottle_pad
15. move_playingcard_away
16. move_stapler_pad
17. open_laptop
18. open_microwave
19. pick_diverse_bottles
20. pick_dual_bottles

21. place_a2b_left
22. place_a2b_right
23. place_bread_basket
24. place_bread_skillet
25. place_burger_fries
26. place_can_basket
27. place_cans_plasticbox
28. place_container_plate
29. place_dual_shoes
30. place_empty_cup

31. place_fan
32. place_mouse_pad
33. place_object_basket
34. place_object_scale
35. place_object_stand
36. place_phone_stand
37. place_shoe
38. press_stapler
39. put_bottles_dustbin
40. put_object_cabinet

41. rotate_qrcode
42. scan_object
43. shake_bottle
44. shake_bottle_horizontally
45. stack_blocks_three
46. stack_blocks_two
47. stack_bowls_three
48. stack_bowls_two
49. stamp_seal
50. turn_switch