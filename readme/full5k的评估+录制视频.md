新模型非 RTC 录像：
cd /home/lixiang/PROJECT/LingBotVLA2.0

bash evaluation/lingbot_action_step2000/run_video_eval.sh \
  demo_clean \
  --model full5k \
  --precision bf16 \
  --task click_alarmclock \
  --episodes 1
新模型 RTC 录像：
cd /home/lixiang/PROJECT/LingBotVLA2.0

bash myscripts/simulation_rtc_tools/scripts/run_rtc_video_eval.sh \
  --setting demo_clean \
  --model full5k \
  --precision bf16 \
  --task click_alarmclock \
  --episodes 1 \
  --request-index 20 \
  --fixed-delay-steps 10 \
  --execution-horizon 20 \
  --crossfade-steps 10 \
  --ema-alpha 0.21


rtc评估：
cd /home/lixiang/PROJECT/LingBotVLA2.0

unset ROBOTWIN_REALTIME_VISUALIZATION

bash myscripts/simulation_rtc_tools/scripts/run_rtc_fast_50x50.sh \
  --model full5k \
  --precision bf16 \
  --episodes 50 \
  --request-index 20 \
  --fixed-delay-steps 10 \
  --execution-horizon 20 \
  --crossfade-steps 10 \
  --ema-alpha 0.21


查看成功率：
export RUN="/home/lixiang/PROJECT/LingBotVLA2.0/evaluation/lingbot_action_step2000/results/full5k/bf16/clean/lingbot_full_5k_6_60_5k_demo_clean_20261012_235815"

find "$RUN/eval_logs" -maxdepth 1 -type f -name '*.log' -print0 |
while IFS= read -r -d '' f; do
  result=$(
    sed -r 's/\x1B\[[0-9;]*[mK]//g' "$f" |
    tr '\r' '\n' |
    awk '/Success rate:/{last=$0} END{print last}'
  )

  printf "%-32s %s\n" \
    "$(basename "$f" .log)" \
    "${result:-进行中/尚无完整 episode}"
done | sort