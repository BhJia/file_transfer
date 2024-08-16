#!/bin/bash

# 第一个程序的进程ID
user="jbh"
process_name="python"

# 使用用户名和进程名称获取PID
pid=$(pgrep -u $user -f $process_name)

# 等待16小时
echo "将在16小时后结束进程 $pid 并启动新程序..."
sleep 1s

# 结束第一个程序
echo "正在结束进程 $pid..."
kill -9 $pid

# 确认进程已经被终止
while kill -0 $pid 2>/dev/null; do
    echo "等待进程 $pid 结束..."
    sleep 1s  # 每分钟检查一次
done

echo "进程 $pid 已结束，现在启动第二个程序..."
./2.sh