#!/bin/bash

# 检查是否提供了关键词作为参数
if [ $# -eq 0 ]; then
    echo "请提供要查找的关键词作为参数。"
    exit 1
fi

# 获取关键词
keyword=$1

# 查找包含关键词的进程并获取其 PID
pids=$(ps -ef | grep "$keyword" | grep -v grep | awk '{print $2}')

# 检查是否找到匹配的进程
if [ -z "$pids" ]; then
    echo "未找到包含关键词 '$keyword' 的进程。"
else
    # 遍历 PID 列表并使用 kill -9 终止进程
    for pid in $pids; do
        echo "正在终止进程 $pid..."
        kill -9 $pid
        if [ $? -eq 0 ]; then
            echo "进程 $pid 已成功终止。"
        else
            echo "终止进程 $pid 时出错。"
        fi
    done
fi