

可用的环境变量

- NET_SOURCE_CHECK
    创建PlaceDB实例时是否校验每个net只有一个输入引脚，通过设置`NET_SOURCE_CHECK=0`关闭校验
    具体逻辑见`lefdef_placedb.py:net_info`

- PLACEENV_IGNORE_PORT
    PlaceEnv在reset时，是否导入芯片的引脚位置，如果导入芯片引脚位置，后续计算HPWL时，会计算相应的线长。通过设置`PLACEENV_IGNORE_PORT=1`关闭
    具体见`place_env.py:reset`

分布训练常用命令

1. 启动dist_reverb
    `python dist_reverb.py > logs/reverb.log 2>&1 &`
2. 启动dist_collect
    `python launch_collect.py --jobs 8 --gpu 4 --reverb_ip localhost --noport`
3. 启动dist_train
    `python launch_train.py --cuda 0 --init_chpt xx/yy/zz --train_place > logs/train.log 2>&1 &`
4. 启动dist_eval
    `python dist_eval.py --cuda 3 --result_dir result/ariane133_04251457_pnm_176_seed_1745564233 --clean  > logs/dist_eval2.log 2>&1 &`