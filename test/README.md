### 与飞控通信的注意事项

- 程序默认使用 `/dev/ttyS1`、230400 baud、8N1；板端必须确认 UART 设备、波特率、A28/A29 交叉接线和共地与飞控配置一致。一个串口只能由一个进程独占，先停止 `landing`、心跳监听器等占用同一 UART 的程序。
- 收到 PX4 心跳后，程序通过该 UART 发送 `MAV_CMD_SET_MESSAGE_INTERVAL`（command 511），请求 `OPTICAL_FLOW_RAD`（message ID 106）以 20 Hz 输出，并等待 `COMMAND_ACK`。只有收到接受确认后才继续预检；飞控端还需允许该命令、该串口 MAVLink 实例需正常收发。
- 光流消息请求是按当前 MAVLink 链路发出的，不等于永久修改飞控参数；重启程序后会再次请求。若请求被拒绝或超时，程序会停止，不会切模式或解锁。可在 PX4 NSH 用 `mavlink status` 确认对应 serial instance，并用 `listener vehicle_optical_flow` 检查飞控内部光流；本程序实际订阅的是 MAVLink `OPTICAL_FLOW_RAD.distance`，不是直接读取 UORB。
- 进入动作前要求心跳与有效光流连续 5 秒。运行中若心跳、光流、Offboard 或解锁状态失效，程序停止发送非零运动设定并持续发零速度；零速度并不等同于位置/高度锁定，遥控器必须随时可接管。
- 可在 NSH 通过 `listener trajectory_setpoint 5` 核对垂直速度和 yaw 速度设定。MAVLink `SET_POSITION_TARGET_LOCAL_NED` 的 `yaw_rate` 单位为 rad/s；程序命令行则用度/秒。