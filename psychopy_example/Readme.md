# psychopy范式接入说明

## 范式生成

范式生成采用`psychopy`开源软件进行设计，其最后的产物主要包括主.py文件以及相应的材料内容；具体可以参考`psychopy_example`的内容；

## 范式数据

- 范式数据定义：指的是用户在范式任务中所作出的响应，主要包括键盘输入等；目前系统主要接受的是键盘输入；
- 范式数据格式：以`psychopy_example`为例，其格式为

```C
marker = f"MARKER {startOfTheParadigm}"
timestamp = local_clock()
outlet.push_sample([marker], timestamp)

// 最后格式如下，其中startOfTheParadigm为自定义常量，一般为字符串字面量
MARKER {startOfTheParadigm} timestamp
// timestamp非必须，如果不添加timestamp，LSL会自动加上timestamp
```

- 范式数据接入方式：由系统定义好对应的LSL流，范式通过连接特定的LSL流进行数据传输

```C
# -----------------------------
# 创建 StreamInfo
# -----------------------------
info = StreamInfo(
     name="eprime",   # 其与系统创建的LSL流一致
     type="Markers",
     channel_count=1,
     nominal_srate=0,             # irregular
     channel_format="string",
     source_id="test_marker_stream"
 )

# -----------------------------
# 创建 Outlet
# -----------------------------
outlet = StreamOutlet(info)
```



