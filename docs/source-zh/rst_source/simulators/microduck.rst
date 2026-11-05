Microduck
=========

在独立的 Docker 仿真容器中执行有界速度指令，并保存视频。这是开环控制接口，不是经过验证的导航基准。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      镜像内置行走控制器；主机不运行 VLA。

   .. grid-item-card:: 规划器

      支持工具调用的 API 规划器。

   .. grid-item-card:: 任务

      按顺序执行有界速度指令。

   .. grid-item-card:: 硬件

      本地 Linux Docker Engine；通过 OSMesa 在 CPU 上渲染。

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 类别
     - 范围
   * - 开环运动
     - 请求前进、后退、横移或转向速度；不提供任务成功传感信号。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 接口
     - 行为
   * - 观测
     - 返回空字典；规划器不接收图像或位姿反馈。
   * - 动作
     - ``robot_move(vx=0.4, vy=0.0, vyaw=0.0, duration_s=3.0)``。
   * - 奖励与成功判定
     - 不进行原生任务成功评估；``finish`` 仅表示规划器结束执行。
   * - 任务指令
     - 必须通过 ``--task`` 提供非空的自然语言指令。

安装与资源准备
--------------

使用已安装 RPent 及其 ``microduck`` extra 的当前主机 venv。缺少依赖时，须先取得操作者许可再安装。以下命令中的 ``python`` 均指该 venv 的解释器，不是系统 Python，也不创建新环境：

.. code-block:: bash

   python -c 'import sys; assert sys.prefix != sys.base_prefix, "Use the current host venv"; print(sys.executable)'

Python 子进程原样使用 ``sys.executable``。容器内的 Python 与主机分离；仿真器和固定版本依赖只安装在镜像中。Docker 必须通过本地 Linux Unix socket 无需 sudo 即可访问，不支持远程引擎或远程 context。使用 Docker 原生架构，不检查 CPU 指令集，也不强制指定平台。主机 UID:GID 须对输出绑定目录具有写权限。无需 GPU、特权模式、主机网络或挂载 Docker socket。

第一步：在仓库根目录单独构建镜像。运行时不会构建、拉取镜像、下载或升级依赖：

.. code-block:: bash

   docker build \
     --build-arg http_proxy --build-arg https_proxy --build-arg all_proxy --build-arg no_proxy \
     --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg ALL_PROXY --build-arg NO_PROXY \
     -f robots/microduck/docker/Dockerfile -t rpent-microduck:local .

Docker 按名称读取已导出的代理变量，保留大小写、原值和绕过列表。不要将值展开到命令参数中，不要启用 shell 跟踪或导出环境变量清单。不要在 Dockerfile 中声明代理 ARG/ENV 或显式引用代理变量，否则可能持久化敏感值，并破坏预定义代理的历史记录和缓存保护。构建步骤的代理参数不负责访问镜像仓库或拉取基础镜像；操作者须单独配置 Docker daemon/BuildKit。主机回环代理地址在构建容器内并不指向同一端点；遇到连通性问题应报告，不要擅自改写地址或改用主机网络。原始日志应私下保存，对外分享前先清除 URL 和凭据。监督进程不显式转发代理或规划器凭据，但操作者的 Docker 客户端默认配置仍可能注入环境变量；它不会清理任意 Docker 配置。

在禁用网络的条件下验证已构建的镜像：

.. code-block:: bash

   docker image inspect rpent-microduck:local --format '{{.Id}} {{index .Config.Labels "org.rpent.microduck.env-protocol"}}'
   docker run --rm --network none --entrypoint python rpent-microduck:local /opt/rpent/robots/microduck/docker/verify_image.py

记录不可变镜像 ID 和协议标签 ``1``。离线镜像验证不能证明步态质量。应私下检查镜像配置和历史记录中是否意外保存代理信息。容器源码或版本固定信息变化后，需要显式重新构建。源码与策略的版本来源、保留的许可声明见 ``robots/microduck/README.md``。代码和模型的许可须分别审核；此流程不代表商业使用授权。

运行一个任务
------------

第二步：配置 ``OPENAI_API_KEY``；使用自定义 OpenAI 兼容端点时，还需配置 ``OPENAI_BASE_URL``。端点必须提供指定的 ``openai:deepseek-flash`` 模型，不要替换为其他模型。使用全新的输出目录：

.. code-block:: bash

   python -m rpent.cli.main --planner api --model openai:deepseek-flash --robot microduck \
     --task "move forward 5 seconds, and then move backward" --output-dir ./logs/xxxx

只有入口脚本属于当前 venv 时，才能使用 ``rpent`` 命令。``--microduck-image`` 可覆盖默认的 ``rpent-microduck:local``；镜像必须已存在于本地，协议标签为 ``1``。省略输出目录时使用 ``logs/<timestamp>_microduck``。每次运行自动选择全新的空本地记忆。不支持 Dashboard、Flash 或探索模式。

预期工具调用顺序为：显式指定前进五秒；以 ``vx=-0.4`` 后退并省略时长，使用默认三秒；最后调用 ``finish``。默认前进速度意图为 ``vx=0.4`` m/s，``vy=vyaw=0``。所有数值必须有限且不能是布尔值：``|vx|<=0.4`` m/s、``|vy|<=0.10`` m/s、``|vyaw|<=0.5`` rad/s，时长满足 ``0<duration_s<=30`` 秒。指令串行执行；部分执行后失败的动作不会自动重试。空观测无法证明实际位移、避障、平衡或任务成功。不支持直接关节控制或实体机器人模式。

较小的请求速度仍然有效，但可能只产生微小位移。速度请求表达的是开环控制意图，不是实测速度，也不保证精确运动。模型可能选择不同的调用，也可能无法完成指令序列；应检查执行记录和视频，不要假定模型一定能正确完成任务。

查看结果
--------

查看 ``transcript_microduck.json`` 中的工具调用顺序、空观测和错误，以及 ``states.json`` 中的根目录产物登记。正常关闭时，先完成录像并发布 ``microduck.mp4``，再停止容器。视频采用固定视角参数、跟踪躯干的相机，格式为 H.264/yuv420p、640×480、25 fps，涵盖 SIT、起身、等待、运动指令和最终稳定阶段。录制上限为 300 仿真秒，队列容量为 250；离线渲染期限为 300 秒。应人工检查视频；规划器调用 ``finish`` 不代表任务已通过验证。

诊断与检查
----------

``.microduck/`` 保留暂存视频、轨迹、记录镜像和运行身份的 ``runtime.json``、``supervisor.log``、``container.log``、控制器和渲染日志，以及可用的失败记录。结束阶段的错误写入 ``finalization-error.json``，不会悄悄转为成功。缺少镜像时，错误信息给出上面同一条构建命令；测试不得自行构建或拉取镜像。模型端点错误须与仿真器、控制器或渲染错误分别判断。

每次运行独占一个带 UUID 名称和标签的容器，以主机 UID:GID 运行，并发布临时回环端口。正常关闭只删除本次容器。父进程死亡后，只要监督进程仍存活，就会触发协作清理；强制关闭不能保证保存 MP4。Docker 不可用，或创建、清理超时时，不能确认容器已移除：应遵照失败记录中的精确名称和标签检查指引操作，不要批量清理或删除无关容器。主机或监督进程同时死亡时，需要人工检查。

离线检查与显式启用的 CLI 检查见 ``tests/README.md``。确定性规划器测试验证的是集成链路，不是模型质量或步态成功。测试要求手动准备镜像并安装 ``pytest-timeout``，每个 CLI 用例限时 600 秒。未启用时跳过；启用后缺少前提会明确失败。
