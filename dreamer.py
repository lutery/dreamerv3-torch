import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
from ruamel.yaml import YAML

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        '''
        param obs_space: 观察空间
        param act_space: 动作空间
        param config: 配置文件
        param logger: 日志
        param dataset: 数据集，字典类型，包含多个 episode 的数据
        '''

        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length # 这里就是数据的总长度
        self._should_train = tools.Every(batch_steps / config.train_ratio) # 按照默认的参数来算，batch_steps / config.train_ratio = 2，这里是在计算模型应多久执行一次训练更新
        # 也就是保证模型会充分利用已有的数据进行训练，稳定后再收集新的环境数据进行训练
        self._should_pretrain = tools.Once() # # 如果是继续训练，这里会将shoule_pretrain设置为False
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat # 得到实际执行的步数，因为会有重复执行步数的操作
        self._update_count = 0
        self._dataset = dataset
        # 完成构建世界模型
        self._wm = models.WorldModel(obs_space, act_space, self._step, config) # 构建世界模型
        # 构建动作 价值预测网络
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            # 预编译模型，提高运行效率，对dreamerv3不是必备的
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        # 感觉这里在构建探索模型
        # config.expl_behavior = greedy
        # [config.expl_behavior]().to(self._config.device)迁移到指定的设备运行
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space), # 随机探索，train是空的
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        ''''
        param obs: 环境观察
        param reset: 是否重置,是否结束
        '''
        step = self._step
        if training:
            # 训练模式
            # 如果是初始训练，那么steps=self._config.pretrain,如果是中断继续训练，那么steps=self._should_train(step)
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            # 训练模型
            for _ in range(steps):
                # todo self._dataset中的数据是在什么时候填充的
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            # 这边应该是记录日志，不太重要 todo 后续增加上记录视频
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        # 这边应该是正常的根据状态预测动作
        policy_output, state = self._policy(obs, state, training)

        if training:
            # 计算执行的步数
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        '''
        训练模型
        param data: 字典类型，key是states,actions,rewards,next_states,dones， value时对应的数据,batch size, batch length, obs/act space/reward
        '''
        metrics = {}
        # 完成世界模型的训练
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        # 奖励预测
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        # 训练动作、价值预测网络
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            # 训练探索模型
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


# 函数在代码中的作用是计算已经完成的训练步骤数。它通常会遍历指定目录中的训练数据文件，并根据文件的数量或内容来计算已经完成的训练步骤数
def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    '''
    param episodes: episodes 加载的环境连续过程的数据
    param config: 配置文件
    '''
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    '''
    param config: 配置文件
    param mode: 模式,train or eval
    param id: id,环境id

    todo 仅完成atari环境的注释，其他环境的注释待补充
    '''
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    # 已完成的步数
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        '''
        在 DreamerV3 算法中，离线训练数据是指预先收集并存储在磁盘上的环境交互数据。这些数据可以用于训练模型，而不需要在训练过程中实时与环境进行交互。离线训练数据通常包含状态、动作、奖励和下一状态等信息。

        生成离线训练数据的步骤通常如下：

        与环境交互：使用一个策略（可以是随机策略、预训练策略或专家策略）与环境进行交互，收集状态、动作、奖励和下一状态等信息。
        存储数据：将收集到的数据存储在磁盘上，通常以文件的形式保存，每个文件包含一个或多个回合的数据。
        加载数据：在训练过程中，从磁盘加载这些离线数据，并用于训练模型
        '''
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    # 这个加载的是数据集？todo
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    # 方法用于从指定目录加载训练或评估数据集。它会读取目录中的 .npz 文件，并将每个文件的内容加载到一个有序字典中。每个文件代表一个 episode，包含状态、动作、奖励等信息
    # todo 这个数据集的结构是什么样子的
    '''
    数据集是一个有序字典（OrderedDict），其中每个键是一个 episode 的文件名（去掉扩展名），每个值是一个包含该 episode 数据的字典。每个 episode 数据的字典包含多个键值对，通常包括状态、动作、奖励和下一状态等信息。

    假设每个 .npz 文件包含以下数据结构：

    states: 一个数组，表示每个时间步的状态。
    actions: 一个数组，表示每个时间步的动作。
    rewards: 一个数组，表示每个时间步的奖励。
    next_states: 一个数组，表示每个时间步的下一状态。
    dones: 一个数组，表示每个时间步是否结束
    '''
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    # 是否开启多进程，todo 这次的dreamerv3的代码引入这个功能，提高训练速度
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    # 动作空间的维度
    acts = train_envs[0].action_space
    print("Action Space", acts)
    # 获取动作数量n
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        # 如果没有离线训练数据，则需要预填充数据集
        # 获取缓冲区还要填充的步数，如果缓冲区已经有数据，则减去已经有的步数
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            # 如果动作空间是离散的，则使用随机分布生成随机动作
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            # 如果动作空间是连续的，则使用均匀分布生成随机动作
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            # return: 采样的动作和动作的对数概率
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    # 这里开始模拟数据转换为训练数据和评估数据
    train_dataset = make_dataset(train_eps, config)
    # 一开始评估数据集是空的，后续逐步填充
    eval_dataset = make_dataset(eval_eps, config)
    '''
    atari下
    observation_space: 
    gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
            }

    action_space:
    space = self._env.action_space
        space.discrete = True
        return space
    '''
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    # 这里是将所有的模型梯度都关闭吗 todo
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        # 模型的网络加载权重
        agent.load_state_dict(checkpoint["agent_state_dict"])
        # 模型的优化器加载保存
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        # 如果是继续训练，这里会将shoule_pretrain设置为False
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        # 代理器能够训练的步数
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            # 评估，每次评估都会将运行的预测游戏画面保存到日志中
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            # todo 实现记录视频日志
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        # 训练
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default='atari100k')
    '''
    args 包含解析后的命令行参数，通常是一个命名空间对象，其中属性名对应参数名，属性值对应参数值。
    remaining 包含未被解析的命令行参数列表。
    这里只是在解析参数名字
    '''
    args, remaining = parser.parse_known_args()
    yaml = YAML(typ='rt')
    configs = yaml.load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    # 加载默认配置文件和指定的配置文件，并将它们合并到一个字典中。
    name_list = ["defaults", args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    # 将默认配置中的每个键值对添加为命令行参数，并将其值转换为相应的类型。
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    # 解析命令行参数 todo：为什么要解析未被解析的命令行参数列表remaining
    main(parser.parse_args(remaining))
