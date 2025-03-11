import copy
import torch
from torch import nn

import networks
import tools

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    '''
    dreamerv3 世界模型
    '''
    def __init__(self, obs_space, act_space, step, config):
        '''
        param obs_space: dict, 观测空间
        param act_space: dict, 动作空间
        param step: int, 获取预热缓冲区的实际执行的步数
        param config: dict, 配置
        '''

        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        '''
        "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
        => "image": (3, 64, 64)
        '''
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        # 构建特征编码层
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        # 获取特征编码层输出的维度，作为特征编码后的向量维度
        self.embed_size = self.encoder.outdim
        # 构建RSSM
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act, 
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        # todo 结合实际的数据流向，搞清楚输入的时什么
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        # 解码层，解码层的输入是RSSM的输出
        '''
        "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
        => "image": (64, 64， 3)
        观察空间解码
        '''
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        # 奖励解码
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        # todo 表示环境是否继续的标识，在 DreamerV3 算法中，cont 是一个用于表示环境中是否继续的标志（continuation flag）。它通常用于处理非终止状态（non-terminal states）和终止状态（terminal states），并在训练过程中用于计算折扣因子（discount factor）
        '''
        详细解释
        cont 的作用
        表示非终止状态：

        cont 是一个二进制标志，用于表示当前状态是否为非终止状态。
        如果 cont 为 1，表示当前状态是非终止状态。
        如果 cont 为 0，表示当前状态是终止状态。
        计算折扣因子：

        在强化学习中，折扣因子用于计算未来奖励的现值。cont 用于调整折扣因子，以便在终止状态时停止累积奖励。
        例如，如果 cont 为 0，则折扣因子将被设置为 0，从而停止累积未来奖励。
        训练过程中的使用：

        在训练过程中，cont 被用于计算损失函数和目标值。它确保模型在终止状态时不会继续累积奖励，从而提高训练的稳定性和效果
        '''
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name

        # 这个优化器应该是优化在这个世界模型类里面定义的所有子模型参数
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        # todo 这个是缩放到时候的对应的损失吗
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)
        # 处理完成后，data还是一个字典类型，key时action\image\discount\cont\is_first\is_terminal这些

        with tools.RequiresGrad(self):
            with torch.amp.autocast('cuda', self._use_amp):
                # embed shape = (batch_size, batch_length, embed_size)
                embed = self.encoder(data)
                # 获取先验和后验概率
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                # 计算先验和后验的KL散度
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                # 遍历每一个预测头
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    # 把后验状态中的随机状态和确定性状态拼接起来
                    # 后验状态中的deter就是先验的deter
                    # 而stoch则是结合实际状态编码和先验deter预测得到的动作分布
                    feat = self.dynamics.get_feat(post)
                    # todo grad_head是什么，从配置可知：'decoder', 'reward', 'cont'都是True
                    feat = feat if grad_head else feat.detach()
                    # 根据不同的预测头，获取不同的预测结果，比如图像、奖励、是否继续等
                    # 根据后验特征获取每一步预测的结果
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        # 后验状态，一些上下文信息，一些度量信息
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        '''
        对像素数据进行预处理，归一化为0-1之间的数值
        '''
        # 1. 转换为tensor
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        # 2. 归一化
        obs["image"] = obs["image"] / 255.0
        if "discount" in obs:
            '''
            Collecting workspace information在 DreamerV3 的代码中，对于 Atari 游戏环境，discount 值是在 atari.py 中收集的。在世界模型的 `preprocess` 函数中可以看到对 discount 的处理：

            ````python
            if "discount" in obs:
                # 如果环境有折扣那么就乘以折扣
                obs["discount"] *= self._config.discount
                # (batch_size, batch_length) -> (batch_size, batch_length, 1)
                obs["discount"] = obs["discount"].unsqueeze(-1)
            ````

            折扣因子有两个作用:

            1. 作为奖励的折扣系数，用于计算长期回报
            2. 用于处理游戏终止状态，当环境结束时 discount=0，否则为 1

            在 Atari 环境中:
            - 非终止状态: discount = 1 
            - 终止状态 (游戏结束): discount = 0

            这样可以帮助模型区分游戏的终止和非终止状态，并正确计算长期回报。
            '''
            # 如果环境有折扣那么就乘以折扣
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        # 在收集观察时，is_first和is_terminal是必要的
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        # 得到cont，congt时是否结束的标志的反转，即是否继续，cont表示continuation flag
        obs["cont"] = (1.0 - obs["is_terminal"]).unsqueeze(-1)
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        '''
        这个类就就是在预测动作和值
        todo 搞清楚传入的数据有哪些

          actor:
            {layers: 2, dist: 'normal', entropy: 3e-4, unimix_ratio: 0.01, std: 'learned', min_std: 0.1, max_std: 1.0, temp: 0.1, lr: 3e-5, eps: 1e-5, grad_clip: 100.0, outscale: 1.0}
        critic:
            {layers: 2, dist: 'symlog_disc', slow_target: True, slow_target_update: 1, slow_target_fraction: 0.02, lr: 3e-5, eps: 1e-5, grad_clip: 100.0, outscale: 0.0}
        '''
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        # todo 到时候写代码时，要把这个MLP层分开使用
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        # 价值预测
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        # 价值网络有一个TargetNet
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        # 动作优化器
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        # 价值优化器
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # todo 这部分是在做什么
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):
        '''
        start: 后验状态
        objective: 奖励预测函数
        '''
        # 每次训练时，更新一次价值目标网络
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # self._config.imag_horizon参数的作用
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                # 将预测的动作和各种状态传入奖励预测函数
                reward = objective(imag_feat, imag_state, imag_action)
                # 传入相同的确定性状态和随机状态的结合预测动作的熵
                actor_ent = self.actor(imag_feat).entropy()
                # todo 这里好像并没有将state_ent纳入计算，这里也是获取动作的分布
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        '''
        动作想象
        param start: 后验状态
        param policy: 动作策略
        param horizon: todo 这个参数是什么，好像传入的是一个常量
        '''
        # rssm模型
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        # 展品后验状态的前两个维度，也就是batch_size和batch_length
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            '''
            prev：后验状态 (start, None, None)
            '''
            state, _, _ = prev
            # 提取特征（结合随机性状态和确定性状态）
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            # 预测执行的动作
            action = policy(inp).sample()
            # 在这里作为上一个状态和上一个动作，预测到先验状态
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        # 遍历每一个时间步，获取每一个时间步的特征、状态和动作
        # 得到预测的先验状态、特征（结合随机性状态和确定性状态）和动作
        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        # 这边好像是将后验状态和预测的先验状态拼接起来
        # todo 这里拼接的形状
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        # 得到预测的先验状态、特征（结合随机性状态和确定性状态），预测的先验状态和传入的后验状态拼接，预测的动作
        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            # 又是获取随机状态和确定性状态的组合
            inp = self._world_model.dynamics.get_feat(imag_state)
            # 预测折扣
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            # 如果不包含则直接使用配置的折扣
            discount = self._config.discount * torch.ones_like(reward)
        # 预测价值
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        # 计算折扣因子
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
