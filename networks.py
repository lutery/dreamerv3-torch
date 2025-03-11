import math
import numpy as np
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools


class RSSM(nn.Module):
    '''
    dreamer特有的rssm模型
    '''

    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none", # 默认参数是none
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        super(RSSM, self).__init__()
        self._stoch = stoch # dreamer 随机特征编码的维度 todo
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth # 在 DreamerV3 的 RSSM (Recurrent State-Space Model) 中，rec_depth 参数用于控制 GRU（Gated Recurrent Unit）循环的深度。具体来说，它决定了在一个时间步内 GRU 单元重复处理特征的次数。
        self._discrete = discrete # 动作的表示方式，是连续的还是离散的
        act = getattr(torch.nn, act)
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions # 动作数量
        self._embed = embed
        self._device = device

        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        # todo 看实际输入时，具体时什么数据传入的
        # 看起来是将随机特征编码和动作拼接在一起进行特征菜样，应该是先验状态特征吧
        # 采集先验状态和动作，然后将其映射到隐藏层
        inp_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act())
        self._img_in_layers = nn.Sequential(*inp_layers)
        self._img_in_layers.apply(tools.weight_init)
        # 构建grucell的单元，传入的先验状态特征编码，输出确定性状态特征编码
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        img_out_layers = []
        inp_dim = self._deter
        # 构建确定性编码再次提取特征的层道隐藏层的映射
        img_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act())
        # 传入确定性状态特征编码，输出随机特征编码
        self._img_out_layers = nn.Sequential(*img_out_layers)
        self._img_out_layers.apply(tools.weight_init)

        obs_out_layers = []
        # todo 构建确定性编码和一个不知道是什么的编码拼接的层，提取特征
        inp_dim = self._deter + self._embed
        obs_out_layers.append(nn.Linear(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)
        self._obs_out_layers.apply(tools.weight_init)

        # self._discrete相当于dreamerv2中的rssm_type
        # todo 搞清楚这里是在输出什么数据
        if self._discrete:
            # 如果时离散的，那么就是logit
            # 这里离散的输出和dreamerv2不一样，不知道时不是dreamerv3的特点 todo
            self._imgs_stat_layer = nn.Linear(
                self._hidden, self._stoch * self._discrete
            )
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))
        else:
            # 如果是连续的，那么就是均值和方差
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(
                torch.zeros((1, self._deter), device=torch.device(self._device)),
                requires_grad=True,
            )

        # todo 注释上面这么多网络的数据流动

    def initial(self, batch_size):
        '''
        初始化先验状态特征编码，初始化的时候，这里就是batch_size，因为此时处于每一个时间步遍历的时候
        '''
        # 这里构建的是一个不可学习的确定性状态
        deter = torch.zeros(batch_size, self._deter, device=self._device)
        # 接下来分离散和连续环境两种情况
        if self._discrete:
            '''
            logit: 随机特征编码，shape = (batch_size, stoch, discrete_num)
            stoch: 随机特征编码，shape = (batch_size, stoch, discrete_num)
            deter: 确定性状态特征编码，shape = (batch_size, deter)
            '''
            state = dict(
                logit=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                stoch=torch.zeros(
                    [batch_size, self._stoch, self._discrete], device=self._device
                ),
                deter=deter,
            )
        else:
            '''
            mean: 均值，shape = (batch_size, stoch)
            std: 方差，shape = (batch_size, stoch)
            stoch: 随机特征编码，shape = (batch_size, stoch)
            deter: 确定性状态特征编码，shape = (batch_size, deter)
            '''
            state = dict(
                mean=torch.zeros([batch_size, self._stoch], device=self._device),
                std=torch.zeros([batch_size, self._stoch], device=self._device),
                stoch=torch.zeros([batch_size, self._stoch], device=self._device),
                deter=deter,
            )
        # 按照默认参数来看，这里应该时learned   
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            # 这里将确定性状态特征编码设置为一个可学习的参数
            # shape = (batch_size, deter)
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            # 这里根据可以学习的参数来计算随机特征编码,这个随机状态编码和预测的动作有很大的关系
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        '''
        param embed: 特征采样的输出
        param action: 动作
        param is_first: 是否是第一次，一般时reset后
        param state: 先验状态特征编码，在训练时是None
        '''
        # swap 交换维度，也就是batch time ch -> time batch ch
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            (action, embed, is_first),
            (state, state),
        )

        # (time, batch, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state, dtype=None):
        # 根据动作时连续还是离散来获取分布，所以我认为_discrete是一个标志位，标识动作时连续还是离散
        if self._discrete:
            logit = state["logit"]
            dist = torchd.independent.Independent(
                tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1
            )
        else:
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
            )
        return dist

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        '''
        param prev_state: 先验状态特征编码,第一个时None，可能包含 'deter'、'stoch'、'mean'、'std' 等状态
        param prev_action: 动作
        param embed: 环境观察特征采样的输出
        param is_first: 是否是第一次，一般时reset后或者第一次
        '''
        # initialize all prev_state
        if prev_state == None or torch.sum(is_first) == len(is_first):
            # 如果时第一次，那么就初始化
            # 初始化状态特征编码，包含logit、deter、stoch或者mean、std、stoch、deter
            prev_state = self.initial(len(is_first))
            # shape = (batch_size, num_actions)
            prev_action = torch.zeros(
                (len(is_first), self._num_actions), device=self._device
            )
        # overwrite the prev_state only where is_first=True
        elif torch.sum(is_first) > 0:
            # 如果存在reset的情况
            is_first = is_first[:, None]
            # 将prev_action设置为0，应该是重置了，之前的动作不再有意义
            prev_action *= 1.0 - is_first
            # 初始化状态特征编码
            init_state = self.initial(len(is_first))
            # 遍历前一个状态字典中的所有键值对
            # 可能包含 'deter'、'stoch'、'mean'、'std' 等状态
            for key, val in prev_state.items():
                # 将 is_first 张量的形状扩展以匹配状态值的形状
                # 通过添加适当数量的单位维度实现广播
                is_first_r = torch.reshape(
                    is_first,
                    is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
                )
                # val * (1.0 - is_first_r): 保持未重置环境的原有状态
                # init_state[key] * is_first_r: 对重置的环境使用初始状态，因为is_first_r为false时是0
                # 两者相加实现选择性更新
                prev_state[key] = (
                    val * (1.0 - is_first_r) + init_state[key] * is_first_r
                )

        # 前一个状态+前一个动作，获取先验状态
        # todo prev_state来自上一次的后验状态
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        # 结合先验状态特征编码和环境观察特征采样的输出，提取特征
        x = self._obs_out_layers(x)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        # 根据实际观察特征预测动作的分布
        stats = self._suff_stats_layer("obs", x)
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        # 获取后验状态
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        # 返回后验状态和先验状态，这里和之前一样，后验状态包含本次的环境状态，而先验没有，只有动作和上一次的先验状态
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        '''
        param prev_state: 先验状态特征编码
        param prev_action: 动作
        '''

        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = torch.cat([prev_stoch, prev_action], -1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = self._img_in_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        # 根据离散和连续环境获取不同的输出
        # 根据连续状态预测动作的分布
        # stats = {"mean": mean, "std": std}或者stats = {"logit": logit}
        stats = self._suff_stats_layer("ims", x)
        # 根据sample来决定是采样还是最大概率，默认是true，可能测试时是False
        if sample:
            stoch = self.get_dist(stats).sample()
        else:
            stoch = self.get_dist(stats).mode()
        # 先验状态特征编码
        # stoch：来自上一个状态和上一次的动作预测输出
        # deter：来自上一个状态的确定性特征编码经过GRU处理后的输出
        # stats：预测的动作分布
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def get_stoch(self, deter):
        # 采集确定性特征编码特征
        x = self._img_out_layers(deter)
        # imgs应该时想象吧 todo，如果是在训练遍历每一时间步时
        # 返回预测的动作分布
        stats = self._suff_stats_layer("ims", x)
        dist = self.get_dist(stats)
        # 方法是用于获取分布的最可能值（众数）
        # # 连续分布的众数就是均值
        #  # 离散分布的众数是最大概率的类别
        # 更多查看readme
        return dist.mode()

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            # 离散环境
            if name == "ims":
                # 输入的是img_out_layers层的输出
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                # todo 确定这里哪里输入的
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            # 获取离散环境下的动作预测
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            # 连续环境
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            #获取连续环境下的均值和方差
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            # 这里的动作查看reamdme
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: torch.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            # 返回预测的均值和方差
            return {"mean": mean, "std": std}

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


class MultiEncoder(nn.Module):
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
    ):
        
        '''
         atari:
        {mlp_keys: '$^', cnn_keys: 'image', act: 'SiLU', norm: True, cnn_depth: 32, kernel_size: 4, minres: 4, mlp_layers: 5, mlp_units: 1024, symlog_inputs: True}
        '''
        super(MultiEncoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        '''
        atari:
        "image": (3, 64, 64)
        '''
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        '''
        atari:
        k == image
       cnn_keys: 'image'
       cnn_shapes: {'image': (3, 64, 64)}
        '''
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }

        '''
        atari:
        mlp_shapes: {}
        '''
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            # 如果是图像空间则采用卷积
            # 从这里来看，从观察空间获取的顺序应该是（h, w, c）
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            # input_shape = (64, 64, 3)
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(
                input_shape, cnn_depth, act, norm, kernel_size, minres
            )
            # 得到特征编码层的输出维度大小
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            # cnn_shapes中只有images，所以这里时提取图像特征，并将所有图像特征拼接在一起，之前的stack还知识np.stack的，这里应该也是转换为了tensor
            # shape = (batch, time, h, w, ch)
            inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
            # 经过_cnn后，输出的时b t embed_size
            outputs.append(self._cnn(inputs))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        outputs = torch.cat(outputs, -1)
        return outputs


class MultiDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
        outscale,
    ):
        '''
        参数可参考config文件
        {mlp_keys: '$^', cnn_keys: 'image', act: 'SiLU', norm: True, cnn_depth: 32, kernel_size: 4, minres: 4, mlp_layers: 5, mlp_units: 1024, cnn_sigmoid: False, image_dist: mse, vector_dist: symlog_mse, outscale: 1.0}
        解码层，卷积就是上菜样，mlp就是上菜样到内存映射
        '''

        super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        # 这里应该是保证去除其他多余的键值
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        # "image": (64, 64， 3)
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        # 在atari中，这里是空的
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            # 这里应该是将 64 64 3 这个shape转换为 3 64 64 todo
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                act,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            return tools.ContDist(
                torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3)
            )
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        '''
        具体参数看config文件

        depth: 32表示每次卷积层的输出通道数
        minres：4表示最小的分辨率,也就是卷积的输出尺寸
        '''
        super(ConvEncoder, self).__init__()
        act = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        # 这里就是计算要将输入的图片压缩到最小分辨率的输出尺寸所需要的层数
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            # 这里是保证每次卷积后，输出的尺寸减少一半
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act())
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2

        # 这里应该是计算展平后的维度
        # 这里除以2是因为之前的循环最后一次乘以2是无效，所以要还原
        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        # 权重初始化
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        obs -= 0.5
        # 这里的转换也很象之前的dreamv2中的转换,但是不同的是dreamerv2中时time， batch h w c
        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch * time, ...) -> (batch * time, -1)
        # 展平
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act=nn.ELU,
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        '''
        卷积解码器，应该是反卷积吧

        这里仅仅只做上菜样
        '''

        super(ConvDecoder, self).__init__()
        act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        # 这里计算的是从特征编码到最小分辨率的输出尺寸所需要的层数
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        # 计算到最小分辨率是的输出通道数
        out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        # todo embed_size是什么
        self._embed_size = out_ch

        # 将rssm输出的特征编码映射到最小分辨率的输出尺寸
        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        # 计算每一层的输入输出通道数，由于minres时最小分辨率，所以输入通道数时输出通道数的两倍，因为时h和w
        in_dim = out_ch // (minres**2)
        # 每经过一层计算，输出通道数减半
        out_dim = in_dim // 2

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                # 确保最后一层输出的通道数时3
                out_dim = self._shape[0]
                # 不使用激活函数
                act = False
                bias = True
                # 不使用归一化
                norm = False

            if i != 0:
                # 第一层不需要改变输入输出通道数
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            # 这里应该是利用反卷积的公式，计算每次上菜样尺寸翻倍需要填充的大小
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act())
            in_dim = out_dim
            # 每次输出通道数减半
            out_dim //= 2
            # 每次尺寸翻倍
            h, w = h * 2, w * 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        # (batch, time, -1) -> (batch, time, ch, h, w)
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch, time, ch, h, w) -> (batch, time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        else:
            mean += 0.5
        return mean


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act="SiLU",
        norm=True,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="cuda",
        name="NoName",
    ):
        '''
        传入rssm编码的输出
        具体参数查看config文件

        shape是空的
       在世界状态模型和价值模型里面 std的值不会改变
       但是在动作这个值会改变
        '''
        super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        # 在世界状态模型中，因为在代码中self._shape是()，所以这里会转变为(1,)
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device

        self.layers = nn.Sequential()
        # 构建解码层
        for i in range(layers):
            self.layers.add_module(
                f"{name}_linear{i}", nn.Linear(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.add_module(
                    f"{name}_norm{i}", nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.add_module(f"{name}_act{i}", act())
            if i == 0:
                inp_dim = units
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict()
            for name, shape in self._shape.items():
                self.mean_layer[name] = nn.Linear(inp_dim, np.prod(shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.ModuleDict()
                for name, shape in self._shape.items():
                    self.std_layer[name] = nn.Linear(inp_dim, np.prod(shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            # 因为时(1,)所以会进入这里
            self.mean_layer = nn.Linear(inp_dim, np.prod(self._shape))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if self._std == "learned":
                # 在动作预测时，会进入这里，因为要计算均值和方差预测动作
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = nn.Linear(units, np.prod(self._shape))
                self.std_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = self.layers(x)
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if self._std == "learned":
                std = self.std_layer(out)
            else:
                std = self._std
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        if dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(
                dist, tools.TanhBijector()
            )
            dist = torchd.independent.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
            dist = tools.ContDist(
                torchd.independent.Independent(dist, 1), absmax=self._absmax
            )
        elif dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif dist == "onehot_gumble":
            dist = tools.ContDist(
                torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax
            )
        elif dist == "huber":
            dist = tools.ContDist(
                torchd.independent.Independent(
                    tools.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = tools.Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=mean), len(shape)
                )
            )
        elif dist == "symlog_disc":
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSamePad(torch.nn.Conv2d):
    def calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class ImgChLayerNorm(nn.Module):

    '''
    todo 很奇怪的正则化方式
    '''
    def __init__(self, ch, eps=1e-03):
        super(ImgChLayerNorm, self).__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
