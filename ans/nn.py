from typing import Any, Optional, Self, Union

import torch

from ans.autograd import Variable


class Function:

    @classmethod
    def apply(cls, *inputs: Any, **params: Any) -> Variable:
        tensor_args = [i.data if isinstance(i, Variable) else i for i in inputs]
        output_data, cache = cls.forward(*tensor_args, **params)

        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, ...]:
            dinputs = cls.backward(dout, cache=cache)
            return tuple(dinputs[i] for i, inp in enumerate(inputs) if isinstance(inp, Variable))

        grad_fn.name = f"{cls.__name__}.backward"
        return Variable(output_data, parents=tuple(i for i in inputs if isinstance(i, Variable)), grad_fn=grad_fn)

    @staticmethod
    def forward(*inputs: torch.Tensor, **params: Any) -> tuple[torch.Tensor, tuple]:
        raise NotImplementedError

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return str(self)


class BatchNorm1dFunction(Function):

    @staticmethod
    def forward(
        input: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        momentum: float = 0.1,
        eps: float = 1e-05,
        training: bool = False,
    ) -> tuple[torch.Tensor, tuple]:
        """

        Args:
            input: shape (num_samples, num_features)
            weight: shape (num_features,)
            bias: shape (num_features,)
            running_mean: shape (num_features,)
            running_var: shape (num_features,)
            momentum: running average smoothing coefficient
            eps: for numerical stabilization
            training: whether in training mode or eval mode
        Returns:
            output: shape (num_samples, num_features)
            cache: tuple of intermediate results to use in backward
        """

        N, D = input.shape  # N=num_samples, D=num_features

        if weight is None:
            weight = Variable(torch.ones(D, dtype=input.dtype, device=input.device), name='weight')

        if bias is None:
            bias = Variable(torch.zeros(D, dtype=input.dtype, device=input.device), name='bias')

        # training mode
        if training:    
            batch_mean = input.mean(dim=0)
            batch_var = input.var(dim=0, unbiased=False)

            # normilize input
            x_hat = (input - batch_mean) / torch.sqrt(batch_var + eps)

            # accumulate running mean and var
            if running_mean is not None and running_var is not None:
                running_mean.data = (1. - momentum) * running_mean + momentum * batch_mean
                running_var.data = (1. - momentum) * running_var + momentum * (batch_var * N / (N - 1))

            # cached necessary parametrs
            cache = (N, x_hat, weight, batch_var, eps, training)

        # evaluation mode
        else:
            x_hat = (input - running_mean) / torch.sqrt(running_var + eps)

            # cached necessary parametrs
            cache = (N, x_hat, weight, running_var, eps, training)      

        # calculate output
        output = weight.data * x_hat + bias.data

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_features)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """
        # extract context from forward pass
        n_samples, x_hat, weight, var, eps, training = cache

        # evaluation mode:
        if not training:
            # gradient on input
            dinput = (doutput * weight.data) / torch.sqrt(var + eps)
        # training mode:
        else:
            # gradient on input
            dinput = weight.data / torch.sqrt(var + eps) * (doutput - doutput.mean(dim=0) - 1/n_samples * x_hat * torch.sum(doutput * x_hat, dim=0))
        
        # gradients on weight and bias
        dweight = torch.sum(doutput * x_hat, dim=0)
        dbias = torch.sum(doutput, dim=0)
        
        return (dinput, dweight, dbias)


class BatchNorm2dFunction(Function):

    @staticmethod
    def forward(
        input: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        momentum: float = 0.9,
        eps: float = 1e-05,
        training: bool = False,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Spatial BatchNorm for convolutional networks

        Args:
            input: shape (num_samples, num_channels, height, width)
            weight: shape (num_channels,)
            bias: shape (num_channels,)
            running_mean: shape (num_channels,)
            running_var: shape (num_channels,)
            momentum: running average smoothing coefficient
            eps: for numerical stabilization
            training: whether in training mode or eval mode
        Returns:
            output: shape (num_samples, num_channels, height, width)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_channels, height, width)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return dinput, dweight, dbias


class DropoutFunction(Function):

    @staticmethod
    def forward(
        input: torch.Tensor,
        p_drop: float = 0.5,
        training: bool = False,
    ) -> tuple[torch.Tensor, tuple]:
        
        if training:
            # init m as random number from Uniform[0, 1] for every data instance of input
            m = torch.rand_like(input, dtype=input.dtype, device=input.device)

            # calculate output for every data instance if its m >= p_dout else 0
            output = input / (1 - p_drop) * (m >= p_drop).int()
            cache = (p_drop, m)
        else:
            # if its not training mode, behaives as identity
            output = input.clone()
            cache = ()

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        if len(cache) != 0:
            p_drop, m = cache
            # calculate input for every data instance if its m >= p_dout else 0
            dinput = doutput / (1 - p_drop) * (m >= p_drop).int()
        else:
            dinput = doutput.clone()
        return (dinput,)


class Conv2dFunction(Function):

    @staticmethod
    def forward(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input: shape (num_samples, num_channels, height, width)
            weight: shape (num_filters, num_channels, kernel_size[0], kernel_size[1])
            bias: shape (num_filters,)
            stride: convolution step size
            padding: how much should the input be padded on each side by zeroes
            dilation: see torch.nn.functional.conv2d
            groups: see torch.nn.functional.conv2d

        Returns:
            output: shape (num_samples, num_filters, output_height, output_width)
            cache: tuple of intermediate results to use in backward
        """

        # apply convolution
        output = torch.nn.functional.conv2d(
            input=input,
            weight=weight,
            bias=torch.zeros(weight.shape[0]) if bias is None else bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )

        # save the required parameters
        cache = (input, weight, bias, stride, padding, dilation, groups)

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_filters, output_height, output_width)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input, weight and bias
        """

        input, weight, bias, stride, padding, dilation, groups = cache

        kernel_size = weight.shape[2]
        num_samples, in_channels, h, w = input.shape  # num_samples, in_channels, in_height, in_width
        out_channels, h_out, w_out = doutput.shape[1:]  # out_channels, out_height, out_width

        output_padding = (
            w - ((w_out - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1),
            h - ((h_out - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1),
        )
        
        # gradient w.r.t. inputs
        dinput = torch.nn.functional.conv_transpose2d(
            doutput,
            weight,
            torch.zeros(in_channels),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            output_padding=output_padding
        )

        # gradient w.r.t. weights
        dweight = torch.zeros_like(weight)
        for c in range(in_channels):
            for f in range(out_channels):
                # accumulate over the batch
                for n in range(num_samples):
                    conv = torch.nn.functional.conv2d(
                        input[n:n+1, c:c+1, :, :],
                        doutput[n:n+1, f:f+1, :, :],
                        None,
                        stride=dilation,
                        padding=padding,
                        dilation=stride,
                    )
                    # crop the convolution result to match the kernel size
                    conv_cropped = conv[:, :, :kernel_size, :kernel_size]
                    dweight[f, c] += conv_cropped[0].squeeze(0)

        # gradient w.r.t. bias
        dbias = doutput.sum(dim=(0, 2, 3)) if bias is not None else None

        return dinput, dweight, dbias


class MaxPool2dFunction(Function):

    @staticmethod
    def forward(input: torch.Tensor, kernel_size: int = 2) -> tuple[torch.Tensor, tuple]:
        """

        Args:
            input: shape (num_samples, num_channels, height, width)
            window_size: size of pooling window
        Returns:
            output: shape (num_samples, num_channels, height / window_size, width / window_size)
            cache: tuple of intermediate results to use in backward
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return output, cache

    @staticmethod
    def backward(doutput: torch.Tensor, cache=()) -> tuple[torch.Tensor, ...]:
        """
        Args:
            doutput: gradient w.r.t. output of the forward pass; shape (num_samples, num_channels, height / window_size, width / window_size)
            cache: cache from the forward pass
        Returns:
            tuple of gradients w.r.t. input (single-element tuple)
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

        return (dinput,)


class Module:

    def __init__(self) -> None:
        self.training = True

    def __call__(self, *x: Variable) -> Variable:
        return self.forward(*x)

    def device(self) -> torch.device:
        return next(iter(self.parameters())).data.device

    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).data.dtype

    def forward(self, *x: Variable) -> Variable:
        raise NotImplementedError

    def named_modules(self) -> list[tuple[str, Self]]:
        named_modules = []

        def depth_first_append(obj, prefix=''):
            if isinstance(obj, Module):
                named_modules.append((prefix, obj))
                for name in dir(obj):
                    attr = getattr(obj, name)
                    if isinstance(attr, (list, tuple)):
                        for i, item in enumerate(attr):
                            depth_first_append(item, prefix=f"{prefix}.{i}" if prefix else str(i))
                    else:
                        depth_first_append(attr, prefix=f"{prefix}.{name}" if prefix else name)

        depth_first_append(self)
        return named_modules

    def named_parameters(self) -> list[tuple[str, Variable]]:
        return [
            (f"{name + '.' if name else ''}{attr}", getattr(module, attr))
            for name, module in self.named_modules()
            for attr in dir(module)
            if isinstance(getattr(module, attr), Variable)
        ]

    def parameters(self) -> list[Variable]:
        return [p for n, p in self.named_parameters()]

    def num_params(self) -> int:
        return sum(p.data.numel() for p in self.parameters())

    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[str] = None) -> Self:
        def to(obj: Any) -> None:
            if isinstance(obj, torch.Tensor):
                obj.data = obj.to(dtype=dtype, device=device)
            elif isinstance(obj, (tuple, list)):
                for elem in obj:
                    to(elem)
            elif isinstance(obj, dict):
                for val in obj.values():
                    to(val)
            elif isinstance(obj, Variable):
                to(obj.data)
                to(obj.grad)
            elif isinstance(obj, Module):
                for attr in dir(obj):
                    to(getattr(obj, attr))

        to(self)
        return self

    def train(self) -> None:
        for name, layer in self.named_modules():
            layer.training = True

    def eval(self) -> None:
        for name, layer in self.named_modules():
            layer.training = False

    def zero_grad(self) -> None:
        for name, par in self.named_parameters():
            par.grad = None


class Linear(Module):

    def __init__(self, num_in: int, num_out: int) -> None:
        super().__init__()
        # Xavier/Glorot initialization for weights
        limit = 1 / torch.sqrt(torch.tensor(num_in, dtype=torch.float32))
        self.weight = Variable(torch.empty(num_in, num_out).uniform_(-limit, limit))
        self.bias = Variable(torch.empty(num_out).uniform_(-limit, limit))
        
    def forward(self, x: Variable) -> Variable:
        return x @ self.weight + self.bias


class Sigmoid(Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Variable) -> Variable:
        return x.sigmoid()


class ReLU(Module):

    def __init__(self, negative_slope: float = 0.0) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Variable) -> Variable:
        return x.relu()


class Dropout(Module):

    def __init__(self, p_drop: float = 0.5) -> None:
        super().__init__()
        self.p_drop = p_drop

    def forward(self, x: Variable) -> Variable:
        return DropoutFunction.apply(x, p_drop=self.p_drop, training=self.training)



class BatchNorm1d(Module):

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.affine = affine

        self.weight = Variable(torch.ones(self.num_features), name='weight') if self.affine else None
        self.bias = Variable(torch.zeros(self.num_features), name='bias') if self.affine else None
        self.running_mean = torch.zeros(self.num_features)
        self.running_var = torch.ones(self.num_features)

    def forward(self, x: Variable) -> Variable:
        return BatchNorm1dFunction.apply(
            x,
            self.weight,
            self.bias,
            running_mean=self.running_mean,
            running_var=self.running_var,
            momentum=self.momentum,
            eps=self.eps,
            training=self.training,
        )


class BatchNorm2d(BatchNorm1d):

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class Conv2d(Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Xavier (Glorot) uniform init
        bound = 1. / (in_channels * kernel_size ** 2) ** 0.5 
        self.weight = Variable(torch.FloatTensor(out_channels, in_channels // groups, kernel_size, kernel_size).uniform_(-bound, bound))
        self.bias = Variable(torch.zeros(out_channels)) if bias is True else None

    def forward(self, x: Variable) -> Variable:
        return Conv2dFunction.apply(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class MaxPool2d(Module):

    def __init__(self, kernel_size: int) -> None:
        super().__init__()

        self.kernel_size = kernel_size

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class Flatten(Module):

    def forward(self, x: Variable) -> Variable:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################


class Sequential(Module):

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: Variable) -> Variable:
        for layer in self.layers:
            x = layer.forward(x)
        return x


class Optimizer:

    def __init__(self, parameters: list[Variable]) -> None:
        self.parameters = parameters

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = None


class SGD(Optimizer):

    def __init__(
        self, parameters: list[Variable], learning_rate: float = 1e-3, momentum: float = 0.0, weight_decay: float = 0.0
    ) -> None:
        super().__init__(parameters)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        # init velocities to zeros
        self._velocities: dict[Variable, torch.Tensor] = {param: torch.tensor(0.0, dtype=param.data.dtype, device=param.data.device) for param in self.parameters}

    def step(self) -> None:
        # for every model parametr do one normalisation step
        for param in self.parameters:
            qt = param.grad + self.weight_decay * param.data
            vt = self.momentum * self._velocities.get(param) - self.learning_rate * qt  # new velocity
            self._velocities[param] = vt
            param.data = param.data + vt  # accumulate parametr


class Adam(Optimizer):

    def __init__(
        self,
        parameters: list[Variable],
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self._num_steps = 0
        # buffers for the first and second moments, which are updated at each iteration
        self._m: dict[Variable, torch.Tensor] = {param: torch.tensor(0.0, dtype=param.data.dtype, device=param.data.device) for param in self.parameters}
        self._v: dict[Variable, torch.Tensor] = {param: torch.tensor(0.0, dtype=param.data.dtype, device=param.data.device) for param in self.parameters}

    def step(self) -> None:
        self._num_steps += 1

        # for every model parametr do one normalisation step
        for param in self.parameters:
            qt = param.grad + self.weight_decay * param.data
            
            # update buffers
            mt = self.beta1 * self._m.get(param) + (1 - self.beta1) * qt
            self._m[param] = mt
            vt = self.beta2 * self._v.get(param) + (1 - self.beta2) * qt**2
            self._v[param] = vt

            # offset corrections for the first and the second moments
            mu = mt / (1 - self.beta1 ** self._num_steps)
            vi = vt / (1 - self.beta2 ** self._num_steps)

            # accumulate parametr
            param.data = param.data - self.learning_rate * mu / (torch.sqrt(vi) + self.eps)
            
            
