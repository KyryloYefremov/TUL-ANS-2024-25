from numbers import Number
from typing import Callable, Optional, Self, TypeAlias, Union

import torch


# Variable can be added, subtracted, multiplied, and divided with the following
BinOpOtherType: TypeAlias = Union[Number, torch.Tensor, Self]


class Variable:

    def __init__(
            self,
            data: torch.Tensor,
            parents: tuple[Self, ...] = (),
            grad_fn: Optional[Callable[[torch.Tensor], tuple[torch.Tensor, ...]]] = None,
            name: Optional[str] = None
    ) -> None:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        self.data = data
        self.grad: Optional[torch.Tensor] = None
        self.parents = parents
        self.grad_fn = grad_fn
        self.name = name

    def __repr__(self):
        if hasattr(self.grad_fn, 'func'):
            grad_fn_repr = self.grad_fn.func.__qualname__
        elif self.grad_fn is not None:
            grad_fn_repr = self.grad_fn.__qualname__
        else:
            grad_fn_repr = 'None'
        if self.data.ndim == 0 or (self.data.shape[-1] == self.data.numel() and self.data.numel() < 5):
            return f"{self.__class__.__name__}({self.data}, grad_fn={grad_fn_repr})"
        else:
            return f"{self.__class__.__name__}(shape={tuple(self.data.shape)}, grad_fn={grad_fn_repr})"
        
    @staticmethod
    def broadcast(grad: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """
        Adjusts the shape of gradient tensor to match the target shape by summing along broadcasted dimensions.
        @params:
            grad - The gradient tensor that shape needs to be adjusted.
            target_shape - The target shape. This is the shape of the input tensor before broadcasting.
        @return:
            torch.Tensor - The adjusted gradient tensor.
        """
        grad_shape = grad.shape
        
        # If target_shape is empty => scalar, sum all elements from grad to get a scalar value
        if not target_shape:
            return grad.sum()

        # Per dimenzion
        for dim in range(len(target_shape)):
            # If dims of grad and target differ - was broadcasted => sum this dim
            if grad_shape[dim] != target_shape[dim]:
                grad = grad.sum(dim=dim, keepdim=True)

        # If grad still has additional dims (is bigger than target), sum it from the end
        while len(grad.shape) > len(target_shape):
            grad = grad.sum(dim=-1)
        
        return grad

    def __add__(self, other: BinOpOtherType, is_reversed=False) -> Self:
        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Funtion that represents gradient computation for '+' operation. Used in backprop.
            @params:
                dout - gradient value from previous node.
            """
            # Compute grads
            grad_self = dout.clone()
            grad_other = dout.clone() if isinstance(other, Variable) else torch.tensor(0., dtype=self.data.dtype, device=self.data.device)
            
            # Use broadcasting for self grad
            if grad_self.shape != self.data.shape:
                grad_self = self.broadcast(grad_self, self.data.shape)
            
            # Use broadcasting for other grad if it is Variable
            if isinstance(other, Variable) and grad_other.shape != other.data.shape:
                grad_other = self.broadcast(grad_other, other.data.shape)
            
            # If adding is reverserd, change gradients
            if is_reversed:
                return grad_other, grad_self
            else:
                return grad_self, grad_other
        
        # If other isnt a Variable, create it as new one
        if not isinstance(other, Variable):
            other_var = self.__class__(other)
        else:
            other_var = other
        
        # Return result as new instance of Variable. Also reverse parents
        return self.__class__(
            self.data + other_var.data,
            parents = (self, other_var) if not is_reversed else (other_var, self),
            grad_fn = grad_fn
        )

    def __radd__(self, other: BinOpOtherType) -> Self:
        # Call basic add with reversed order
        return self.__add__(other, is_reversed=True)

    def __sub__(self, other: BinOpOtherType, is_reversed=False) -> Self:
        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Funtion that represents gradient computation for '-' operation. Used in backprop.
            @params:
                dout - gradient value from previous node.
            """
            # Compute grads
            grad_self = dout.clone()
            grad_other = dout.clone() if isinstance(other, Variable) else torch.tensor(0., dtype=self.data.dtype, device=self.data.device)

            # Use broadcasting for self grad
            if grad_self.shape != self.data.shape:
                grad_self = self.broadcast(grad_self, self.data.shape)

            # Use broadcasting for other grad if it is Variable
            if isinstance(other, Variable) and grad_other.shape != other.data.shape:
                grad_other = self.broadcast(grad_other, other.data.shape)

            # Reverse the gradients if the subtract operation is of inverted order. Gradient of right operand must be multiplied by -1
            if is_reversed:
                return grad_other, -grad_self
            else:
                return grad_self, -grad_other
        
        # If other isnt a Variable, create it as new one
        if not isinstance(other, Variable):
            other_var = self.__class__(other)
        else:
            other_var = other

        # Return result as new Variable instance. Reverse operands if __rsub__ was called. Also reverse parents
        return self.__class__(
            self.data - other_var.data if not is_reversed else other_var.data - self.data,
            parents=(self, other_var) if not is_reversed else (other_var, self),
            grad_fn=grad_fn
        )

    def __rsub__(self, other: BinOpOtherType) -> Self:
        # Call basic 'sub' with reversed orded
        return self.__sub__(other, is_reversed=True)

    def __mul__(self, other: BinOpOtherType, is_reversed=False) -> Self:
        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Funtion that represents gradient computation for '*' operation. Used in backprop.
            @params:
                dout - gradient value from previous node.
            """

            # Compute grads
            grad_self = dout * other.data if isinstance(other, Variable) else dout * other
            grad_other = dout * self.data if isinstance(other, Variable) else torch.tensor(0., dtype=self.data.dtype, device=self.data.device)

            # Use broadcasting for self grad
            if grad_self.shape != self.data.shape:
                grad_self = self.broadcast(grad_self, self.data.shape)
            # Use broadcasting for other grad
            if grad_other.shape != (other.data.shape if isinstance(other, Variable) else torch.Size()):
                grad_other = self.broadcast(grad_other, other.data.shape if isinstance(other, Variable) else torch.Size())

            # Reverse the gradients if the multiplicate operation is of inverted order
            if is_reversed:
                return grad_other, grad_self
            else:
                return grad_self, grad_other
        
        # If other is not a Variable instance, create it as new one
        if not isinstance(other, Variable):
            other_var = self.__class__(other)
        else:
            other_var = other

        # Return result as new instance of Variable. Also reverse parents if __rmul__ called.
        return self.__class__(
            self.data * other_var.data,
            parents=(self, other_var) if not is_reversed else (other_var, self),
            grad_fn=grad_fn
        )

    def __rmul__(self, other: BinOpOtherType) -> Self:
        # Call basic 'mul' with reversed order
        return self.__mul__(other, is_reversed=True)
    
    def __truediv__(self, other: BinOpOtherType, is_reversed=False) -> Self:
        def grad_fn(dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Funtion that represents gradient computation for '/' operation. Used in backprop.
            @params:
                dout - gradient value from previous node.
            """
            # Compute grads
            if isinstance(other, Variable):
                grad_self = dout / other.data if not is_reversed else -dout * other.data / (self.data ** 2)
                grad_other = -dout * self.data / (other.data ** 2)
            else:
                grad_self = dout / other if not is_reversed else -dout * other / (self.data ** 2)
                grad_other = torch.tensor(0., dtype=self.data.dtype, device=self.data.device)

            # Use broadcasting for self grad
            if grad_self.shape != self.data.shape:
                grad_self = self.broadcast(grad_self, self.data.shape)
            # Use broadcasting for other grad
            if grad_other.shape != (other.data.shape if isinstance(other, Variable) else torch.Size()):
                grad_other = self.broadcast(grad_other, other.data.shape if isinstance(other, Variable) else torch.Size())

            # Reverse the gradients if the multiplicate operation is of inverted order
            if is_reversed:
                return grad_other, grad_self
            else:
                return grad_self, grad_other
        
        # If other is not a Variable instance, create it as new one
        if not isinstance(other, Variable):
            other_var = self.__class__(other)
        else:
            other_var = other

        # Return result as new instance of Variable. Reverts operands if order is reversed. Also reverse parents if __rmul__ called.
        return self.__class__(
            self.data / other_var.data if not is_reversed else other_var.data / self.data,
            parents=(self, other_var) if not is_reversed else (other_var, self),
            grad_fn=grad_fn
        )

    def __rtruediv__(self, other: BinOpOtherType) -> Self:
        # Call basic 'truediv' with reversed order
        return self.__truediv__(other, is_reversed=True)

    def __matmul__(self, other: BinOpOtherType) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
    
    def __rmatmul__(self, other: BinOpOtherType) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################

    def __getitem__(self, item) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError
    
        # ENDTODO
        ########################################
    
    def log(self) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError
    
        # ENDTODO
        ########################################

    def exp(self) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError
    
        # ENDTODO
        ########################################
    
    def sigmoid(self) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError
    
        # ENDTODO
        ########################################
    
    def sum(self, dim: Union[None, int, tuple[int, ...]] = None, keepdim: bool = False) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError
    
        # ENDTODO
        ########################################
    
    def mean(self, dim: Union[None, int, tuple[int, ...]] = None, keepdim: bool = False) -> Self:
        ########################################
        # TODO: implement

        raise NotImplementedError
    
        # ENDTODO
        ########################################


    def backprop(self, dout: Optional[torch.Tensor] = None) -> None:
        """
        Runs full backpropagation starting from self. Fills the grad attribute with dself/dpredecessor for all
        predecessors of self.

        Args:
            dout: Incoming gradient on self; if None, then set to tensor of ones with proper shape and dtype
        """

        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
