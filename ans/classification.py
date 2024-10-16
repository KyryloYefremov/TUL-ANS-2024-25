from typing import Any, Self

import torch

import ans


class LinearSoftmaxModel:

    def __init__(self, in_size: int, out_size: int, weight_scale: float = 1e-3) -> None:
        ########################################

        self.weight = torch.randn(in_size, out_size) * weight_scale
        self.bias = torch.randn(out_size) * weight_scale

        ########################################

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3
    ) -> tuple[float, torch.Tensor]:
        """
        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
            learning_rate: gradient descent step size
        Returns:
            loss: average loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ########################################

        N = inputs.shape[0]

        logits = inputs @ self.weight + self.bias  # (N, K)

        # Compute softmax
        exp_logits = torch.exp(logits)
        softmax = exp_logits / exp_logits.sum(dim=1, keepdim=True)

        # Compute loss using cross-entropy
        log_probs = -torch.log(softmax[torch.arange(N), targets])  # Shape: (N,)
        loss = log_probs.mean().item()  # Total loss for a batch

        # Compute gradients
        softmax[torch.arange(N), targets] -= 1  # Gradient for correct classes
        grad_weight = (softmax.T @ inputs / N).T  # (K, D)
        grad_bias = softmax.mean(dim=0)  # (K,)

        # Update parameters
        self.weight -= learning_rate * grad_weight
        self.bias -= learning_rate * grad_bias
        
        # ENDTODO
        ########################################

        return loss, logits

    def val_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        """
        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
        Returns:
            loss: average loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ########################################
        
        N = inputs.shape[0]

        logits = inputs @ self.weight + self.bias  # (N, K)

        # Compute softmax
        exp_logits = torch.exp(logits)
        softmax = exp_logits / exp_logits.sum(dim=1, keepdim=True)

        # Compute loss using cross-entropy
        log_probs = -torch.log(softmax[torch.arange(N), targets])  # Shape: (N,)
        loss = log_probs.mean().item()  # Total loss for a batch

        # ENDTODO
        ########################################
        
        return loss, logits


class LinearSVMModel(LinearSoftmaxModel):
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3
    ) -> tuple[float, torch.Tensor]:
        ########################################
        N = inputs.shape[0]

        logits = inputs @ self.weight + self.bias  # (N, K)

        # Compute hinge loss
        correct_class_scores = logits[torch.arange(N), targets].unsqueeze(1)  # (N, 1)
        margins = torch.clamp(logits - correct_class_scores + 1, min=0)  # (N, K)
        margins[torch.arange(N), targets] = 0

        loss = (margins.sum() / N).item() # Mean hinge loss over batch

        # Compute gradients
        mask = (margins > 0).float()  # (N, K)
        mask[torch.arange(N), targets] = -mask.sum(dim=1)
        
        grad_weight = (mask.T @ inputs / N).T  # Shape: (K, D)
        grad_bias = mask.mean(dim=0)  # Shape: (K,)

        # Update parameters
        self.weight -= learning_rate * grad_weight
        self.bias -= learning_rate * grad_bias

        # ENDTODO
        ########################################

        return loss, logits

    def val_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        ########################################
        N = inputs.shape[0]

        logits = inputs @ self.weight + self.bias  # (N, K)

        # Compute hinge loss
        correct_class_scores = logits[torch.arange(N), targets].unsqueeze(1)  # (N, 1)
        margins = torch.clamp(logits - correct_class_scores + 1, min=0)  # (N, K)
        margins[torch.arange(N), targets] = 0
        
        loss = (margins.sum() / N).item()  # Mean hinge loss over batch

        # ENDTODO
        ########################################
        
        return loss, logits


class TwoLayerPerceptron:

    def __init__(self, in_size: int, hidden_size: int, out_size: int, weight_scale: float = 0.001) -> None:
        ########################################
        # TODO: implement

        self.weight1 = ...
        self.bias1 = ...
        self.weight2 = ...
        self.bias2 = ...
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        learning_rate: float = 1e-3
    ) -> tuple[float, torch.Tensor]:
        ########################################
        # TODO: implement
        # 
        # Some of the intermediate gradients that are to be expected for the test data used in
        # test_two_layer_perceptron.TestTrainStep:
        #   dlogits = torch.tensor([  # gradient w.r.t. output of the network
        #     [ 0.0072,  0.0924, -0.0996],
        #     [-0.0484,  0.0199,  0.0285],
        #     [ 0.0047, -0.0258,  0.0211],
        #     [ 0.0126,  0.0778, -0.0904],
        #     [ 0.0333, -0.0696,  0.0362],
        #     [ 0.0696,  0.0257, -0.0953],
        #     [-0.0290,  0.0066,  0.0223],
        #     [ 0.1003,  0.0059, -0.1062]
        #  ])
        #  dhidden = torch.tensor([  # gradient w.r.t. output of sigmoid
        #    [ 0.1521,  0.0635, -0.1338, -0.0515],
        #    [ 0.0342,  0.0584, -0.1164,  0.0261],
        #    [-0.0427, -0.0237,  0.0491,  0.0094],
        #    [ 0.1279,  0.0477, -0.1013, -0.0482],
        #    [-0.1156, -0.0825,  0.1688,  0.0099],
        #    [ 0.0404, -0.0429,  0.0813, -0.0646],
        #    [ 0.0118,  0.0310, -0.0613,  0.0182],
        #    [ 0.0070, -0.0853,  0.1665, -0.0776]
        #  ])
        #  dprehidden = torch.tensor([  # gradient w.r.t. output of the first linear layer (before sigmoid)
        #    [ 0.0338,  0.0038, -0.0153, -0.0002],
        #    [ 0.0047,  0.0002, -0.0291,  0.0004],
        #    [-0.0092, -0.0003,  0.0007,  0.0001],
        #    [ 0.0315,  0.0097, -0.0107, -0.0039],
        #    [-0.0284, -0.0081,  0.0328,  0.0020],
        #    [ 0.0091, -0.0002,  0.0203, -0.0102],
        #    [ 0.0003,  0.0007, -0.0072,  0.0028],
        #    [ 0.0003, -0.0001,  0.0153, -0.0004]])
        #  ])

        raise NotImplementedError

        # ENDTODO
        ########################################

        return loss, logits
    
    def val_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        """
        Args:
            inputs: input data batch; shape (N, D)
            targets: vector of class indicies (integers); shape (N,)
        Returns:
            loss: average loss over the batch; float
            logits: classification scores predicted on the batch; shape (N, K)
        """
        ########################################
        # TODO: implement

        raise NotImplementedError

        # ENDTODO
        ########################################
        
        return loss, logits
    
    def save(self, filename: str) -> None:
        torch.save(
            dict(weight1=self.weight1, bias1=self.bias1, weight2=self.weight2, bias2=self.bias2),
            filename
        )
    
    @classmethod
    def load(cls, filename: str) -> Self:
        dic = torch.load(filename, weights_only=True)
        model = cls(*dic['weight1'].shape, dic['weight2'].shape[1])
        model.weight1 = dic['weight1']
        model.bias1 = dic['bias1']
        model.weight2 = dic['weight2']
        model.bias2 = dic['bias2']
        return model


def accuracy(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Args:
        scores: output linear scores (logits or probabilities); shape (num_samples, num_classes)
        targets: vector of class indicies (integers); shape (num_samples,)
    Returns:
        acc: averare accuracy on the batch; tensor containing single number (scalar), e.g. "tensor(0.364)"
    """
    
    ########################################
    predicted_classes = torch.argmax(scores, dim=1)  # Find indeces of the highest scores = the highest probability

    correct_predictions = (predicted_classes == targets).float()  # (num_samples,)
    acc = (1 / scores.shape[0]) * torch.sum(correct_predictions)
    acc = acc.item()
    
    # ENDTODO
    ########################################
    
    return acc


def train_epoch(
    model: LinearSoftmaxModel,
    loader: ans.data.BatchLoader,
    **train_step_kwargs
) -> Any:
    """
    Trains `model` on the dataset `loader` for one epoch.

    Args:
        model: Model to train. Must have method `train_step`.
        loader: loader of the training dataset
    Returns:
        return whatever is needed
    """
    ########################################

    total_loss = 0.0
    correct_predictions = 0
    num_samples = 0

    # Iterate over each batch in the train loader
    for inputs, targets in loader:
        batch_loss, logits = model.train_step(inputs, targets, **train_step_kwargs)
        
        total_loss += batch_loss * inputs.shape[0]  # Scale loss
        correct_predictions += (logits.argmax(dim=1) == targets).sum().item()
        num_samples += inputs.shape[0]

    mean_loss = total_loss / num_samples
    mean_acc = correct_predictions / num_samples

    return mean_loss, mean_acc

    # ENDTODO
    ########################################


def validate(
    model: LinearSoftmaxModel,
    loader: ans.data.BatchLoader,
) -> tuple[float, float]:
    """
    Validates `model` on the dataset `loader`.

    Args:
        model: Model to be validated. Must have method `val_step`.
        loader: loader of the training dataset
    Returns:
        mean_loss: average loss achieved on the dataset during training
        mean_acc: average accuracy achieved on the dataset during model training
    """
    ########################################
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Iterate over each batch in the validation loader
    for inputs, targets in loader:
        loss, logits = model.val_step(inputs, targets)
        
        total_loss += loss * inputs.size(0)  # Scale loss by batch size
        
        # Calculate accuracy for this batch
        _, predicted_classes = torch.max(logits, dim=1)
        batch_predictions = (predicted_classes == targets).sum().item()
        correct_predictions += batch_predictions
        
        total_samples += inputs.size(0)

    mean_loss = total_loss / total_samples if total_samples > 0 else 0.0
    mean_acc = correct_predictions / total_samples if total_samples > 0 else 0.0

    return mean_loss, mean_acc

    # ENDTODO
    ########################################
