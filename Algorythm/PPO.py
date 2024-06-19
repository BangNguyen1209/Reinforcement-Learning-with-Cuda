import warnings
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

#from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

#from Algorythm.ActorCriticPolicy import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from Algorythm.OnPolicyAlgorithm import OnPolicyAlgorithm

SelfPPO = TypeVar("SelfPPO", bound="PPO")

"""Policies: abstract base class and concrete implementations."""

# import collections
# import copy
# import warnings
# from abc import ABC, abstractmethod
# from functools import partial
# from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

# import numpy as np
# import torch as th
# from gymnasium import spaces
# from torch import nn

# from stable_baselines3.common.distributions import (
#     BernoulliDistribution,
#     CategoricalDistribution,
#     DiagGaussianDistribution,
#     Distribution,
#     MultiCategoricalDistribution,
#     StateDependentNoiseDistribution,
#     make_proba_distribution,
# )
# from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
# from stable_baselines3.common.torch_layers import (
#     BaseFeaturesExtractor,
#     CombinedExtractor,
#     FlattenExtractor,
#     MlpExtractor,
#     NatureCNN,
#     create_mlp,
# )
# from stable_baselines3.common.type_aliases import Schedule
# from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor

# SelfBaseModel = TypeVar("SelfBaseModel", bound="BaseModel")


# class BaseModel(nn.Module):
#     """
#     The base model object: makes predictions in response to observations.

#     In the case of policies, the prediction is an action. In the case of critics, it is the
#     estimated value of the observation.

#     :param observation_space: The observation space of the environment
#     :param action_space: The action space of the environment
#     :param features_extractor_class: Features extractor to use.
#     :param features_extractor_kwargs: Keyword arguments
#         to pass to the features extractor.
#     :param features_extractor: Network to extract features
#         (a CNN when using images, a nn.Flatten() layer otherwise)
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     """

#     optimizer: th.optim.Optimizer

#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         features_extractor: Optional[BaseFeaturesExtractor] = None,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#     ):
#         super().__init__()

#         if optimizer_kwargs is None:
#             optimizer_kwargs = {}

#         if features_extractor_kwargs is None:
#             features_extractor_kwargs = {}

#         self.observation_space = observation_space
#         self.action_space = action_space
#         self.features_extractor = features_extractor
#         self.normalize_images = normalize_images

#         self.optimizer_class = optimizer_class
#         self.optimizer_kwargs = optimizer_kwargs

#         self.features_extractor_class = features_extractor_class
#         self.features_extractor_kwargs = features_extractor_kwargs
#         # Automatically deactivate dtype and bounds checks
#         if normalize_images is False and issubclass(features_extractor_class, (NatureCNN, CombinedExtractor)):
#             self.features_extractor_kwargs.update(dict(normalized_image=True))

#     def _update_features_extractor(
#         self,
#         net_kwargs: Dict[str, Any],
#         features_extractor: Optional[BaseFeaturesExtractor] = None,
#     ) -> Dict[str, Any]:
#         """
#         Update the network keyword arguments and create a new features extractor object if needed.
#         If a ``features_extractor`` object is passed, then it will be shared.

#         :param net_kwargs: the base network keyword arguments, without the ones
#             related to features extractor
#         :param features_extractor: a features extractor object.
#             If None, a new object will be created.
#         :return: The updated keyword arguments
#         """
#         net_kwargs = net_kwargs.copy()
#         if features_extractor is None:
#             # The features extractor is not shared, create a new one
#             features_extractor = self.make_features_extractor()
#         net_kwargs.update(dict(features_extractor=features_extractor, features_dim=features_extractor.features_dim))
#         return net_kwargs

#     def make_features_extractor(self) -> BaseFeaturesExtractor:
#         """Helper method to create a features extractor."""
#         return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)

#     def extract_features(self, obs: th.Tensor, features_extractor: BaseFeaturesExtractor) -> th.Tensor:
#         """
#         Preprocess the observation if needed and extract features.

#          :param obs: The observation
#          :param features_extractor: The features extractor to use.
#          :return: The extracted features
#         """
#         preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
#         return features_extractor(preprocessed_obs)

#     def _get_constructor_parameters(self) -> Dict[str, Any]:
#         """
#         Get data that need to be saved in order to re-create the model when loading it from disk.

#         :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
#         """
#         return dict(
#             observation_space=self.observation_space,
#             action_space=self.action_space,
#             # Passed to the constructor by child class
#             # squash_output=self.squash_output,
#             # features_extractor=self.features_extractor
#             normalize_images=self.normalize_images,
#         )

#     @property
#     def device(self) -> th.device:
#         """Infer which device this policy lives on by inspecting its parameters.
#         If it has no parameters, the 'cpu' device is used as a fallback.

#         :return:"""
#         for param in self.parameters():
#             return param.device
#         return get_device("cpu")

#     def save(self, path: str) -> None:
#         """
#         Save model to a given location.

#         :param path:
#         """
#         th.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)

#     @classmethod
#     def load(cls: Type[SelfBaseModel], path: str, device: Union[th.device, str] = "auto") -> SelfBaseModel:
#         """
#         Load model from path.

#         :param path:
#         :param device: Device on which the policy should be loaded.
#         :return:
#         """
#         device = get_device(device)
#         saved_variables = th.load(path, map_location=device)

#         # Create policy object
#         model = cls(**saved_variables["data"])  # pytype: disable=not-instantiable
#         # Load weights
#         model.load_state_dict(saved_variables["state_dict"])
#         model.to(device)
#         return model

#     def load_from_vector(self, vector: np.ndarray) -> None:
#         """
#         Load parameters from a 1D vector.

#         :param vector:
#         """
#         th.nn.utils.vector_to_parameters(th.as_tensor(vector, dtype=th.float, device=self.device), self.parameters())

#     def parameters_to_vector(self) -> np.ndarray:
#         """
#         Convert the parameters to a 1D vector.

#         :return:
#         """
#         return th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()

#     def set_training_mode(self, mode: bool) -> None:
#         """
#         Put the policy in either training or evaluation mode.

#         This affects certain modules, such as batch normalisation and dropout.

#         :param mode: if true, set to training mode, else set to evaluation mode
#         """
#         self.train(mode)

#     def is_vectorized_observation(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> bool:
#         """
#         Check whether or not the observation is vectorized,
#         apply transposition to image (so that they are channel-first) if needed.
#         This is used in DQN when sampling random action (epsilon-greedy policy)

#         :param observation: the input observation to check
#         :return: whether the given observation is vectorized or not
#         """
#         vectorized_env = False
#         if isinstance(observation, dict):
#             for key, obs in observation.items():
#                 obs_space = self.observation_space.spaces[key]
#                 vectorized_env = vectorized_env or is_vectorized_observation(maybe_transpose(obs, obs_space), obs_space)
#         else:
#             vectorized_env = is_vectorized_observation(
#                 maybe_transpose(observation, self.observation_space), self.observation_space
#             )
#         return vectorized_env

#     def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[th.Tensor, bool]:
#         """
#         Convert an input observation to a PyTorch tensor that can be fed to a model.
#         Includes sugar-coating to handle different observations (e.g. normalizing images).

#         :param observation: the input observation
#         :return: The observation as PyTorch tensor
#             and whether the observation is vectorized or not
#         """
#         vectorized_env = False
#         if isinstance(observation, dict):
#             # need to copy the dict as the dict in VecFrameStack will become a torch tensor
#             observation = copy.deepcopy(observation)
#             for key, obs in observation.items():
#                 obs_space = self.observation_space.spaces[key]
#                 if is_image_space(obs_space):
#                     obs_ = maybe_transpose(obs, obs_space)
#                 else:
#                     obs_ = np.array(obs)
#                 vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
#                 # Add batch dimension if needed
#                 observation[key] = obs_.reshape((-1, *self.observation_space[key].shape))

#         elif is_image_space(self.observation_space):
#             # Handle the different cases for images
#             # as PyTorch use channel first format
#             observation = maybe_transpose(observation, self.observation_space)

#         else:
#             observation = np.array(observation)

#         if not isinstance(observation, dict):
#             # Dict obs need to be handled separately
#             vectorized_env = is_vectorized_observation(observation, self.observation_space)
#             # Add batch dimension if needed
#             observation = observation.reshape((-1, *self.observation_space.shape))

#         observation = obs_as_tensor(observation, self.device)
#         return observation, vectorized_env


# class BasePolicy(BaseModel, ABC):
#     """The base policy object.

#     Parameters are mostly the same as `BaseModel`; additions are documented below.

#     :param args: positional arguments passed through to `BaseModel`.
#     :param kwargs: keyword arguments passed through to `BaseModel`.
#     :param squash_output: For continuous actions, whether the output is squashed
#         or not using a ``tanh()`` function.
#     """

#     features_extractor: BaseFeaturesExtractor

#     def __init__(self, *args, squash_output: bool = False, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._squash_output = squash_output

#     @staticmethod
#     def _dummy_schedule(progress_remaining: float) -> float:
#         """(float) Useful for pickling policy."""
#         del progress_remaining
#         return 0.0

#     @property
#     def squash_output(self) -> bool:
#         """(bool) Getter for squash_output."""
#         return self._squash_output

#     @staticmethod
#     def init_weights(module: nn.Module, gain: float = 1) -> None:
#         """
#         Orthogonal initialization (used in PPO and A2C)
#         """
#         if isinstance(module, (nn.Linear, nn.Conv2d)):
#             nn.init.orthogonal_(module.weight, gain=gain)
#             if module.bias is not None:
#                 module.bias.data.fill_(0.0)

#     @abstractmethod
#     def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
#         """
#         Get the action according to the policy for a given observation.

#         By default provides a dummy implementation -- not all BasePolicy classes
#         implement this, e.g. if they are a Critic in an Actor-Critic method.

#         :param observation:
#         :param deterministic: Whether to use stochastic or deterministic actions
#         :return: Taken action according to the policy
#         """

#     def predict(
#         self,
#         observation: Union[np.ndarray, Dict[str, np.ndarray]],
#         state: Optional[Tuple[np.ndarray, ...]] = None,
#         episode_start: Optional[np.ndarray] = None,
#         deterministic: bool = False,
#     ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
#         """
#         Get the policy action from an observation (and optional hidden state).
#         Includes sugar-coating to handle different observations (e.g. normalizing images).

#         :param observation: the input observation
#         :param state: The last hidden states (can be None, used in recurrent policies)
#         :param episode_start: The last masks (can be None, used in recurrent policies)
#             this correspond to beginning of episodes,
#             where the hidden states of the RNN must be reset.
#         :param deterministic: Whether or not to return deterministic actions.
#         :return: the model's action and the next hidden state
#             (used in recurrent policies)
#         """
#         # Switch to eval mode (this affects batch norm / dropout)
#         self.set_training_mode(False)

#         observation, vectorized_env = self.obs_to_tensor(observation)

#         with th.no_grad():
#             actions = self._predict(observation, deterministic=deterministic)
#         # Convert to numpy, and reshape to the original action shape
#         actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

#         if isinstance(self.action_space, spaces.Box):
#             if self.squash_output:
#                 # Rescale to proper domain when using squashing
#                 actions = self.unscale_action(actions)
#             else:
#                 # Actions could be on arbitrary scale, so clip the actions to avoid
#                 # out of bound error (e.g. if sampling from a Gaussian distribution)
#                 actions = np.clip(actions, self.action_space.low, self.action_space.high)

#         # Remove batch dimension if needed
#         if not vectorized_env:
#             actions = actions.squeeze(axis=0)

#         return actions, state

#     def scale_action(self, action: np.ndarray) -> np.ndarray:
#         """
#         Rescale the action from [low, high] to [-1, 1]
#         (no need for symmetric action space)

#         :param action: Action to scale
#         :return: Scaled action
#         """
#         low, high = self.action_space.low, self.action_space.high
#         return 2.0 * ((action - low) / (high - low)) - 1.0

#     def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
#         """
#         Rescale the action from [-1, 1] to [low, high]
#         (no need for symmetric action space)

#         :param scaled_action: Action to un-scale
#         """
#         low, high = self.action_space.low, self.action_space.high
#         return low + (0.5 * (scaled_action + 1.0) * (high - low))


# class ActorCriticPolicy(BasePolicy):
#     """
#     Policy class for actor-critic algorithms (has both policy and value prediction).
#     Used by A2C, PPO and the likes.

#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param ortho_init: Whether to use or not orthogonal initialization
#     :param use_sde: Whether to use State Dependent Exploration or not
#     :param log_std_init: Initial value for the log standard deviation
#     :param full_std: Whether to use (n_features x n_actions) parameters
#         for the std instead of only (n_features,) when using gSDE
#     :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
#         a positive standard deviation (cf paper). It allows to keep variance
#         above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
#     :param squash_output: Whether to squash the output using a tanh function,
#         this allows to ensure boundaries when using gSDE.
#     :param features_extractor_class: Features extractor to use.
#     :param features_extractor_kwargs: Keyword arguments
#         to pass to the features extractor.
#     :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     """

#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         lr_schedule: Schedule,
#         net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
#         activation_fn: Type[nn.Module] = nn.Tanh,
#         ortho_init: bool = True,
#         use_sde: bool = False,
#         log_std_init: float = 0.0,
#         full_std: bool = True,
#         use_expln: bool = False,
#         squash_output: bool = False,
#         features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         share_features_extractor: bool = True,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#     ):
#         if optimizer_kwargs is None:
#             optimizer_kwargs = {}
#             # Small values to avoid NaN in Adam optimizer
#             if optimizer_class == th.optim.Adam:
#                 optimizer_kwargs["eps"] = 1e-5

#         super().__init__(
#             observation_space,
#             action_space,
#             features_extractor_class,
#             features_extractor_kwargs,
#             optimizer_class=optimizer_class,
#             optimizer_kwargs=optimizer_kwargs,
#             squash_output=squash_output,
#             normalize_images=normalize_images,
#         )

#         if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
#             warnings.warn(
#                 (
#                     "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
#                     "you should now pass directly a dictionary and not a list "
#                     "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
#                 ),
#             )
#             net_arch = net_arch[0]

#         # Default network architecture, from stable-baselines
#         if net_arch is None:
#             if features_extractor_class == NatureCNN:
#                 net_arch = []
#             else:
#                 net_arch = dict(pi=[64, 64], vf=[64, 64])

#         self.net_arch = net_arch
#         self.activation_fn = activation_fn
#         self.ortho_init = ortho_init

#         self.share_features_extractor = share_features_extractor
#         self.features_extractor = self.make_features_extractor()
#         self.features_dim = self.features_extractor.features_dim
#         if self.share_features_extractor:
#             self.pi_features_extractor = self.features_extractor
#             self.vf_features_extractor = self.features_extractor
#         else:
#             self.pi_features_extractor = self.features_extractor
#             self.vf_features_extractor = self.make_features_extractor()

#         self.log_std_init = log_std_init
#         dist_kwargs = None
#         # Keyword arguments for gSDE distribution
#         if use_sde:
#             dist_kwargs = {
#                 "full_std": full_std,
#                 "squash_output": squash_output,
#                 "use_expln": use_expln,
#                 "learn_features": False,
#             }

#         self.use_sde = use_sde
#         self.dist_kwargs = dist_kwargs

#         # Action distribution
#         self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

#         self._build(lr_schedule)

#     def _get_constructor_parameters(self) -> Dict[str, Any]:
#         data = super()._get_constructor_parameters()

#         default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

#         data.update(
#             dict(
#                 net_arch=self.net_arch,
#                 activation_fn=self.activation_fn,
#                 use_sde=self.use_sde,
#                 log_std_init=self.log_std_init,
#                 squash_output=default_none_kwargs["squash_output"],
#                 full_std=default_none_kwargs["full_std"],
#                 use_expln=default_none_kwargs["use_expln"],
#                 lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
#                 ortho_init=self.ortho_init,
#                 optimizer_class=self.optimizer_class,
#                 optimizer_kwargs=self.optimizer_kwargs,
#                 features_extractor_class=self.features_extractor_class,
#                 features_extractor_kwargs=self.features_extractor_kwargs,
#             )
#         )
#         return data

#     def reset_noise(self, n_envs: int = 1) -> None:
#         """
#         Sample new weights for the exploration matrix.

#         :param n_envs:
#         """
#         assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
#         self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

#     def _build_mlp_extractor(self) -> None:
#         """
#         Create the policy and value networks.
#         Part of the layers can be shared.
#         """
#         # Note: If net_arch is None and some features extractor is used,
#         #       net_arch here is an empty list and mlp_extractor does not
#         #       really contain any layers (acts like an identity module).
#         self.mlp_extractor = MlpExtractor(
#             self.features_dim,
#             net_arch=self.net_arch,
#             activation_fn=self.activation_fn,
#             device=self.device,
#         )

#     def _build(self, lr_schedule: Schedule) -> None:
#         """
#         Create the networks and the optimizer.

#         :param lr_schedule: Learning rate schedule
#             lr_schedule(1) is the initial learning rate
#         """
#         self._build_mlp_extractor()

#         latent_dim_pi = self.mlp_extractor.latent_dim_pi

#         if isinstance(self.action_dist, DiagGaussianDistribution):
#             self.action_net, self.log_std = self.action_dist.proba_distribution_net(
#                 latent_dim=latent_dim_pi, log_std_init=self.log_std_init
#             )
#         elif isinstance(self.action_dist, StateDependentNoiseDistribution):
#             self.action_net, self.log_std = self.action_dist.proba_distribution_net(
#                 latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
#             )
#         elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
#             self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
#         else:
#             raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

#         self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
#         # Init weights: use orthogonal initialization
#         # with small initial weight for the output
#         if self.ortho_init:
#             # TODO: check for features_extractor
#             # Values from stable-baselines.
#             # features_extractor/mlp values are
#             # originally from openai/baselines (default gains/init_scales).
#             module_gains = {
#                 self.features_extractor: np.sqrt(2),
#                 self.mlp_extractor: np.sqrt(2),
#                 self.action_net: 0.01,
#                 self.value_net: 1,
#             }
#             if not self.share_features_extractor:
#                 # Note(antonin): this is to keep SB3 results
#                 # consistent, see GH#1148
#                 del module_gains[self.features_extractor]
#                 module_gains[self.pi_features_extractor] = np.sqrt(2)
#                 module_gains[self.vf_features_extractor] = np.sqrt(2)

#             for module, gain in module_gains.items():
#                 module.apply(partial(self.init_weights, gain=gain))

#         # Setup optimizer with initial learning rate
#         self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

#     def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
#         """
#         Forward pass in all the networks (actor and critic)

#         :param obs: Observation
#         :param deterministic: Whether to sample or use deterministic actions
#         :return: action, value and log probability of the action
#         """
#         # Preprocess the observation if needed
#         features = self.extract_features(obs)
#         if self.share_features_extractor:
#             latent_pi, latent_vf = self.mlp_extractor(features)
#         else:
#             pi_features, vf_features = features
#             latent_pi = self.mlp_extractor.forward_actor(pi_features)
#             latent_vf = self.mlp_extractor.forward_critic(vf_features)
#         # Evaluate the values for the given observations
#         values = self.value_net(latent_vf)
#         distribution = self._get_action_dist_from_latent(latent_pi)
#         actions = distribution.get_actions(deterministic=deterministic)
#         log_prob = distribution.log_prob(actions)
#         actions = actions.reshape((-1, *self.action_space.shape))
#         return actions, values, log_prob

#     def extract_features(self, obs: th.Tensor) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
#         """
#         Preprocess the observation if needed and extract features.

#         :param obs: Observation
#         :return: the output of the features extractor(s)
#         """
#         if self.share_features_extractor:
#             return super().extract_features(obs, self.features_extractor)
#         else:
#             pi_features = super().extract_features(obs, self.pi_features_extractor)
#             vf_features = super().extract_features(obs, self.vf_features_extractor)
#             return pi_features, vf_features

#     def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
#         """
#         Retrieve action distribution given the latent codes.

#         :param latent_pi: Latent code for the actor
#         :return: Action distribution
#         """
#         mean_actions = self.action_net(latent_pi)

#         if isinstance(self.action_dist, DiagGaussianDistribution):
#             return self.action_dist.proba_distribution(mean_actions, self.log_std)
#         elif isinstance(self.action_dist, CategoricalDistribution):
#             # Here mean_actions are the logits before the softmax
#             return self.action_dist.proba_distribution(action_logits=mean_actions)
#         elif isinstance(self.action_dist, MultiCategoricalDistribution):
#             # Here mean_actions are the flattened logits
#             return self.action_dist.proba_distribution(action_logits=mean_actions)
#         elif isinstance(self.action_dist, BernoulliDistribution):
#             # Here mean_actions are the logits (before rounding to get the binary actions)
#             return self.action_dist.proba_distribution(action_logits=mean_actions)
#         elif isinstance(self.action_dist, StateDependentNoiseDistribution):
#             return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
#         else:
#             raise ValueError("Invalid action distribution")

#     def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
#         """
#         Get the action according to the policy for a given observation.

#         :param observation:
#         :param deterministic: Whether to use stochastic or deterministic actions
#         :return: Taken action according to the policy
#         """
#         return self.get_distribution(observation).get_actions(deterministic=deterministic)

#     def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
#         """
#         Evaluate actions according to the current policy,
#         given the observations.

#         :param obs: Observation
#         :param actions: Actions
#         :return: estimated value, log likelihood of taking those actions
#             and entropy of the action distribution.
#         """
#         # Preprocess the observation if needed
#         features = self.extract_features(obs)
#         if self.share_features_extractor:
#             latent_pi, latent_vf = self.mlp_extractor(features)
#         else:
#             pi_features, vf_features = features
#             latent_pi = self.mlp_extractor.forward_actor(pi_features)
#             latent_vf = self.mlp_extractor.forward_critic(vf_features)
#         distribution = self._get_action_dist_from_latent(latent_pi)
#         log_prob = distribution.log_prob(actions)
#         values = self.value_net(latent_vf)
#         entropy = distribution.entropy()
#         return values, log_prob, entropy

#     def get_distribution(self, obs: th.Tensor) -> Distribution:
#         """
#         Get the current policy distribution given the observations.

#         :param obs:
#         :return: the action distribution.
#         """
#         features = super().extract_features(obs, self.pi_features_extractor)
#         latent_pi = self.mlp_extractor.forward_actor(features)
#         return self._get_action_dist_from_latent(latent_pi)

#     def predict_values(self, obs: th.Tensor) -> th.Tensor:
#         """
#         Get the estimated values according to the current policy given the observations.

#         :param obs: Observation
#         :return: the estimated values.
#         """
#         features = super().extract_features(obs, self.vf_features_extractor)
#         latent_vf = self.mlp_extractor.forward_critic(features)
#         return self.value_net(latent_vf)


# class ActorCriticCnnPolicy(ActorCriticPolicy):
#     """
#     CNN policy class for actor-critic algorithms (has both policy and value prediction).
#     Used by A2C, PPO and the likes.

#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param ortho_init: Whether to use or not orthogonal initialization
#     :param use_sde: Whether to use State Dependent Exploration or not
#     :param log_std_init: Initial value for the log standard deviation
#     :param full_std: Whether to use (n_features x n_actions) parameters
#         for the std instead of only (n_features,) when using gSDE
#     :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
#         a positive standard deviation (cf paper). It allows to keep variance
#         above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
#     :param squash_output: Whether to squash the output using a tanh function,
#         this allows to ensure boundaries when using gSDE.
#     :param features_extractor_class: Features extractor to use.
#     :param features_extractor_kwargs: Keyword arguments
#         to pass to the features extractor.
#     :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     """

#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         lr_schedule: Schedule,
#         net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
#         activation_fn: Type[nn.Module] = nn.Tanh,
#         ortho_init: bool = True,
#         use_sde: bool = False,
#         log_std_init: float = 0.0,
#         full_std: bool = True,
#         use_expln: bool = False,
#         squash_output: bool = False,
#         features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         share_features_extractor: bool = True,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             ortho_init,
#             use_sde,
#             log_std_init,
#             full_std,
#             use_expln,
#             squash_output,
#             features_extractor_class,
#             features_extractor_kwargs,
#             share_features_extractor,
#             normalize_images,
#             optimizer_class,
#             optimizer_kwargs,
#         )


# class MultiInputActorCriticPolicy(ActorCriticPolicy):
#     """
#     MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
#     Used by A2C, PPO and the likes.

#     :param observation_space: Observation space (Tuple)
#     :param action_space: Action space
#     :param lr_schedule: Learning rate schedule (could be constant)
#     :param net_arch: The specification of the policy and value networks.
#     :param activation_fn: Activation function
#     :param ortho_init: Whether to use or not orthogonal initialization
#     :param use_sde: Whether to use State Dependent Exploration or not
#     :param log_std_init: Initial value for the log standard deviation
#     :param full_std: Whether to use (n_features x n_actions) parameters
#         for the std instead of only (n_features,) when using gSDE
#     :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
#         a positive standard deviation (cf paper). It allows to keep variance
#         above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
#     :param squash_output: Whether to squash the output using a tanh function,
#         this allows to ensure boundaries when using gSDE.
#     :param features_extractor_class: Uses the CombinedExtractor
#     :param features_extractor_kwargs: Keyword arguments
#         to pass to the features extractor.
#     :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param optimizer_class: The optimizer to use,
#         ``th.optim.Adam`` by default
#     :param optimizer_kwargs: Additional keyword arguments,
#         excluding the learning rate, to pass to the optimizer
#     """

#     def __init__(
#         self,
#         observation_space: spaces.Dict,
#         action_space: spaces.Space,
#         lr_schedule: Schedule,
#         net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
#         activation_fn: Type[nn.Module] = nn.Tanh,
#         ortho_init: bool = True,
#         use_sde: bool = False,
#         log_std_init: float = 0.0,
#         full_std: bool = True,
#         use_expln: bool = False,
#         squash_output: bool = False,
#         features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
#         features_extractor_kwargs: Optional[Dict[str, Any]] = None,
#         share_features_extractor: bool = True,
#         normalize_images: bool = True,
#         optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
#         optimizer_kwargs: Optional[Dict[str, Any]] = None,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             net_arch,
#             activation_fn,
#             ortho_init,
#             use_sde,
#             log_std_init,
#             full_std,
#             use_expln,
#             squash_output,
#             features_extractor_class,
#             features_extractor_kwargs,
#             share_features_extractor,
#             normalize_images,
#             optimizer_class,
#             optimizer_kwargs,
#         )


# class ContinuousCritic(BaseModel):
#     """
#     Critic network(s) for DDPG/SAC/TD3.
#     It represents the action-state value function (Q-value function).
#     Compared to A2C/PPO critics, this one represents the Q-value
#     and takes the continuous action as input. It is concatenated with the state
#     and then fed to the network which outputs a single value: Q(s, a).
#     For more recent algorithms like SAC/TD3, multiple networks
#     are created to give different estimates.

#     By default, it creates two critic networks used to reduce overestimation
#     thanks to clipped Q-learning (cf TD3 paper).

#     :param observation_space: Obervation space
#     :param action_space: Action space
#     :param net_arch: Network architecture
#     :param features_extractor: Network to extract features
#         (a CNN when using images, a nn.Flatten() layer otherwise)
#     :param features_dim: Number of features
#     :param activation_fn: Activation function
#     :param normalize_images: Whether to normalize images or not,
#          dividing by 255.0 (True by default)
#     :param n_critics: Number of critic networks to create.
#     :param share_features_extractor: Whether the features extractor is shared or not
#         between the actor and the critic (this saves computation time)
#     """

#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Box,
#         net_arch: List[int],
#         features_extractor: BaseFeaturesExtractor,
#         features_dim: int,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         normalize_images: bool = True,
#         n_critics: int = 2,
#         share_features_extractor: bool = True,
#     ):
#         super().__init__(
#             observation_space,
#             action_space,
#             features_extractor=features_extractor,
#             normalize_images=normalize_images,
#         )

#         action_dim = get_action_dim(self.action_space)

#         self.share_features_extractor = share_features_extractor
#         self.n_critics = n_critics
#         self.q_networks = []
#         for idx in range(n_critics):
#             q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
#             q_net = nn.Sequential(*q_net)
#             self.add_module(f"qf{idx}", q_net)
#             self.q_networks.append(q_net)

#     def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
#         # Learn the features extractor using the policy loss only
#         # when the features_extractor is shared with the actor
#         with th.set_grad_enabled(not self.share_features_extractor):
#             features = self.extract_features(obs, self.features_extractor)
#         qvalue_input = th.cat([features, actions], dim=1)
#         return tuple(q_net(qvalue_input) for q_net in self.q_networks)

#     def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
#         """
#         Only predict the Q-value using the first network.
#         This allows to reduce computation when all the estimates are not needed
#         (e.g. when updating the policy in TD3).
#         """
#         with th.no_grad():
#             features = self.extract_features(obs, self.features_extractor)
#         return self.q_networks[0](th.cat([features, actions], dim=1))

class PPO(OnPolicyAlgorithm):
    
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        # "CnnPolicy": ActorCriticCnnPolicy,
        # "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            )

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            )
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"The buffer_size is not a multiple of the batch_size"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
        ####################################################################
        #New with CUDA
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        self.policy.to(self.device)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                ####################################################################
                # Old
                #actions = rollout_data.actions
                # New with CUDA
                observations = rollout_data.observations.to(self.device)
                actions = rollout_data.actions.to(self.device)
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)
                ####################################################################
                # Old
                #values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                # New with CUDA
                values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
                values = values.flatten()
                # Normalize advantage
                ####################################################################
                # Old
                #advantages = rollout_data.advantages
                # New with CUDA
                advantages = rollout_data.advantages.to(self.device)
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break
        # explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # # Logs
        # self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        # self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        # self.logger.record("train/value_loss", np.mean(value_losses))
        # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        # self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        # self.logger.record("train/loss", loss.item())
        # self.logger.record("train/explained_variance", explained_var)
        # if hasattr(self.policy, "log_std"):
        #     self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.record("train/clip_range", clip_range)
        # if self.clip_range_vf is not None:
        #     self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )




