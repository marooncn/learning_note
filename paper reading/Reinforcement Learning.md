## Reinforcement Learning
### Algorithm
[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)(NIPS 2013, Deep Q-learning with Experience Replay ) <br>
[DeterministicPolicyGradientAlgorithms](http://proceedings.mlr.press/v32/silver14.pdf)(ICML 2014, DPG) <br>
[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236.pdf)(Nature 2015, traget Q-network)<br>
[Deep Reinforcement Learning with Double Q-learning](https://pdfs.semanticscholar.org/3b97/32bb07dc99bde5e1f9f75251c6ea5039373e.pdf)(AAAI 2016, Double DQN) <br>
[Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)(ICLR 2016, Prioritized replay)<br>
[Hindsight Experience Replay](https://arxiv.org/pdf/1707.01495v1.pdf)(arxiv 2017, HER) <br>
[Dueling Network Architectures for Deep Reinforcement](https://arxiv.org/pdf/1511.06581.pdf)(ICML 2016, Dueling DQN)<br>
[Mastering the game of Go with deep neural networks and tree search](http://www.cs.cmu.edu/afs/cs/academic/class/15780-s16/www/AlphaGo.nature16961.pdf)(Nature 2016, AlphaGo) <br>
[Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)(ICLR 2016, DDPG)<br>
[Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/pdf/1603.00748.pdf)([blog](https://blog.csdn.net/u013236946/article/details/73243310/) & [Zhihu](https://zhuanlan.zhihu.com/p/21609472), DeepMInd 2016, NAF)<br>
[Asynchronous Methods for Deep Reinforcement Learning ](https://arxiv.org/pdf/1602.01783.pdf)(ICML 2016, A3C)<br>
[Reinforcement Learning thorugh Asynchronous Advantage Actor-Critic on a GPU](https://openreview.net/forum?id=r1VGvBcxl&noteId=r1VGvBcxl)(ICLR 2017, GA3C) <br>
[Generative Adversial Imitation Learning](http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning)(NIPS 2016, GAIL) <br>
[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)(arxiv 2017, OpenAI PPO)<br>
[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf) (arxiv 2017, DeepMind PPO)<br>
[Reinforcement learning with Deep Energy-Based Polices](https://arxiv.org/pdf/1702.08165.pdf)([blog](https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/), ICML 2017, Soft Q-learning) <br>
[A Distributional Perspective on Reinforcement Learning](https://deepmind.com/blog/going-beyond-average-reinforcement-learning/)(ICML 2017, Distributional RL, C51) <br>
[Meta-Learning Shared Hierarchies](https://arxiv.org/pdf/1710.09767.pdf)([blog](https://blog.openai.com/learning-a-hierarchy/), OpenAI 2017, Hierarchical RL) <br>
[Rainbow: Combining improvements in deep reinforcement learning](https://arxiv.org/pdf/1710.02298.pdf)(AAAI 2018, Rainbow)<br>
[Multi-task Deep Reinforcement Learning with PopArt](https://deepmind.com/blog/preserving-outputs-precisely-while-adaptively-rescaling-targets/)(PopArt, train a single agent that can play a whole set of 57 diverse Atari video games with reward signal normalization) <br>
[Neural scene representation and rendering](http://science.sciencemag.org/content/360/6394/1204/tab-pdf)([blog](https://deepmind.com/blog/neural-scene-representation-and-rendering/), Science 2018, Generative Query Network (GQN)) <br>
[World Models](https://arxiv.org/pdf/1803.10122.pdf)([blog](https://dylandjian.github.io/world-models/), NIPS 2018, World Models=Vison model([VAE](http://kvfrans.com/variational-autoencoders-explained/))+Memory(RNN+[MDN](http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/))+Compact Controller(CMA-ES) <br>

### Meta-Learning
[Learning to reinforcement learn](https://arxiv.org/pdf/1611.05763.pdf)(DeepMind 2017) <br>
[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)(ICML 2017, MAML) <br>
[Reptile: A Scalable Meta-Learning Algorithm](https://blog.openai.com/reptile/)(OpenAI 2018, Retile)


### Curiosity & Exploration & Reward Shaping
#### With sparse external reward
[Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/pdf/1611.05397.pdf)(DeepMind 2016, UNREAL) <br>
[Curiosity-driven exploration by self-supervised prediction](https://arxiv.org/pdf/1705.05363.pdf)(ICML 2017, Intrinsic Curiosity Module) <br>
[Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf)([blog](https://blog.openai.com/reinforcement-learning-with-prediction-based-rewards/#RNDjump), RND, exceed average human performance on Montezuma’s Revenge) 
#### Even without external reward
[Apprenticeship learning via Inverse Reinforcement Learning](http://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf)(ICML 2004, Inverse Imitation Learning) <br>
[Deep reinforcement learning from human preferences](https://arxiv.org/pdf/1706.03741.pdf)([blog](https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/), arxiv 2017, Just need 900 bits of feedback from a human evaluator to learn to backflip — a seemingly simple task which is simple to judge but challenging to specify.) <br>
[Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play](https://cims.nyu.edu/~sainbar/selfplay/)(ICLR 2018, Self-play: Alice and Bob) <br>
[Large-Scale Study of Curiosity-Driven Learning](https://pathak22.github.io/large-scale-curiosity/)(OpenAI 2018, Curiosity-driven Learning)  <br>


### Reality Gap
[Sim-to-Real: Learning Agile Locomotion For Quadruped Robots](https://arxiv.org/pdf/1804.10332.pdf)(arxiv 2018, Google, "We  narrow  this  reality  gap  by  improving  the  physics simulator and learning robust policies.") <br>
[Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://xbpeng.github.io/projects/SimToReal/2018_SimToReal.pdf)(ICRA 2018, "By randomizing the dynamics of the simulator during training, we are able to develop policies that are capable of adapting to very different dynamics") <br>

### Benchmark
[gym](https://gym.openai.com/envs/)(OpenAI, big gays) <br>
[Roboschool](https://blog.openai.com/roboschool/)(OpenAI 2017) <br>
[gym Retro](https://blog.openai.com/gym-retro/)(OpenAI, game platform) <br>
[Retro Contest](https://blog.openai.com/retro-contest/)(a transfer learning contest for generalization test, [contest result](https://blog.openai.com/first-retro-contest-retrospective/)) <br>
[CoinRun](https://blog.openai.com/quantifying-generalization-in-reinforcement-learning/)(OpenAI 2018, provide a metric for an agent’s ability to transfer its experience to novel situations) <br>
[DeepMind Lab](https://github.com/deepmind/lab)(DeepMind 2016, first-person 3D game platform) <br>
[Control Suite](https://github.com/deepmind/dm_control)(DeepMind 2018) <br>
[Unity](https://unity3d.com/machine-learning/) <br>
[pybullet](https://pypi.org/project/pybullet/) 
#### a comparision from [AI2-THOR](http://ai2thor.allenai.org/) 
<img alt="simulation framework" src="https://github.com/marooncn/learning_note/blob/master/paper%20reading/image/simulation%20framework.png" width="400"> <br>

### Implementations
[OpenAI Baselines](https://github.com/openai/baselines)(OpenAI) <br>
[rllab](https://rllab.readthedocs.io/en/latest/index.html#)(Berkeley) <br>
[RLlib](https://ray.readthedocs.io/en/latest/rllib.html)(Berkeley, multi-agent) <br>
[Horizon](https://github.com/facebookresearch/Horizon)(Facebook)  <br>
[TensorForce](https://github.com/reinforceio/tensorforce)(reinforce.io)  <br>
[Dopamine](https://github.com/google/dopamine)(Google) <br>
[Coach](https://github.com/NervanaSystems/coach)(Intel)  <br>

### Manipulation
[Reinforcement and Imitation Learning for Diverse Visuomotor Skills](https://arxiv.org/pdf/1802.09564.pdf)([blog](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650739530&idx=4&sn=4b08a6f9253473da9ae2396ca78fae05&chksm=871ad734b06d5e22256df7c21d56d3ecace4170866f7868fcd252daf2848b7f56a890bc04c78&mpshare=1&scene=1&srcid=0319nOHWy1wXHfKTqEm3wXgd&pass_ticket=NFFgROZS%2B2E12ics9enIgh0g9UP35ouHDe07%2FZfe5koayvAfbE5TgsaUjUrsXXLV#rd),DeepMind 2018, few demostrations+PPO+LSTM+GAIL) <br>
[Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research](https://arxiv.org/abs/1802.09464)([blog](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650738356&idx=2&sn=381fdd93dc1858580143ee2dff9cf304&chksm=871acacab06d43dcee1d41ff6a8c7b00c1709928a2c1c4f3c252611f6d079b13cdb4339e1fd4&mpshare=1&scene=1&srcid=0309hQxWBV8tlgG779mgswab&pass_ticket=NFFgROZS%2B2E12ics9enIgh0g9UP35ouHDe07%2FZfe5koayvAfbE5TgsaUjUrsXXLV#rd), OpenAI 2018, DDPG + HER with sparse rewards) <br> 
[Composable Deep Reinforcement Learning for Robotic Manipulation](https://lanl.arxiv.org/pdf/1803.06773v1)([blog](https://mp.weixin.qq.com/s?__biz=MzAxMzc2NDAxOQ==&mid=2650366547&idx=1&sn=5c75e5362e2951e68f0b2921e854f7db&chksm=8390568fb4e7df99601064e1bd6493250dcf9e058e2e64cd65e3e63f5c4843bae5b91583ce82&mpshare=1&scene=1&srcid=041005bm6qlvLmW563fbESQG&pass_ticket=NFFgROZS%2B2E12ics9enIgh0g9UP35ouHDe07%2FZfe5koayvAfbE5TgsaUjUrsXXLV#rd), Berkeley 2018, two strenghts of Soft Q-learning: multimodal exploration; composed) <br> 
[One-Shot Visual Imitation Learning via Meta-Learning](https://arxiv.org/pdf/1709.04905.pdf)(CoRL 2017, combine imitation learning with MAML)  <br> 
[One-Shot Imitation from Observing Humans via Domain-Adaptive Meta-Learning](https://arxiv.org/pdf/1802.01557.pdf)([blog](https://bair.berkeley.edu/blog/2018/06/28/daml/), RSS 2018, One-Shot Imitation from Watching Videos without labeled expert actions) <br>
[Grasp2Vec: Learning Object Representations from Self-Supervised Grasping](https://arxiv.org/pdf/1811.06964.pdf)(CoRL 2018, Google Brain) <br>

### Character Skills
[DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/index.html)([blog](https://bair.berkeley.edu/blog/2018/04/10/virtual-stuntman/), ACM Transactions on Graphic 2018, Reference State Initialization (RSI)+Early Termination (ET)) <br> 
[SFV: Reinforcement Learning of Physical Skills from Videos](https://xbpeng.github.io/projects/SFV/index.html)([blog](https://bair.berkeley.edu/blog/2018/10/09/sfv/)， ACM Transactions on Graphic 2018) <br>

### Computer Vision
[Active Object Localization with Deep Reinforcement Learning](http://slazebni.cs.illinois.edu/publications/iccv15_active.pdf)(ICCV 2015)<br>
[Hierarchical Object Detection with Deep Reinforcement Learning](https://imatge-upc.github.io/detection-2016-nipsws/)(NIPS 2016) <br>
[Crafting a Toolchain for Image Restoration by Deep Reinforcement Learning](http://mmlab.ie.cuhk.edu.hk/projects/RL-Restore/)(CVPR 2018) <br>

### Doom
[ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning](https://arxiv.org/pdf/1605.02097.pdf) <br>
[Playing doom with slam-augmented deep reinforcement learning](https://arxiv.org/pdf/1612.00380.pdf)(CVPR 2016) <br>
[Training Agent for First-Person Shooter Game with Actor-Critic Curriculum Learning](https://openreview.net/pdf?id=Hk3mPK5gg)(ICLR 2017, VIZDoom2016 Track1冠军) <br>
[Learning to Act by Predicting the Future](https://arxiv.org/pdf/1611.01779.pdf)(ICLR 2017, VIZDoom2016 Track2冠军) <br>
[Playing FPS Games with Deep Reinforcement Learning](https://arxiv.org/pdf/1609.05521.pdf)(AAAI 2017, VIZDoom2017冠军) <br>


### Video
[Neural Adaptive Video Streaming with Pensieve](http://web.mit.edu/pensieve/)(ACM 2017) <br>

### Good Blog & Tutorial
[Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)(Policy Gradient) <br>
[Using Keras and Deep Q-Network to Play FlappyBird](https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html)(DQN) <br>
[Build an AI to play Dino Run](https://blog.paperspace.com/dino-run/)(DQN) <br>
[Using Deep Q-Learning in FIFA 18 to perfect the art of free-kicks](https://towardsdatascience.com/using-deep-q-learning-in-fifa-18-to-perfect-the-art-of-free-kicks-f2e4e979ee66)(DQN)<br>
[Using Keras and Deep Deterministic Policy Gradient to play TORCS](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html)(DDPG) <br>
[Self-driving cars in the browser](https://janhuenermann.com/blog/learning-to-drive)(DDPG) <br>
[Use proximal policy optimization to play BipedalWalker and Torcs](https://junhongxu.github.io/JunhongXu.github.io/Proximal-Policy-Optimization/)(PPO) <br>
[复现PPO](https://zhuanlan.zhihu.com/p/50322028) <br>
[Simple Reinforcement Learning with Tensorflow Part 8](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)(A3C) <br>
[Reinforcement learning with the A3C algorithm](https://cgnicholls.github.io/reinforcement-learning/2017/03/27/a3c.html)(A3C) <br>
[A3C Blog Post](https://github.com/tensorflow/models/tree/a2a943da2635bfe93cd0c17a1d186f1f3235126c/research/a3c_blogpost) <br>
[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) <br>
[李宏毅：Deep Reinforcement Learning](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/RL%20(v6).pdf) <br>
[DeepMind - Deep Reinforcement Learning - RLSS 2017.pdf](https://drive.google.com/file/d/0BzUSSMdMszk6UE5TbWdZekFXSE0/view)<br>
[OpenAI Spinning up](https://spinningup.openai.com/en/latest/)  <br>
[AlphaGo Zero demystified](https://dylandjian.github.io/alphago-zero/) <br>
[World Models applied to Sonic](https://dylandjian.github.io/world-models/) <br>

### Rethink
[Deep reinforcement learning that matters](https://arxiv.org/pdf/1709.06560.pdf) <br>
[Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html) <br>
[Reinforcement Learning never worked, and 'deep' only helped a bit.](https://himanshusahni.github.io/2018/02/23/reinforcement-learning-never-worked.html) <br>
[Lessons Learned Reproducing a Deep Reinforcement Learning Paper](http://amid.fish/reproducing-deep-rl) <br>
[强化学习路在何方？](https://zhuanlan.zhihu.com/p/39999667) <br>

### Evolution Strategy
[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/pdf/1703.03864.pdf)([blog](https://blog.openai.com/evolution-strategies/), OpenAI 2017, ES，advantages of not calculating gradients/ easy to parallelize/more robust(such as frame-skip)) <br>
[A Visual Guide to Evolution Strategies](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/) <br>
[Evolving Stable Strategies](http://blog.otoro.net/2017/11/12/evolving-stable-strategies/)(ES on robot; task augmentation techniques) <br>


### Overview
[A Survey of Deep Network Solutions for Learning Control in Robotics: From Reinforcement to Imitation](https://arxiv.org/pdf/1612.07139.pdf) <br>
[Multi-Agent Reinforcement Learning: A Report on Challenges and Approaches](https://arxiv.org/pdf/1807.09427.pdf) <br>
