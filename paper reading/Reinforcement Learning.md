## Reinforcement Learning
### Algorithm
[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)(NIPS 2013, Deep Q-learning with Experience Replay ) <br>
[Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)(ICML 2014, DPG) <br>
[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236.pdf)(Nature 2015, traget Q-network)<br>
[Deep Reinforcement Learning with Double Q-learning](https://pdfs.semanticscholar.org/3b97/32bb07dc99bde5e1f9f75251c6ea5039373e.pdf)(AAAI 2016, Double DQN) <br>
[Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)(ICLR 2016, Prioritized replay)<br>
[Hindsight Experience Replay](https://arxiv.org/pdf/1707.01495v1.pdf)(arxiv 2017, HER) <br>
[Dueling Network Architectures for Deep Reinforcement](https://arxiv.org/pdf/1511.06581.pdf)(ICML 2016, Dueling DQN)<br>
[Mastering the game of Go with deep neural networks and tree search](http://www.cs.cmu.edu/afs/cs/academic/class/15780-s16/www/AlphaGo.nature16961.pdf)(Nature 2016, AlphaGo) <br>
[Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)(ICLR 2016, DDPG)<br>
[Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/pdf/1603.00748.pdf)([blog](https://blog.csdn.net/u013236946/article/details/73243310/) & [Zhihu](https://zhuanlan.zhihu.com/p/21609472), DeepMind 2016, NAF)<br>
[Asynchronous Methods for Deep Reinforcement Learning ](https://arxiv.org/pdf/1602.01783.pdf)(ICML 2016, A3C)<br>
[Reinforcement Learning thorugh Asynchronous Advantage Actor-Critic on a GPU](https://openreview.net/forum?id=r1VGvBcxl&noteId=r1VGvBcxl)(ICLR 2017, GA3C) <br>
[Generative Adversial Imitation Learning](http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning)(NIPS 2016, GAIL) <br>
[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)(arxiv 2017, OpenAI PPO)<br>
[Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf) (arxiv 2017, DeepMind PPO)<br>
[Reinforcement learning with Deep Energy-Based Polices](https://arxiv.org/pdf/1702.08165.pdf)([blog](https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/), ICML 2017, Soft Q-learning) <br>
[Mastering the game of Go without human knowledge](http://web.iitd.ac.in/~sumeet/Silver16.pdf)(AlphaGo zero, Nature 2017) <br>
[Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)(arxiv 2018, Soft Actor-Critic) <br>
[A Distributional Perspective on Reinforcement Learning](https://deepmind.com/blog/going-beyond-average-reinforcement-learning/)(ICML 2017, Distributional RL, C51) <br>
[Meta-Learning Shared Hierarchies](https://arxiv.org/pdf/1710.09767.pdf)([blog](https://blog.openai.com/learning-a-hierarchy/), OpenAI 2017, Hierarchical RL) <br>
[Rainbow: Combining improvements in deep reinforcement learning](https://arxiv.org/pdf/1710.02298.pdf)(AAAI 2018, Rainbow)<br>
[Multi-task Deep Reinforcement Learning with PopArt](https://deepmind.com/blog/preserving-outputs-precisely-while-adaptively-rescaling-targets/)(PopArt, train a single agent that can play a whole set of 57 diverse Atari video games with reward signal normalization) <br>
[Neural scene representation and rendering](http://science.sciencemag.org/content/360/6394/1204/tab-pdf)([blog](https://deepmind.com/blog/neural-scene-representation-and-rendering/), Science 2018, Generative Query Network (GQN)) <br>
[World Models](https://arxiv.org/pdf/1803.10122.pdf)([blog](https://dylandjian.github.io/world-models/), NIPS 2018, World Models=Vison model([VAE](http://kvfrans.com/variational-autoencoders-explained/))+Memory(RNN+[MDN](http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/))+Compact Controller(CMA-ES), the first known agent to solve OpenAi Gym Race Car) <br>
[Reinforcement Learning for Improving Agent Design](https://designrl.github.io/)( Joint learning of policy and structure, Google 2018)<br>
[A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Applications_files/alphazero-science.pdf)(AlphaZero, Science 2018) <br>
[Learning Latent Dynamics for Planning from Pixels](https://planetrl.github.io/)(PlaNet, Google 2019) <br>
[Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/pdf/1911.08265.pdf)(MuZero, DeepMind 2019) <br>

### Meta-Learning
[Learning to reinforcement learn](https://arxiv.org/pdf/1611.05763.pdf)(DeepMind 2017) <br>
[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)(ICML 2017, MAML) <br>
[Reptile: A Scalable Meta-Learning Algorithm](https://blog.openai.com/reptile/)(OpenAI 2018, Retile)

### Curriculum learning
[Curriculum learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf)(ACM 2009, Curriculum learning gradually increases the complexity of the learning task by choosing more and more difficult examples for the learning algorith) <br>
[Automated Curriculum Learning for Neural Networks](http://proceedings.mlr.press/v70/graves17a/graves17a.pdf) <br>


### Curiosity & Exploration & Reward Shaping
#### With sparse external reward
[Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/pdf/1611.05397.pdf)(DeepMind 2016, UNREAL) <br>
[Curiosity-driven exploration by self-supervised prediction](https://arxiv.org/pdf/1705.05363.pdf)(ICML 2017, Intrinsic Curiosity Module) <br>
[Exploration by Random Network Distillation](https://arxiv.org/pdf/1810.12894.pdf)([blog](https://blog.openai.com/reinforcement-learning-with-prediction-based-rewards/#RNDjump), RND, exceed average human performance on Montezuma’s Revenge) 
[Episodic curiosity through reachability](https://arxiv.org/pdf/1810.02274.pdf)([code](https://github.com/google-research/episodic-curiosity) & [blog](https://towardsdatascience.com/whats-new-in-deep-learning-research-how-google-builds-curiosity-into-reinforcement-learning-32d77af719e8), Google Brain & DeepMind 2019, maximize curiosity only if is conducive to the ultimate goal)
#### Even without external reward
[Apprenticeship learning via Inverse Reinforcement Learning](http://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf)(ICML 2004, Inverse Imitation Learning) <br>
[Deep reinforcement learning from human preferences](https://arxiv.org/pdf/1706.03741.pdf)([blog](https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/), arxiv 2017, Just need 900 bits of feedback from a human evaluator to learn to backflip — a seemingly simple task which is simple to judge but challenging to specify.) <br>
[Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play](https://cims.nyu.edu/~sainbar/selfplay/)(ICLR 2018, Self-play: Alice and Bob) <br>
[Large-Scale Study of Curiosity-Driven Learning](https://github.com/marooncn/learning_note/blob/master/paper%20reading/notes/Large-Scale%20Study%20of%20Curiosity-Driven%20Learning.pdf)([website](https://pathak22.github.io/large-scale-curiosity/), OpenAI 2018, "More generally, these results suggest that, in environments designed by humans, the extrinsic reward
is perhaps often aligned with the objective of seeking novelty.")  <br>
[End-to-End Robotic Reinforcement Learning without Reward Engineering](https://arxiv.org/pdf/1904.07854.pdf)([code](https://github.com/avisingh599/reward-learning-rl), RSS 2019, Berkeley, using successful outcome images to train a success classifier, then use log-probabilities obtained from the success classifier as reward for running reinforcement learning and actively query the human user to optimize the success classifier) <br>
#### Others
[Visceral Machines: Risk-Aversion in Reinforcement Learning with Intrinsic Physiological Rewards](https://arxiv.org/pdf/1805.09975v2.pdf)(ICLR 2019, just train a CNN to predict response as intrinsic reward in navigation task) <br>

### Reality Gap 
[Sim-to-Real: Learning Agile Locomotion For Quadruped Robots](https://arxiv.org/pdf/1804.10332.pdf)(arxiv 2018, Google, "We  narrow  this  reality  gap  by  improving  the  physics simulator and learning robust policies.") <br>
[Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://xbpeng.github.io/projects/SimToReal/2018_SimToReal.pdf)(ICRA 2018, "By randomizing the dynamics of the simulator during training, we are able to develop policies that are capable of adapting to very different dynamics".) <br>
[Solving Rubik’s Cube with a Robot Hand](https://openai.com/blog/solving-rubiks-cube/)(OpenAI 2019, "we developed a new method called Automatic Domain Randomization (ADR), which endlessly generates progressively more difficult environments in simulation. This frees us from having an accurate model of the real world, and enables the transfer of neural networks learned in simulation to be applied to the real world.") <br>
[Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience](https://arxiv.org/pdf/1810.05687.pdf)(ICRA 2019, "Rather than manually tuning the randomization of simulations, we adapt the simulation parameter distribution using a few real world roll-outs interleaved with policy training") <br>

### Multi-Agent
[Human-level performance in 3D multiplayer games with population-based reinforcement learning](https://deepmind.com/blog/article/capture-the-flag-science)(multiplayer FPS game, DeepMind, Science 2019) <br>

### Other issue
#### Discrete-Continuous Hybrid Action Spaces
[Deep Multi-Agent Reinforcement Learning with Discrete-Continuous Hybrid Action Spaces](https://arxiv.org/pdf/1903.04959.pdf)(IJCAI 2019) <br>

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
[Pommerman](https://www.pommerman.com/)(Multi-Agent "Bomberman"-like game) <br>
[football](https://github.com/google-research/football)(Google 2019) <br>
[ROBEL](www.roboticsbenchmarks.org)(Google 2019, ROBEL is an open-source platform of cost-effective robots designedfor reinforcement learning in the real world)  <br>

### Implementations
[OpenAI Baselines](https://github.com/openai/baselines)(OpenAI) <br>
[keras-rl](https://github.com/keras-rl/keras-rl])(keras) <br>
[rllab](https://rllab.readthedocs.io/en/latest/index.html#)(Berkeley) <br>
[RLlib](https://ray.readthedocs.io/en/latest/rllib.html)(Berkeley, multi-agent) <br>
[Horizon](https://github.com/facebookresearch/Horizon)(Facebook)  <br>
[TensorForce](https://github.com/reinforceio/tensorforce)(reinforce.io)  <br>
[Dopamine](https://github.com/google/dopamine)(Google) <br>
[Coach](https://github.com/NervanaSystems/coach)(Intel)  <br>
[rlkit](https://github.com/vitchyr/rlkit)(personal) <br>
[TRFL](https://github.com/deepmind/trfl)(DeepMind) <br>
[Catalyst.RL](https://github.com/catalyst-team/catalyst-rl-framework)(catalyst-team) <br>
<img alt="RL framework" src="https://github.com/marooncn/learning_note/blob/master/paper%20reading/image/RL%20framework.PNG" width="600"> <br>

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
[Emergence of exploratory look-around behaviors through active observation completion](https://robotics.sciencemag.org/content/4/30/eaaw6326/tab-pdf)(Science Robotics 2019) <br>

### Doom
[ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning](https://arxiv.org/pdf/1605.02097.pdf) <br>
[Playing doom with slam-augmented deep reinforcement learning](https://arxiv.org/pdf/1612.00380.pdf)(CVPR 2016) <br>
[Training Agent for First-Person Shooter Game with Actor-Critic Curriculum Learning](https://openreview.net/pdf?id=Hk3mPK5gg)(ICLR 2017, VIZDoom2016 Track1冠军) <br>
[Learning to Act by Predicting the Future](https://arxiv.org/pdf/1611.01779.pdf)(ICLR 2017, VIZDoom2016 Track2冠军) <br>
[Playing FPS Games with Deep Reinforcement Learning](https://arxiv.org/pdf/1609.05521.pdf)(AAAI 2017, VIZDoom2017冠军) <br>


### Video
[Neural Adaptive Video Streaming with Pensieve](http://web.mit.edu/pensieve/)(ACM 2017) <br>

### Legged locomotion
[Feedback Control For Cassie With Deep Reinforcement Learning](https://arxiv.org/pdf/1803.05580.pdf)(IROS 2018) <br>
[Learning agile and dynamic motor skills for legged robots](https://arxiv.org/pdf/1901.08652.pdf)(Science Robotics 2019, ETH. train >2000 ANYmals in real time in simulation platform together; train a NN representing the complex dynamics with data from the real robot, so the trained policy can be directly deployed on the real system without any modification) <br>
[Iterative Reinforcement Learning Based Design of Dynamic Locomotion Skills for Cassie](https://arxiv.org/pdf/1903.09537.pdf)(arxiv 2019) <br>
[learning to adapt in dynamic, real-world environments through meta-reinforcement learning](https://arxiv.org/pdf/1803.11347.pdf)(ICLR 2019, Berkeley, use meta-learning to train a dynamics model prior such that, when combined with recent data, this prior can be rapidly adapted to the local context) <br>

### Perception
[Manipulation by Feel: Touch-Based Control with Deep Predictive Models](https://arxiv.org/pdf/1903.04128.pdf)(arxiv 2019, Berkeley, Haptic sensor)  <br>
[Motion Perception in Reinforcement Learning with Dynamic Objects](https://lmb.informatik.uni-freiburg.de/projects/flowrl/)(arxiv 2019, image + flow rather than stacked images to include motion information) <br>
[Making  Sense  of  Vision  and  Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks](https://arxiv.org/pdf/1810.10191.pdf)(ICRA 2019) <br>


### Others
[Hacking Google reCAPTCHA v3 using Reinforcement Learning](https://arxiv.org/pdf/1903.01003.pdf)(arxiv 2019, Password cracking) <br>

### Blog 
#### Application
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
[AlphaGo Zero demystified](https://dylandjian.github.io/alphago-zero/) <br>
[World Models applied to Sonic](https://dylandjian.github.io/world-models/) <br>
#### Tutorial
[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) <br>
[OpenAI Spinning up](https://spinningup.openai.com/en/latest/)  <br>
[李宏毅：Deep Reinforcement Learning](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/RL%20(v6).pdf) <br>
[CMU 10703:  Deep Reinforcement Learning and Control ](https://katefvision.github.io/) <br>
#### Overview
[DeepMind - Deep Reinforcement Learning - RLSS 2017.pdf](https://drive.google.com/file/d/0BzUSSMdMszk6UE5TbWdZekFXSE0/view)<br>
[A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html) <br>
[Meta-Learning: Learning to Learn Fast](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)(Metric-based: Convolutional Siamese Neural Network/Matching Networks/Relation Network; Model-based:Memory-Augmented Neural Networks(MANN); Optimization-Based:Model-Agnostic Meta-Learning(MAML)/Reptile) <br>
#### Rethink
[Deep reinforcement learning that matters](https://arxiv.org/pdf/1709.06560.pdf) <br>
[Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html) <br>
[Reinforcement Learning never worked, and 'deep' only helped a bit.](https://himanshusahni.github.io/2018/02/23/reinforcement-learning-never-worked.html) <br>
[Lessons Learned Reproducing a Deep Reinforcement Learning Paper](http://amid.fish/reproducing-deep-rl)([notes](https://github.com/marooncn/learning_note/blob/master/paper%20reading/notes/Lessons%20Learned%20Reproducing%20a%20Deep%20Reinforcement%20Learning%20Paper.pdf)) <br>
[强化学习路在何方？](https://zhuanlan.zhihu.com/p/39999667) <br>
[Reinforcement Learning, Fast and Slow](https://pdf.sciencedirectassets.com/271877/1-s2.0-S1364661318X00060/1-s2.0-S1364661319300610/main.pdf?x-amz-security-token=AgoJb3JpZ2luX2VjEEYaCXVzLWVhc3QtMSJGMEQCIBXJCHYsoc%2F0%2BBxhOJYLsqZQjZRyPA%2FnqiGk%2FZQM4uyZAiBDb%2BEY8u%2BuUaiExsnlfzmHajhkXhXCMDDyBkEstZk4yyraAwhfEAIaDDA1OTAwMzU0Njg2NSIMDHSwkNOHbYxO01ezKrcDol8PC8fqoYFdjIXXV%2BrXKUbDyNgOZUfeO5%2Fpxb28olEtjwBesSfiFL9kae79Ta5ztr8mewuJ4Vy1j1QYJ%2F5MBduuWePyGjuhUj6sJs%2F2zWUETG64RbJEKIyt5lT128NIAR03ikQZsi1SsyI%2BsPRnR6P6TfR7%2B0%2FS06eEdYm0KdUr42%2Fa5EvvK2FO750vXgJsdR9cj%2F24jn%2BdtWli0s7awtI9%2B8SQErvISz0dVZCCdneQpVxW5weByf0LTPrUW%2FN7x%2FYRfNMbpmn%2Bj0fV4O1%2FGoq27LHmJQ3CsWXJsLLn8nMNXXTWELWwJI%2FzJ0UG5KL4p%2Fa%2FJepXmjbbQ5u27HEd8Nrv9QAX6NBMkY7xcWg5mAxON7%2FmtN%2FNC%2FsGkTpxRc7ogra5zkh7ymB4vZs3lSCqfOdm7TvE44EmvR7oE44GKFDejE8elEdot0ADxkD%2B%2Fz4yhu%2BhWzJ65OM%2BtVu7eXbHlUh1KctojYaRJQU1pfWFaZrzzyM6J9jKYKzctjEwBHXINEWz79kR%2F4WYJTMTWKMIYzEXjwMVrCG4UjRstDVHLxgJfAwZvX28GfY%2Fuyp8iq3WwqwkOvc%2BYjC3mbrnBTq1AX1JtAJmH2E1gkKHm9u3iUDAqD2iAsWAfbPgG%2BLSoBEeYM09lnzduAEhKRmNvlDHl5Yt2f9hWyrv0znj39HCelCkIquiRV17c9maRRORMZG3tOmtn15MvJTpBj1AfVm6BPA0RIPky%2Fd6YoL45cSIMr5rIxh1ONRIRhiEZNjOzf8VWMXHwy78yWNrVA0do1UO%2BHpKfFJfKj70dfZJkLtboZV9%2BsQY%2FyMe%2BS9DC6ii1Nho53UHHP8%3D&AWSAccessKeyId=ASIAQ3PHCVTY6GFHSJ5D&Expires=1559140946&Signature=KNc01ZJj6ldumNsNnKochO7XpBk%3D&hash=f0144d26e7ac98ee5e23856799b2bf20a31192b31ec80acf2304de9a360484d4&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1364661319300610&tid=spdf-b147aad7-a62a-46a7-b55e-dc99d514caa4&sid=184488533b11b441814ad6a794f4e2e86772gxrqa&type=client) <br>

### Evolution Strategy
[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/pdf/1703.03864.pdf)([blog](https://blog.openai.com/evolution-strategies/), OpenAI 2017, ES，advantages of not calculating gradients/ easy to parallelize/more robust(such as frame-skip)) <br>
[A Visual Guide to Evolution Strategies](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/) <br>
[Evolving Stable Strategies](http://blog.otoro.net/2017/11/12/evolving-stable-strategies/)(ES on robot; task augmentation techniques) <br>


### Overview
[A Survey of Deep Network Solutions for Learning Control in Robotics: From Reinforcement to Imitation](https://arxiv.org/pdf/1612.07139.pdf) <br>
[Multi-Agent Reinforcement Learning: A Report on Challenges and Approaches](https://arxiv.org/pdf/1807.09427.pdf) <br>
[Deep Learning for Video Game Playing](https://arxiv.org/pdf/1708.07902.pdf) <br>
