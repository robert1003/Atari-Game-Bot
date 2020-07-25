# ADL HW3

## Homework details

* [Link](https://docs.google.com/presentation/d/12IjcsZVCKjcsHtCRKSJfku3HWlGN4yc9DvCT6XRJZlk/edit?usp=sharing) to the homework slide.
* [Link](https://youtu.be/cwdiBWjQDk0) to the homework video.
* [Link](https://drive.google.com/file/d/1Ctf8hVx-JCFTR5p0incL-F5ASPfwLKtP/view?usp=sharing) to the sample code.

## Installation
Type the following command to install OpenAI Gym Atari environment.

`$ pip3 install opencv-python gym gym[atari]`

Please refer to [OpenAI's page](https://github.com/openai/gym) if you have any problem while installing.

## How to run :
training policy gradient:
* `$ python3 main.py --train_pg --model_name pg --log_name pg.log`

testing policy gradient:
* `$ python3 test.py --test_pg`

training DQN:
* `$ python3 main.py --train_dqn --model_name dqn --log_name dqn.log --dqn_gamma 0.9`

testing DQN:
* `$ python3 test.py --test_dqn`

If you want to see your agent playing the game,
* `$ python3 test.py --test_[pg|dqn] --do_render`

## Plot figure

* p1_pg: python3 p1_pg.py
* p1_dqn: python3 p1_dqn.py
* p2: python3 p2.py
* p3 variance reduction: python3 p3_pg.py
* p3 DDQN: python3 p3_dqn.py
