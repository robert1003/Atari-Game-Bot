def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--model_name', type=str, help='model file prefix')
    parser.add_argument('--log_name', type=str, help='log file name')
    parser.add_argument('--dqn_gamma', type=float, default=0.99, help='GAMMA for dqn')
    parser.add_argument('--dqn_target_update_freq', type=int, default=1000, help='how frequent do we update target network')

    return parser
