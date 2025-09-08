import argparse
import json

"""
for name, param in model.named_parameters():
    print(name, param.is_cuda)
"""


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--train_percentage', type=float, default=1.0, help='Percentage of training data to use')
        self.add_argument('--calculate_complexity', action='store_true', help='Whether to calculate model complexity')
        self.add_argument('--dataset_name', type=str, default='mimiciv')
        self.add_argument('--model_name', type=str, default='TransformerEncoder')
        self.add_argument('--data_dir', type=str, required=True)
        self.add_argument('--output_dir', type=str, required=True)

        self.add_argument('--max_num_codes', type=int, default=30)
        self.add_argument('--feature_keys', action='append', default=['dx_ints', 'proc_ints'])
        self.add_argument('--prior_scalar', type=float, default=0.5)
        self.add_argument('--loss_coef', type=float, default=0.0)
        
        self.add_argument('--mimiciv_vocab_sizes', type=json.loads, default={'dx_ints': 7813, 'proc_ints': 8752, 'demographics_ints': 256, 'vital_signs_ints': 69343})
        self.add_argument('--mimiciii_vocab_sizes', type=json.loads, default={'dx_ints': 6322, 'proc_ints': 1893, 'demographics_ints': 220, 'vital_signs_ints': 59772})

        self.add_argument('--dx_code_size', type=int, default=6508)
        self.add_argument('--dx_num_tree_nodes', type=int, default=100)
        self.add_argument('--dx_num_ccs_cat1', type=int, default=18)

        self.add_argument('--proc_code_size', type=int, default=6508)
        self.add_argument('--proc_num_tree_nodes', type=int, default=100)
        self.add_argument('--proc_num_ccs_cat1', type=int, default=16)

        self.add_argument('--num_stacks', type=int, default=8)
        self.add_argument('--hidden_size', type=int, default=128)
        self.add_argument('--intermediate_size', type=int, default=256)
        self.add_argument('--num_heads', type=int, default=1)
        self.add_argument('--hidden_dropout_prob', type=float, default=0.25)

        self.add_argument('--learning_rate', type=float, default=1e-3)
        self.add_argument('--eps', type=float, default=1e-8)
        self.add_argument('--batch_size', type=int, default=64)
        self.add_argument('--max_grad_norm', type=float, default=1.0)

        self.add_argument('--use_guide', default=False, action='store_true')
        self.add_argument('--use_prior', default=False, action='store_true')

        self.add_argument('--output_hidden_states', default=False, action='store_true')
        self.add_argument('--output_attentions', default=False, action='store_true')

        self.add_argument('--fold', type=int, default=42)
        self.add_argument('--eval_batch_size', type=int, default=256)

        self.add_argument('--warmup', type=float, default=0.05)
        self.add_argument('--logging_steps', type=int, default=100)
        self.add_argument('--max_steps', type=int, default=100000)
        self.add_argument('--num_train_epochs', type=int, default=0)

        self.add_argument('--label_key', type=str, default='expired')
        self.add_argument('--num_labels', type=int, default=2)

        self.add_argument('--reg_coef', type=float, default=0)
        self.add_argument('--seed', type=int, default=42)

        self.add_argument('--do_train', default=False, action='store_true')
        self.add_argument('--do_eval', default=False, action='store_true')
        self.add_argument('--do_test', default=False, action='store_true')
        self.add_argument('--output_model', default=False, action='store_true')

    def parse_args(self):
        args = super().parse_args()
        return args
