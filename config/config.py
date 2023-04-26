import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--wandb', type=bool, default=True)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--load_model_path', type=str, default='./checkpoints')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--cutout', type=bool, default=True)
parser.add_argument('--dataset_path', type=str, default='./data')
parser.add_argument('--wandb_name', type=str, default='nameless')

args = parser.parse_args()

