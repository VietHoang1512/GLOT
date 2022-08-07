from __future__ import print_function, absolute_import

import argparse

from model_mnist import ModelBaseline, ModelADA, ModelMEADA, ModelLatent, ModelInput


def main():
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--seed", type=int, default=1, help="")
    train_arg_parser.add_argument(
        "--algorithm",
        type=str,
        default="MEADA",
        choices=["ERM", "ADA", "MEADA", "LWDR", "LVAT", "WDR", "VAT"],
        help="Choose algorithm.",
    )
    train_arg_parser.add_argument("--test_every", type=int, default=50, help="")
    train_arg_parser.add_argument("--batch_size", type=int, default=32, help="")
    train_arg_parser.add_argument("--num_classes", type=int, default=10, help="")
    train_arg_parser.add_argument("--step_size", type=int, default=1, help="")
    train_arg_parser.add_argument("--loops_train", type=int, default=100000, help="")
    train_arg_parser.add_argument("--loops_min", type=int, default=100, help="")
    train_arg_parser.add_argument("--loops_adv", type=int, default=15, help="")
    train_arg_parser.add_argument("--seen_index", type=int, default=0, help="")
    train_arg_parser.add_argument("--lr", type=float, default=0.0001, help="")
    train_arg_parser.add_argument("--lr_max", type=float, default=1.0, help="")
    train_arg_parser.add_argument("--weight_decay", type=float, default=0.0, help="")
    train_arg_parser.add_argument("--logs", type=str, default="logs/", help="")
    train_arg_parser.add_argument("--model_path", type=str, default="", help="")
    train_arg_parser.add_argument("--deterministic", type=bool, default=False, help="")
    train_arg_parser.add_argument("--k", type=int, default=5, help="")
    train_arg_parser.add_argument("--gamma", type=float, default=1.0, help="")
    train_arg_parser.add_argument("--eta", type=float, default=1.0, help="")
    train_arg_parser.add_argument("--xi", default=1e-2, type=float, help="VAT xi")
    train_arg_parser.add_argument("--eps", default=1, type=float, help="VAT epsilon")
    train_arg_parser.add_argument(
        "--n_particles", default=2, type=int, help="number of adversarial particles"
    )
    train_arg_parser.add_argument(
        "--loss_type",
        type=str,
        default="trade",
        choices=["pgd", "trade"],
        help="Choose loss type.",
    )
    args = train_arg_parser.parse_args()

    if args.algorithm == "ERM":
        model_obj = ModelBaseline(flags=args)
    elif args.algorithm == "ADA":
        model_obj = ModelADA(flags=args)
    elif args.algorithm == "MEADA":
        model_obj = ModelMEADA(flags=args)
    elif args.algorithm in ["LWDR", "LVAT"]:
        model_obj = ModelLatent(flags=args)
    elif args.algorithm in ["WDR", "VAT"]:
        model_obj = ModelInput(flags=args)
    else:
        raise RuntimeError
    model_obj.train(flags=args)


if __name__ == "__main__":
    main()
