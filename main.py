from experiments import confusion_matrix, iterative_adversarial_training

if __name__ == "__main__":
    import sys, argparse

    argv = sys.argv[1:]
    usage_text = "Run as " + __file__ + " [options]"
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument(
        "-a", "--architecture", dest="architecture", required=True,
        choices=['alex', 'vgg19', 'resnet50', 'lenet-5'],
        help="model architecture for training. \nOptions: ['alex', 'vgg19', 'resnet50', 'lenet-5']"
    )
    parser.add_argument(
        "-m", "--method", dest="method", type=str, required=True,
        choices=["FGSM", "GAUSSIAN"],
        default="FGSM",
        help="choose from FGSM, GAUSSIAN"
    )
    parser.add_argument(
        "-e", "--experiment", dest="experiment", type=str, required=True,
        choices=["AT", "CM"]
    )

    if not argv:
        print("Some required argument is missing")
    args = parser.parse_args(argv)

    if args.experiment == "AT":
        iterative_adversarial_training(args.architecture, args.method)
    elif args.experiment == "CM":
        confusion_matrix([0, 1, 2, 3], args.architecture, args.method)