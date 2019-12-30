from experiments import confusion_matrix, iterative_adversarial_training
import networks

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

    parser.add_argument(
        "-l", "--labels", dest="labels", type=str, required=False,
        default="0,1,2,3",
        help="enter labels to confuse in matrix"
    )

    parser.add_argument(
        "-eps", "---epsilon", dest="epsilon", type=float, required=False,
        default=0.03,
        help="epsilon for BIM"
    )

    parser.add_argument(
        "-lam", "---lambda", dest="lam", type=float, required=False,
        default=0,
        help="lambda for input gradient regularization"
    )

    parser.add_argument(
        "-adv", "--adversarial", dest="adversarial", type=bool, required=False,
        default=False,
        help="whether to use adversarial training"
    )

    parser.add_argument(
        "-i", "--iterations", dest="iterations", type=int, required=False,
        default=15,
        help="number of iterations in BIM"
    )

    if not argv:
        print("Some required argument is missing")
    args = parser.parse_args(argv)

    if hasattr(args, "lam"):
        networks.LAMBDA = args.lam

    if args.experiment == "AT":
        iterative_adversarial_training(args.architecture, args.method)
    elif args.experiment == "CM":
        confusion_matrix(args.labels, args.architecture, args.method, args.epsilon, args.iterations, args.adversarial)
