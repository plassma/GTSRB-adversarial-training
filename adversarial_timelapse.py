class AdversarialTimelapse:
    def __init__(self, testdata):
        self.N_SAMPLES = 5
        self.adversarials = []
        self.originals = []
        self.originals = testdata[0:self.N_SAMPLES]

    def add_adversarials(self, adversarials):
        self.adversarials.append(adversarials[0:self.N_SAMPLES])

    def plot_timelapse(self, architecture, adv):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(nrows=self.N_SAMPLES, ncols=len(self.adversarials) + 1, figsize=(10, 4.5))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        architecture = architecture + ("_with" + ("out" if not adv else ""))
        # title = "Adversarials for " + architecture + " adversarial training per Epoch"

        # plt.text(1.5, 1.3, title, horizontalalignment='center',
        #          fontsize=20,
        #          transform=axs[0, 1].transAxes)
        plt.text(0.5, -0.65, "Epoch ", horizontalalignment='center', fontsize=17.5,
                 transform=axs[len(self.adversarials[0]) - 1, 5].transAxes)

        for i, original in enumerate(self.originals):
            ax = axs[i, 0]  # [5,11]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(original)

        plt.text(.5, -0.3, "Original", horizontalalignment='center', fontsize=15,
                 transform=axs[len(self.originals) - 1, 0].transAxes)

        for i, adv_set in enumerate(self.adversarials):
            for j, adversarial in enumerate(adv_set):
                ax = axs[j, i + 1] # [5,11]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(adversarial)
            plt.text(.5, -0.3, i + 1, horizontalalignment='center', fontsize=15,
                     transform=axs[len(adv_set) - 1, i + 1].transAxes)

        plt.savefig("results/adversarials_over_epochs/" + architecture + ".png")
