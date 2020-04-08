class AdversarialTimelapse:
    def __init__(self):
        self.N_SAMPLES = 4
        self.adversarials = []

    def add_adversarials(self, adversarials):
        self.adversarials.append(adversarials[0:self.N_SAMPLES])

    def plot_timelapse(self):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(nrows=self.N_SAMPLES, ncols=len(self.adversarials), figsize=(10, 10))

        for i, adv_set in enumerate(self.adversarials):
            for j, adversarial in enumerate(adv_set):
                ax = axs[j, i]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(adversarial)

        plt.show()
