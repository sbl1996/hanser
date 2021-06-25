import torch
from torch.distributions import RelaxedBernoulli

num_sub_policies, num_ops = 5, 2
temperature = 0.5
probabilities = torch.full((num_sub_policies, num_ops), 0.5, requires_grad=True)
magnitudes = torch.full((num_sub_policies, num_ops), 0.5, requires_grad=True)

def sample():
    probabilities_dist = RelaxedBernoulli(temperature, probabilities)
    sample_probabilities = probabilities_dist.rsample()
    sample_probabilities = sample_probabilities.clamp(0.0, 1.0)
    sample_probabilities_index = sample_probabilities >= 0.5
    sample_probabilities = sample_probabilities_index.float() - sample_probabilities.detach() + sample_probabilities
    return sample_probabilities, sample_probabilities_index


def forward(images, probability, probability_index, magnitude):
    print(images[0,0,0,0])
    index = sum(p_i.item() << i for i, p_i in enumerate(probability_index))
    com_image = 0
    adds = 0
    print(probability_index, index)
    print("-----------------------------------")
    for selection in range(2 ** num_ops):
        trans_probability = 1
        for i in range(num_ops):
            print("Iter: %d/%d" % (selection, i))
            print(float(com_image[0,0,0,0]) if torch.is_tensor(com_image) and com_image.ndim == 4 else float(com_image))
            if selection & (1 << i):
                print("Yes")
                trans_probability = trans_probability * probability[i]
                if selection == index:
                    images = images - magnitude[i].detach()
                    adds = adds + magnitude[i]
            else:
                print("No")
                trans_probability = trans_probability * (1 - probability[i])
        print()
        print("trans", float(trans_probability))
        print("adds", float(adds))
        if selection == index:
            images = images + adds
            com_image = com_image + trans_probability * images
        else:
            com_image = com_image + trans_probability
        print("-----------------------------------")

    # com_image = probability * trans_images + (1 - probability) * origin_images
    return com_image

torch.manual_seed(1)
probabilities.grad = None
magnitudes.grad = None
i = 3
images = torch.randn(2, 4, 4, 1)
sample_probabilities, sample_probabilities_index = sample()
magnitude = magnitudes[i]
probability, probability_index = sample_probabilities[i], sample_probabilities_index[i]
com_image = forward(images, probability, probability_index, magnitude)
com_image.sum().backward()
(probability, probability_index)