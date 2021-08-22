from hanser.train.lr_schedule import FlatCosineLR, CosineLR, CosinePowerAnnealingLR, MultiStepLR, ExponentialDecay, OneCycleLR, Knee

steps_per_epoch = 20
epochs = 100
total_steps = steps_per_epoch * epochs

lr_scheduler1 = FlatCosineLR(0.1, steps_per_epoch, epochs, 75, 0.0001, 5, 0.01)
xs1 = [lr_scheduler1(i).numpy() for i in range(total_steps)]

lr_scheduler2 = CosineLR(0.1, steps_per_epoch, epochs, 0.0001, 5, 0.01)
xs2 = [lr_scheduler2(i).numpy() for i in range(total_steps)]

lr_scheduler3 = CosinePowerAnnealingLR(0.1, steps_per_epoch, epochs, 10, 0.0001, 5, 0.01)
xs3 = [lr_scheduler3(i).numpy() for i in range(total_steps)]

lr_scheduler4 = MultiStepLR(0.1, steps_per_epoch, [30, 60, 90], 0.2, 5, 0.01)
xs4 = [lr_scheduler4(i).numpy() for i in range(total_steps)]

lr_scheduler5 = ExponentialDecay(0.1, steps_per_epoch, 2.4, 0.97, False, 5, 0.01)
xs5 = [lr_scheduler5(i).numpy() for i in range(total_steps)]

lr_scheduler6 = OneCycleLR(0.4, steps_per_epoch, epochs, 0.3, div_factor=10, warmup_epoch=5, warmup_min_lr=0.01)
xs6 = [lr_scheduler6(i).numpy() for i in range(total_steps)]

lr_scheduler7 = Knee(0.1, steps_per_epoch, epochs, 75, 0.0001, 5, 0.01)
xs7 = [lr_scheduler7(i).numpy() for i in range(total_steps)]

import matplotlib.pyplot as plt
plt.plot(xs7)