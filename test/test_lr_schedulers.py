from hanser.train.lr_schedule import FlatCosineLR, CosineLR, CosinePowerAnnealingLR, MultiStepLR, ExponentialDecay

lr_scheduler1 = FlatCosineLR(0.1, 20, 100, 75, 0.0001, 5, 0.01)
xs1 = [lr_scheduler1(i).numpy() for i in range(2000)]

lr_scheduler2 = CosineLR(0.1, 20, 100, 0.0001, 5, 0.01)
xs2 = [lr_scheduler2(i).numpy() for i in range(2000)]

lr_scheduler3 = CosinePowerAnnealingLR(0.1, 20, 100, 10, 0.0001, 5, 0.01)
xs3 = [lr_scheduler3(i).numpy() for i in range(2000)]

lr_scheduler4 = MultiStepLR(0.1, 20, [30, 60, 90], 0.2, 5, 0.01)
xs4 = [lr_scheduler4(i).numpy() for i in range(2000)]

lr_scheduler5 = ExponentialDecay(0.1, 20, 2.4, 0.97, False, 5, 0.01)
xs5 = [lr_scheduler5(i).numpy() for i in range(2000)]
