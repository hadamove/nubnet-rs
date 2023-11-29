# TODO before finish

## Increase accuracy

As of now, we're kind of stuck around 88.5% accuracy, which way too close to the 88% boundary to pass the assignment. We need to try some more things to get the accuracy up:

- [x] multithreading
- [x] try adding momentum back, see if it improves the accuracy with different values
  - tried, it seems to have a large effect
- [x] experiment with layers, try adding more layers
  - tried, 32, 16 in hidden layers performs worse, 128, 64 performs better but is too slow (parallelism might help)
- [ ] hyperparameter search -- in progress, will during night
- [ ] try adding learning rate scheduler
- [ ] try adding dropout
- [ ] try adding batch normalization

> Notes:
>
> - 32 batch size [seems to be winning](https://wandb.ai/stacey/fmnist/reports/Hyperparameters-of-a-Simple-CNN-Trained-on-Fashion-MNIST--Vmlldzo1MjU2Mg) by a small margin in 0-10 epochs range, if we need more performance 128 batch size is the way to go
> - We can create some thread pool and recycle threads so we don't spawn new threads for each `train_on_multiple` call as spawning a thread might be expensive and might actually slow things down

## Other TODOs

- [x] fill in UCOS!
- [ ] setup rust on aura more nicely so that it does not have to install on every run
- [ ] final testing on aura
- [ ] remove logging (marked with TODO) from code before submitting!!!
- [ ] remove this file
- [ ] submit the solution before 4.12.2023 23:59
