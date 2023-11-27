# TODO before finish

## Increase accuracy

As of now, we're kind of stuck around 88.5% accuracy, which way too close to the 88% boundary to pass the assignment. We need to try some more things to get the accuracy up:

- [x] try adding momentum back, see if it improves the accuracy with different values
  - tried, it seems to have a large effect
- [x] experiment with layers, try adding more layers
  - tried, 32, 16 in hidden layers performs worse, 128, 64 performs better but is too slow (parallelism might help)
- [ ] hyperparameter search -- in progress, will during night
- [ ] try adding learning rate scheduler
- [ ] try adding dropout
- [ ] try adding batch normalization

## Path to multi-threading (potentionally speeding things up 32-128 times)

1. decouple weights from the layer struct (place it in model struct directly, it should be Arc\<RwLock>?)
   - We somehow need to rethink the way we store them, is it going to be Vec\<Vec\<Vec\<f64>>> or Vec\<f64> and use some smart indexing?
   - FW/BW pass & weight update will need to be updated to use the new weights, it might completely break the code
2. weights are shared across threads, each thread has its own potentials, activations, and gradients (the model struct holds a copy of Vec\<Layer> for each thread, they also must be wrapped with Arc\<RwLock> or Arc\<Mutex> to avoid copying data between threads)
3. write method `train_on_multiple` which takes a batch of data and labels equal to the number of threads, and train on each thread.
   - perhaps the number of threads could be made const generic
4. spawn a thread for each input
5. pass a reference to the weights to each thread (Arc\<RwLock>), which will be readonly
6. pass reference to the potentials, activations, and gradients to each thread (Arc\<RwLock>) which will be read and write but that is ok because each thread has its own copy
7. run forward and backward pass on each thread (32 threads is a good start)
8. use barrier to synchronize all threads, wait until all finish
9. once all threads are finish FW and BW pass, its safe to aquire the RwLock on the weights on the main thread
10. we update the weights by averaging the gradients from each thread

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
