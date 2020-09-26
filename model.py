import copy
import random


import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras import callbacks as callbacks_module
from tensorflow.python.eager import context
from tensorflow.python.profiler import trace


def get_uncompiled_model(sample_size=128, model_name="", tensorboard_dir=""):
    inputs = tf.keras.Input(shape=(sample_size, sample_size, 3))
    initializer_x = tf.keras.layers.Conv2D(20, 3, (1, 1), padding="valid", activation="relu")(inputs)
    initializer_x = tf.keras.layers.Conv2D(30, 3, (1, 1), padding="valid", activation="relu")(initializer_x)
    initializer_x = tf.keras.layers.Conv2D(40, 3, (1, 1), padding="valid", activation="relu")(initializer_x)
    initializer_x = tf.keras.layers.Conv2D(50, 3, (1, 1), padding="valid", activation="relu")(initializer_x)

    initializer_x = tf.keras.layers.Conv2DTranspose(40, 3, (1, 1), padding="valid", activation="relu")(initializer_x)
    initializer_x = tf.keras.layers.Conv2DTranspose(30, 3, (1, 1), padding="valid", activation="relu")(initializer_x)
    initializer_x = tf.keras.layers.Conv2DTranspose(20, 3, (1, 1), padding="valid", activation="relu")(initializer_x)

    initializer_x = tf.keras.layers.Conv2DTranspose(3, 3, (1, 1), padding="valid", activation="sigmoid")(initializer_x)
    initializer_x = tf.keras.layers.add([tf.keras.layers.multiply([inputs, initializer_x]), 1 - initializer_x])

    return MainModel(inputs=inputs,
                     outputs=initializer_x,
                     name=model_name,
                     trainable=True,
                     tensorboard_dir=tensorboard_dir)


class MainModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        self.tensorboard_dir = kwargs["tensorboard_dir"]
        kwargs.pop("tensorboard_dir")
        from tensorflow.python.keras.engine import training
        from tensorflow.python.keras.engine import functional  # pylint: disable=g-import-not-at-top
        if (training.is_functional_model_init_params(args, kwargs) and
                not isinstance(self, functional.Functional)):
            training.inject_functional_model_class(self.__class__)
            functional.Functional.__init__(self, *args, **kwargs)

        self.predictions_file_writer = tf.summary.create_file_writer(
            logdir=self.tensorboard_dir + "/" + self.name + "/predictions")

        self.gradients_file_writer = tf.summary.create_file_writer(
            logdir=self.tensorboard_dir + "/" + self.name + "/gradients")

    def train_step(self, data):

        from tensorflow.python.keras.engine import data_adapter

        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
            trainable_variables = self.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)

            with self.gradients_file_writer.as_default():
                for gradient in gradients:
                    tf.summary.histogram(gradient.name, gradient, step=self._train_counter)

            self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    @training.enable_multi_worker
    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Arguments:
            x: Input data. It could be:
              - A Numpy array (or array-like), or a list of arrays
                (in case the model has multiple inputs).
              - A TensorFlow tensor, or a list of tensors
                (in case the model has multiple inputs).
              - A dict mapping input names to the corresponding array/tensors,
                if the model has named inputs.
              - A `tf.data` dataset. Should return a tuple
                of either `(inputs, targets)` or
                `(inputs, targets, sample_weights)`.
              - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
                or `(inputs, targets, sample_weights)`.
              A more detailed description of unpacking behavior for iterator types
              (Dataset, generator, Sequence) is given below.
            y: Target data. Like the input data `x`,
              it could be either Numpy array(s) or TensorFlow tensor(s).
              It should be consistent with `x` (you cannot have Numpy inputs and
              tensor targets, or inversely). If `x` is a dataset, generator,
              or `keras.utils.Sequence` instance, `y` should
              not be specified (since targets will be obtained from `x`).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of datasets, generators, or `keras.utils.Sequence` instances
                (since they generate batches).
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
                Note that the progress bar is not particularly useful when
                logged to a file, so verbose=2 is recommended when not running
                interactively (eg, in a production environment).
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `tf.keras.callbacks`.
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This argument is
                not supported when `x` is a dataset, generator or
               `keras.utils.Sequence` instance.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data. Thus, note the fact
                that the validation loss of data provided using `validation_split`
                or `validation_data` is not affected by regularization layers like
                noise and dropuout.
                `validation_data` will override `validation_split`.
                `validation_data` could be:
                  - tuple `(x_val, y_val)` of Numpy arrays or tensors
                  - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays
                  - dataset
                For the first two cases, `batch_size` must be provided.
                For the last case, `validation_steps` could be provided.
                Note that `validation_data` does not support all the data types that
                are supported in `x`, eg, dict, generator or `keras.utils.Sequence`.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch'). This argument is ignored
                when `x` is a generator. 'batch' is a special option for dealing
                with the limitations of HDF5 data; it shuffles in batch-sized
                chunks. Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class.
            sample_weight: Optional Numpy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample. This
                argument is not supported when `x` is a dataset, generator, or
               `keras.utils.Sequence` instance, instead provide the sample_weights
                as the third element of `x`.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                TensorFlow data tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined. If x is a
                `tf.data` dataset, and 'steps_per_epoch'
                is None, the epoch will run until the input dataset is exhausted.
                When passing an infinitely repeating dataset, you must specify the
                `steps_per_epoch` argument. This argument is not supported with
                array inputs.
            validation_steps: Only relevant if `validation_data` is provided and
                is a `tf.data` dataset. Total number of steps (batches of
                samples) to draw before stopping when performing validation
                at the end of every epoch. If 'validation_steps' is None, validation
                will run until the `validation_data` dataset is exhausted. In the
                case of an infinitely repeated dataset, it will run into an
                infinite loop. If 'validation_steps' is specified and only part of
                the dataset will be consumed, the evaluation will start from the
                beginning of the dataset at each epoch. This ensures that the same
                validation samples are used every time.
            validation_batch_size: Integer or `None`.
                Number of samples per validation batch.
                If unspecified, will default to `batch_size`.
                Do not specify the `validation_batch_size` if your data is in the
                form of datasets, generators, or `keras.utils.Sequence` instances
                (since they generate batches).
            validation_freq: Only relevant if validation data is provided. Integer
                or `collections_abc.Container` instance (e.g. list, tuple, etc.).
                If an integer, specifies how many training epochs to run before a
                new validation run is performed, e.g. `validation_freq=2` runs
                validation every 2 epochs. If a Container, specifies the epochs on
                which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
                validation at the end of the 1st, 2nd, and 10th epochs.
            max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
                input only. Maximum size for the generator queue.
                If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up
                when using process-based threading. If unspecified, `workers`
                will default to 1. If 0, will execute the generator on the main
                thread.
            use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children processes.

        Unpacking behavior for iterator-like inputs:
            A common pattern is to pass a tf.data.Dataset, generator, or
          tf.keras.utils.Sequence to the `x` argument of fit, which will in fact
          yield not only features (x) but optionally targets (y) and sample weights.
          Keras requires that the output of such iterator-likes be unambiguous. The
          iterator should return a tuple of length 1, 2, or 3, where the optional
          second and third elements will be used for y and sample_weight
          respectively. Any other type provided will be wrapped in a length one
          tuple, effectively treating everything as 'x'. When yielding dicts, they
          should still adhere to the top-level tuple structure.
          e.g. `({"x0": x0, "x1": x1}, y)`. Keras will not attempt to separate
          features, targets, and weights from the keys of a single dict.
            A notable unsupported data type is the namedtuple. The reason is that
          it behaves like both an ordered datatype (tuple) and a mapping
          datatype (dict). So given a namedtuple of the form:
              `namedtuple("example_tuple", ["y", "x"])`
          it is ambiguous whether to reverse the order of the elements when
          interpreting the value. Even worse is a tuple of the form:
              `namedtuple("other_tuple", ["x", "y", "z"])`
          where it is unclear if the tuple was intended to be unpacked into x, y,
          and sample_weight or passed through as a single element to `x`. As a
          result the data processing code will simply raise a ValueError if it
          encounters a namedtuple. (Along with instructions to remedy the issue.)

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        Raises:
            RuntimeError: 1. If the model was never compiled or,
            2. If `model.fit` is  wrapped in `tf.function`.

            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
        training._keras_api_gauge.get_cell('fit').set(True)
        # Legacy graph support is contained in `training_v1.Model`.
        version_utils.disallow_legacy_graph('Model', 'fit')
        self._assert_compile_was_called()
        self._check_call_args('fit')
        training._disallow_inside_tf_function('fit')

        if validation_split:
            # Create the validation data using the training data. Only supported for
            # `Tensor` and `NumPy` input.
            (x, y, sample_weight), validation_data = (
                data_adapter.train_validation_split(
                    (x, y, sample_weight), validation_split=validation_split))

        if validation_data:
            val_x, val_y, val_sample_weight = (
                data_adapter.unpack_x_y_sample_weight(validation_data))

        with self.distribute_strategy.scope(), \
             training_utils.RespectCompiledTrainableState(self):
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            data_handler = data_adapter.DataHandler(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                initial_epoch=initial_epoch,
                epochs=epochs,
                shuffle=shuffle,
                class_weight=class_weight,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self,
                steps_per_execution=self._steps_per_execution)

            # Container that configures and calls `tf.keras.Callback`s.
            if not isinstance(callbacks, callbacks_module.CallbackList):
                callbacks = callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=self,
                    verbose=verbose,
                    epochs=epochs,
                    steps=data_handler.inferred_steps)

            self.stop_training = False
            train_function = self.make_train_function()
            self._train_counter.assign(0)
            callbacks.on_train_begin()
            training_logs = None
            # Handle fault-tolerance for multi-worker.
            # happen after `callbacks.on_train_begin`.
            data_handler._initial_epoch = (  # pylint: disable=protected-access
                self._maybe_load_initial_epoch_from_ckpt(initial_epoch))

            random_image = int(random.uniform(0, len(validation_data)))

            for epoch, iterator in data_handler.enumerate_epochs():
                self.reset_metrics()
                callbacks.on_epoch_begin(epoch)
                with data_handler.catch_stop_iteration():
                    for step in data_handler.steps():
                        with trace.Trace(
                                'TraceContext',
                                graph_type='train',
                                epoch_num=epoch,
                                step_num=step,
                                batch_size=batch_size):
                            callbacks.on_train_batch_begin(step)
                            tmp_logs = train_function(iterator)
                            if data_handler.should_sync:
                                context.async_wait()
                            logs = tmp_logs  # No error, now safe to assign to logs.
                            end_step = step + data_handler.step_increment
                            callbacks.on_train_batch_end(end_step, logs)
                epoch_logs = copy.copy(logs)

                with self.predictions_file_writer.as_default():
                    tf.summary.image(name="train "+str(random_image),
                                     data=self.__call__(x[random_image:random_image+1]),
                                     step=epoch)

                # Run validation.
                if validation_data and self._should_eval(epoch, validation_freq):

                    with self.predictions_file_writer.as_default():
                        tf.summary.image(name="test " + str(random_image),
                                         data=self.__call__(val_x[random_image:random_image + 1]),
                                         step=epoch)

                    # Create data_handler for evaluation and cache it.
                    if getattr(self, '_eval_data_handler', None) is None:
                        self._eval_data_handler = data_adapter.DataHandler(
                            x=val_x,
                            y=val_y,
                            sample_weight=val_sample_weight,
                            batch_size=validation_batch_size or batch_size,
                            steps_per_epoch=validation_steps,
                            initial_epoch=0,
                            epochs=1,
                            max_queue_size=max_queue_size,
                            workers=workers,
                            use_multiprocessing=use_multiprocessing,
                            model=self,
                            steps_per_execution=self._steps_per_execution)
                    val_logs = self.evaluate(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps=validation_steps,
                        callbacks=callbacks,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        return_dict=True)
                    val_logs = {'val_' + name: val for name, val in val_logs.items()}
                    epoch_logs.update(val_logs)

                callbacks.on_epoch_end(epoch, epoch_logs)
                training_logs = epoch_logs
                if self.stop_training:
                    break

            # If eval data_hanlder exists, delete it after all epochs are done.
            if getattr(self, '_eval_data_handler', None) is not None:
                del self._eval_data_handler
            callbacks.on_train_end(logs=training_logs)
            return self.history
