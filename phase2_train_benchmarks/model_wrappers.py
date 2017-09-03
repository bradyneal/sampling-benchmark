# Ryan Turner (turnerry@iro.umontreal.ca)
import os
import numpy as np
from sklearn.mixture import GaussianMixture

# IGN stuff
import bench_models.ign.ign as ign
import bench_models.ign.t_util as t_util

# RNADE stuff
from bench_models.nade.Data import Dataset, BigDataset
import bench_models.nade.NADE as NADE
import bench_models.nade.Instrumentation as Instrumentation
import bench_models.nade.Backends as Backends
import bench_models.nade.Optimization as Optimization
import bench_models.nade.TrainingController as TrainingController
from bench_models.nade.Utils.DropoutMask import create_dropout_masks
from bench_models.nade.Utils.theano_helpers import floatX

# TODO put in wrappers for:
# simple gauss (diag and full), MoG, RNADE, IGN, RNVP (py2 vs 3 issues?)
# later: other mixtures (MoT?), RNADE variants like laplace,
# other implentations of RNVP/RNADE, maybe fancy stuff like DPM
# factor analysis/PCA also possible as ref points


class GaussianMixture_(GaussianMixture):
    '''We could use multiple inheritence to enforce that all these classes have
    a get_params() method, but that might be more trouble than it is worth.'''

    def get_params(self):
        '''Add get_params object to GaussianMixture.'''
        D = {'weights': self.weights_, 'means': self.means_,
             'covariances': self.covariances_, 'type': self.covariance_type}
        return D


class IGN:
    def __init__(self, n_layers=1, WL_init=1e-2, reg_dict={}, valid_frac=0.2,
                 gauss_basepdf=True, n_epochs=100, batch_size=32):
        self.n_layers = n_layers
        self.WL_init = WL_init
        self.reg_dict = reg_dict
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.valid_frac = valid_frac

        # Only parts that need to be saved in param extraction
        self.gauss_basepdf = gauss_basepdf
        self.fit_layers = None

        self.base_logpdf = t_util.norm_logpdf_T if self.gauss_basepdf \
            else t_util.t_logpdf_T

    def fit(self, x_train):
        N, D = x_train.shape
        n_train = int(np.ceil((1.0 - self.valid_frac) * N))

        # Could also have setting in there for trying LU vs full training
        layers = ign.init_ign(self.n_layers, D)
        # Chunk some in validation
        R = ign.train_ign(x_train[:n_train, :], x_train[n_train:, :],
                          layers, self.reg_dict, base_logpdf=self.base_logpdf,
                          n_epochs=self.n_epochs, batch_size=self.batch_size)
        _, _, _, self.fit_layers = R

    def score_samples(self, x_test):
        assert(self.fit_layers is not None)
        logpdf, _ = ign.ign_log_pdf(x_test, self.fit_layers, self.base_logpdf)
        return logpdf

    def get_params(self):
        assert(self.fit_layers is not None)
        # Make a copy, maybe we should even make a deep copy
        D = dict(self.fit_layers)
        D['gauss_basepdf'] = self.gauss_basepdf
        return D


class RNADE:
    def __init__(self, n_hidden=100, h_layers=2, n_components=10, epochs=20,
                 pretraining_epochs=5, validation_loops=20, valid_frac=0.2,
                 lr=0.02, epoch_size=100, batch_size=100, momentum=0.9,
                 dataset_name='RNADE_test_run', scratch_dir='.'):
        # TODO eliminate need for these things
        self.dataset_name = dataset_name
        self.scratch_dir = scratch_dir

        self.n_hidden = n_hidden
        self.h_layers = h_layers
        self.n_components = n_components
        self.nonlinearity = 'RLU'  # Might just want to hard code this one

        self.epochs = epochs
        self.pretraining_epochs = pretraining_epochs
        self.validation_loops = validation_loops

        self.lr = lr
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.momentum = momentum
        self.valid_frac = valid_frac

        self.nade_class = NADE.OrderlessMoGNADE
        self.nade_obj = None

    def fit(self, x_train):
        # TODO create short aliases on import for some of the long names here
        # to avoid these super long lines
        # TODO change " to ' to keep style consistent

        N, n_visible = x_train.shape
        n_train = int(np.ceil((1.0 - self.valid_frac) * N))

        training_dataset = Dataset(x_train[:n_train, :])
        validation_dataset = Dataset(x_train[n_train:, :])

        # Rest of stuff in here copied/adapted from orderlessNADE.py, seems
        # more complicated than it needs to be.  Can it be made simpler??

        # TODO Maybe we can do this in memory without writing files out
        masks_filename = self.dataset_name + "." + floatX + ".masks"
        masks_route = os.path.join(self.scratch_dir, masks_filename)
        create_dropout_masks(self.scratch_dir, masks_filename, n_visible, ks=1000)
        masks_dataset = BigDataset(masks_route + ".hdf5", "masks/.*", "masks")

        nade = self.nade_class(n_visible, self.n_hidden, 1, self.n_components, nonlinearity=self.nonlinearity)
        loss_function = 'sym_masked_neg_loglikelihood_gradient'
        validation_loss = lambda ins: -ins.model.estimate_average_loglikelihood_for_dataset_using_masks(validation_dataset, masks_dataset, loops=self.validation_loops)
        validation_loss_measurement = Instrumentation.Function('validation_loss', validation_loss)

        # TODO make optional
        console = Backends.Console()
        textfile_log = Backends.TextFile(os.path.join(self.scratch_dir, "NADE_training.log"))
        hdf5_backend = Backends.HDF5(self.scratch_dir, "NADE")

        # Pretrain layerwise
        for l in xrange(1, self.h_layers + 1):
            if l == 1:
                nade.initialize_parameters_from_dataset(training_dataset)
            else:
                nade = self.nade_class.create_from_smaller_NADE(nade, add_n_hiddens=1)

            # Configure training
            trainer = Optimization.MomentumSGD(nade, nade.__getattribute__(loss_function))
            trainer.set_datasets([training_dataset, masks_dataset])
            trainer.set_learning_rate(self.lr)
            trainer.set_datapoints_as_columns(True)
            trainer.add_controller(TrainingController.AdaptiveLearningRate(self.lr, 0, epochs=self.pretraining_epochs))
            trainer.add_controller(TrainingController.MaxIterations(self.pretraining_epochs))
            # TODO use np.inf in these spots
            trainer.add_controller(TrainingController.ConfigurationSchedule("momentum", [(2, 0), (float('inf'), self.momentum)]))
            trainer.set_updates_per_epoch(self.epoch_size)
            trainer.set_minibatch_size(self.batch_size)
            trainer.add_controller(TrainingController.NaNBreaker())

            # Instrument the training
            trainer.add_instrumentation(Instrumentation.Instrumentation([console, textfile_log, hdf5_backend],
                                                                        Instrumentation.Function("training_loss", lambda ins: ins.get_training_loss())))
            trainer.add_instrumentation(Instrumentation.Instrumentation([console, textfile_log, hdf5_backend], Instrumentation.Configuration()))
            trainer.add_instrumentation(Instrumentation.Instrumentation([console, textfile_log, hdf5_backend], Instrumentation.Timestamp()))

            # Train
            trainer.set_context("pretraining_%d" % l)
            trainer.train()

        # Configure training
        ordering = range(n_visible)  # TODO should this be np array??
        np.random.shuffle(ordering)
        trainer = Optimization.MomentumSGD(nade, nade.__getattribute__(loss_function))
        trainer.set_datasets([training_dataset, masks_dataset])
        trainer.set_learning_rate(self.lr)
        trainer.set_datapoints_as_columns(True)
        trainer.add_controller(TrainingController.AdaptiveLearningRate(self.lr, 0, epochs=self.epochs))
        trainer.add_controller(TrainingController.MaxIterations(self.epochs))
        trainer.add_controller(TrainingController.ConfigurationSchedule("momentum", [(2, 0), (float('inf'), self.momentum)]))
        trainer.set_updates_per_epoch(self.epoch_size)
        trainer.set_minibatch_size(self.batch_size)

        trainer.add_controller(TrainingController.NaNBreaker())

        # Instrument the training
        trainer.add_instrumentation(Instrumentation.Instrumentation([console, textfile_log, hdf5_backend],
                                                                    Instrumentation.Function("training_loss", lambda ins: ins.get_training_loss())))

        trainer.add_instrumentation(Instrumentation.Instrumentation([console], validation_loss_measurement))
        trainer.add_instrumentation(Instrumentation.Instrumentation([hdf5_backend], validation_loss_measurement, at_lowest=[Instrumentation.Parameters()]))
        trainer.add_instrumentation(Instrumentation.Instrumentation([console, textfile_log, hdf5_backend], Instrumentation.Configuration()))
        trainer.add_instrumentation(Instrumentation.Instrumentation([console, textfile_log, hdf5_backend], Instrumentation.Timestamp()))

        # Train
        trainer.train()

        if trainer.was_successful():
            self.nade_obj = nade
        else:
            # TODO convert to real warning
            print 'Training error in RNADE!'

    def score_samples(self, x_test):
        assert(self.nade_obj is not None)
        logpdf = self.nade_obj.logdensity(x_test.T)
        assert(logpdf.shape == (x_test.shape[0],))
        return logpdf

    def get_params(self):
        assert(self.nade_obj is not None)
        D = self.nade_obj.get_parameters()
        # This could be a deep copy or cast to np array if we wanted to be safe
        D['orderings'] = self.nade_obj.orderings
        return D

# Dict with sklearn like interfaces for each of the models we will use to train
# samples.
STD_BENCH_MODELS = {'MoG': GaussianMixture_, 'IGN': IGN, 'RNADE': RNADE}
