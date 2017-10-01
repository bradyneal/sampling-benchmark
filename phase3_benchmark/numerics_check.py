# Ryan Turner (turnerry@iro.umontreal.ca)
import cPickle as pkl
import os
import sys
import numpy as np
import theano
import theano.tensor as T
from models import BUILD_MODEL, SAMPLE_MODEL
import fileio as io

DATA_CENTER = 'data_center'
DATA_SCALE = 'data_scale'


def logpdf(x, model_name, p):
    x_std = (x - p[DATA_CENTER]) / p[DATA_SCALE]
    ll = BUILD_MODEL[model_name](x_std, p)
    ll = ll - np.sum(np.log(p[DATA_SCALE]))
    return ll


def main():
    assert(len(sys.argv) == 2)
    config_file = io.abspath2(sys.argv[1])

    runs = 20

    config = io.load_config(config_file)

    model_list = io.get_model_list(config['input_path'], config['pkl_ext'])
    assert(all(io.is_safe_name(ss) for ss in model_list))
    print 'using models:'
    print model_list

    for param_name in model_list:
        if not param_name.endswith('RNADE'):
            continue

        model_file = param_name + config['pkl_ext']
        model_file = os.path.join(config['input_path'], model_file)
        print 'loading %s' % model_file
        assert(os.path.isabs(model_file))
        with open(model_file, 'rb') as f:
            model_setup = pkl.load(f)
        model_name, D, params_dict = model_setup

        x = T.vector('x')
        x.tag.test_value = np.zeros(D)

        ll = logpdf(x, model_name, params_dict)
        gg = T.grad(ll, x)

        logpdf_f = theano.function([x], [ll, gg])

        for ii in xrange(runs):
            x_test_m = SAMPLE_MODEL[model_name](params_dict, N=1)[0, :]
            x_test_n = np.random.randn(D)
            w = np.random.rand()
            x_test = w * x_test_m + (1.0 - w) * x_test_n

            lv, gv = logpdf_f(x_test)
            print lv
            print gv
            assert(np.isfinite(lv))
            assert(np.all(np.isfinite(gv)))
    print 'done'

if __name__ == '__main__':
    main()
