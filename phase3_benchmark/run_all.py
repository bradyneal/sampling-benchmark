# Ryan Turner (turnerry@iro.umontreal.ca)
import sys
from time import time
import fileio as io
from main import run_experiment
# This will import pymc3 which is not needed if the experiments are run in a
# separate process in the future. Loading pymc3 will be a bit of a waste just
# to get the dictionary keys. We could re-work this, but prob not worth effort.
from samplers import BUILD_STEP_PM, BUILD_STEP_MC


def main():
    num_args = len(sys.argv) - 1
    if num_args < 1:
        config_path = '../config.ini'
    elif num_args > 1:
        raise Exception('too many arguments: %d. %d expected' % (num_args, 1))
    else:
        config_path = sys.argv[1]
    config_file = io.abspath2(config_path)

    config = io.load_config(config_file)

    model_list = io.get_model_list(config['input_path'], config['pkl_ext'])
    # model_list = model_list[:5]  # TODO remove, test only
    assert(all(io.is_safe_name(ss) for ss in model_list))
    print 'using models:'
    print model_list

    # Sort for reprodicibility
    sampler_list = sorted(BUILD_STEP_PM.keys() + BUILD_STEP_MC.keys())
    print 'using samplers:'
    print sampler_list

    # Get the exact samples
    for model_name in model_list:
        run_experiment(config, model_name, config['exact_name'])

    # Run n_chains in the outer loop since if process get killed we have less
    # chains but with even distribution over models and samplers.
    for model_name in model_list:
        for _ in xrange(config['n_chains']):
            # TODO could put ADVI init here to keep it fixed across samplers
            for sampler in sampler_list:
                t = time()
                try:
                    run_experiment(config, model_name, sampler)
                except Exception as err:
                    print '%s/%s failed' % (model_name, sampler)
                    print str(err)
                print 'wall time %fs' % (time() - t)
    print 'done'

if __name__ == '__main__':
    main()
