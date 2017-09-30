# Ryan Turner (turnerry@iro.umontreal.ca)
import sys
import fileio as io
from main import run_experiment
# This will import pymc3 which is not needed if the experiments are run in a
# separate process in the future. Loading pymc3 will be a bit of a waste just
# to get the dictionary keys. We could re-work this, but prob not worth effort.
from samplers import BUILD_STEP_PM, BUILD_STEP_MC


def main():
    assert(len(sys.argv) == 2)
    config_file = io.abspath2(sys.argv[1])

    config = io.load_config(config_file)

    model_list = io.get_model_list(config['input_path'], config['pkl_ext'])
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
    for _ in xrange(config['n_chains']):
        for model_name in model_list:
            for sampler in sampler_list:
                run_experiment(config, model_name, sampler)
    print 'done'

if __name__ == '__main__':
    main()
