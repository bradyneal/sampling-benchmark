# Ryan Turner (turnerry@iro.umontreal.ca)
import os
import sys
import main as m


def chomp(ss, ext):
    # TODO move to general util
    L = len(ext)
    assert(ss[-L:] == ext)
    return ss[:-L]


def get_model_list(input_path, ext):
    L = sorted(chomp(fname, ext) for fname in os.listdir(input_path))
    return L


def main():
    # Could use a getopt package if this got fancy, but this is simple enough
    assert(len(sys.argv) == 4)
    config_file = m.abspath2(sys.argv[1])
    max_N = int(sys.argv[2])
    N_chains = int(sys.argv[3])
    # TODO add option to control random seed
    assert(max_N >= 0)

    config = m.load_config(config_file)

    input_path, output_path, pkl_ext, meta_ext, exact_name, t_grid_ms = config
    model_list = get_model_list(input_path, pkl_ext)
    assert(all(m.is_safe_name(ss) for ss in model_list))
    print 'using models:'
    print model_list

    sampler_list = m.BUILD_STEP.keys()
    print 'using samplers:'
    print sampler_list

    for param_name in model_list:
        m.run_experiment(config, param_name, exact_name, max_N)

    for _ in xrange(N_chains):
        for param_name in model_list:
            for sampler in sampler_list:
                m.run_experiment(config, param_name, sampler, max_N)
    print 'done'  # Job will probably get killed before we get here.

if __name__ == '__main__':
    main()
