# Ryan Turner (turnerry@iro.umontreal.ca)
import os
import numpy as np
import pandas as pd
from diagnostics import STD_DIAGNOSTICS
from metrics import STD_METRICS


def chomp(ss, ext):
    # TODO move to common util folder to avoid repetition
    L = len(ext)
    assert(ss[-L:] == ext)
    return ss[:-L]


def find_traces(input_path, exclude=(), ext='.csv'):
    # TODO switch to organized meta-data file in input dir with all files

    # Would prob not cause problem to remove sort here, but general good
    # practice with os.listdir()
    files = sorted(os.listdir(input_path))

    examples = set()
    samplers = set()
    for fname in files:
        # TODO create general parser for filenames in the project util
        fname = chomp(fname, ext)
        # Format: example-name_sampler-name.csv
        # example may also contain _ since it is combo of phases 0-2
        L = fname.rsplit('_', 1)
        assert(len(L) == 2)
        examples.add(L[0])
        samplers.add(L[1])
    samplers.difference(exclude) # For now exclude is only for samplers
    samplers, examples = sorted(samplers), sorted(examples)
    return samplers, examples


def build_trace_name(input_path, example, sampler, ext='.csv'):
    fname = '%s_%s%s' % (example, sampler, ext)
    return os.path.join(input_path, fname)


def load_chain(fname):
    # TODO move to common util
    X = np.loadtxt(fname, delimiter=',')
    return X


def dump_results(df, output_path, tbl_name, ext='.csv'):
    fname = tbl_name + ext
    fname = os.path.join(output_path, fname)
    df.to_csv(fname, na_rep='', header=True, index=True)
    # Can also return where it was written in case we want to log it
    return fname


def main():
    # TODO all of these should go in config file
    input_path = './chains'
    output_path = '.'
    exact = 'exact'
    primary_metric = 'mean'
    primary_diag = 'geweke'

    samplers, examples = find_traces(input_path, exclude=[exact])
    print 'found %d samplers and %d examples' % (len(samplers), len(examples))

    # Will remove this later when the list gets large
    print samplers
    print examples

    # TODO move the bulk of this outside of main
    # TODO use benchmark tools (bt) package and multiple chains for CIs 
    # start bt with loss_summary_table()
    # TODO build constants (or use bt) for these axis names
    # Pandas initializes these as NaN
    cols = pd.MultiIndex.from_product([STD_DIAGNOSTICS.keys(), examples],
                                      names=['diagnostic', 'example'])
    diagnostic_df = pd.DataFrame(index=samplers, columns=cols, dtype=float)
    cols = pd.MultiIndex.from_product([STD_METRICS.keys(), examples],
                                      names=['metric', 'example'])
    metric_df = pd.DataFrame(index=samplers, columns=cols, dtype=float)
    for example in examples:
        fname = build_trace_name(input_path, example, exact)
        # TODO consider using robust standardization fit on exact chain to make
        # metrics unitless
        exact_chain = load_chain(fname)
        for sampler in samplers:
            fname = build_trace_name(input_path, example, sampler)
            curr_chain = load_chain(fname)

            # compute diagnostics
            for diag_name, diag_f in STD_DIAGNOSTICS.iteritems():
                score = diag_f(curr_chain)
                diagnostic_df.loc[sampler, (diag_name, example)] = score

            # compute comparison against exact in param space
            for metric_name, metric_f in STD_METRICS.iteritems():
                score = metric_f(exact_chain, curr_chain)
                metric_df.loc[sampler, (metric_name, example)] = score

            # TODO future comparison of posterior predictive (esp KL)
    # TODO verify these do nan for missing like numpy
    metric_df_agg = metric_df.groupby(level=['metric']).mean()
    metric_df_agg.sort(columns=primary_metric, ascending=False, axis=0,
                       inplace=True)

    diag_df_agg = metric_df.groupby(level=['metric']).mean()
    diag_df_agg.sort(columns=primary_diag, ascending=False, axis=0,
                     inplace=True)

    # save out results to csv file. can pretty print export these in phase 5.
    dump_results(metric_df, output_path, 'metric')
    dump_results(diagnostic_df, output_path, 'diagnostic')
    dump_results(metric_df_agg, output_path, 'metric_agg')
    dump_results(diag_df_agg, output_path, 'diagnostic_agg')
    print 'done'

if __name__ == '__main__':
    main()
