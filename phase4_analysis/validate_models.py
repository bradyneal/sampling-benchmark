# Ryan Turner (turnerry@iro.umontreal.ca)
import sys
import scipy.stats as ss
import ConfigParser
import fileio as io


def main():
    assert(len(sys.argv) == 2)
    config_file = io.abspath2(sys.argv[1])

    config = ConfigParser.RawConfigParser()
    config.read(config_file)
    input_original = io.abspath2(config.get('phase1', 'output_path'))
    input_exact = io.abspath2(config.get('phase3', 'output_path'))
    exact_name = config.get('common', 'exact_name')
    csv_ext = config.get('common', 'csv_ext')
    sep = '_'

    _, examples, file_lookup = io.find_traces(input_exact, exact_name, csv_ext)
    for example in examples:
        original_chain, _ = example.rsplit(sep, 1)
        X_original = io.load_np(input_original, original_chain, csv_ext)
        assert(X_original.ndim == 2)
        D = X_original.shape[1]

        fname_exact, = file_lookup[(example, exact_name)]
        X_exact = io.load_np(input_exact, fname_exact, '')
        assert(X_exact.ndim == 2 and X_exact.shape[1] == D)

        print example
        for ii in xrange(D):
            stat, pval = ss.ttest_ind(X_original[:, ii], X_exact[:, ii],
                                      equal_var=False)
            print 'dim %d t: %f p = %f' % (ii, stat, pval)
            stat, pval = ss.levene(X_original[:, ii], X_exact[:, ii],
                                   center='median')
            print 'dim %d BF: %f p = %f' % (ii, stat, pval)
            stat, pval = ss.ks_2samp(X_original[:, ii], X_exact[:, ii])
            print 'dim %d KS: %f p = %f' % (ii, stat, pval)
        print '-' * 20
    print 'done'

if __name__ == '__main__':
    main()
